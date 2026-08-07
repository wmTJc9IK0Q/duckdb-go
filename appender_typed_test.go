package duckdb

import (
	"database/sql"
	"database/sql/driver"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/stretchr/testify/require"

	"github.com/duckdb/duckdb-go/v2/mapping"
)

func typedRowConn(t *testing.T, ddl ...string) (*sql.DB, driver.Conn) {
	t.Helper()
	db, err := sql.Open("duckdb", filepath.Join(t.TempDir(), "typed.db"))
	require.NoError(t, err)
	t.Cleanup(func() { db.Close() })
	for _, stmt := range ddl {
		_, err := db.Exec(stmt)
		require.NoError(t, err)
	}
	c, err := db.Conn(t.Context())
	require.NoError(t, err)
	t.Cleanup(func() { c.Close() })
	var raw driver.Conn
	require.NoError(t, c.Raw(func(dc any) error { raw = dc.(driver.Conn); return nil }))
	return db, raw
}

// TestAppenderTypedRow covers the typed path against the same column types the
// driver.Value path supports, including the chunk boundary, which is the only
// place BeginRow does anything beyond a bounds check.
func TestAppenderTypedRow(t *testing.T) {
	db, raw := typedRowConn(t, `CREATE TABLE t (
		i BIGINT, s VARCHAR, f DOUBLE, b BOOLEAN, ts TIMESTAMP, blb BLOB, small INTEGER)`)
	a, err := NewAppender(raw, "", "main", "t")
	require.NoError(t, err)

	ts := time.Date(2026, 2, 3, 4, 5, 6, 0, time.UTC)
	// One more than two full data chunks, so the appender crosses the boundary
	// twice and lands mid-chunk.
	rows := 2*GetDataChunkCapacity() + 1
	for i := range rows {
		require.NoError(t, a.BeginRow())
		require.NoError(t, AppendValue(a, 0, int64(i)))
		require.NoError(t, AppendValue(a, 1, "row-"+string(rune('a'+i%26))))
		require.NoError(t, AppendValue(a, 2, float64(i)/4))
		require.NoError(t, AppendValue(a, 3, i%2 == 0))
		require.NoError(t, AppendValue(a, 4, ts.Add(time.Duration(i)*time.Second)))
		require.NoError(t, AppendValue(a, 5, []byte{byte(i), 0x00, 0xff}))
		// int64 into an INTEGER column: the destination width comes from the
		// column, not from the Go type.
		require.NoError(t, AppendValue(a, 6, int64(i%1000)))
		a.EndRow()
	}
	require.NoError(t, a.Close())

	var count, nulls int
	require.NoError(t, db.QueryRow(`SELECT count(*), count(*) FILTER (WHERE i IS NULL) FROM t`).Scan(&count, &nulls))
	require.Equal(t, rows, count)
	require.Equal(t, 0, nulls)

	// Spot-check the last row, which sits just past the second chunk boundary.
	var (
		gotI     int64
		gotS     string
		gotF     float64
		gotB     bool
		gotTS    time.Time
		gotBlob  []byte
		gotSmall int32
	)
	require.NoError(t, db.QueryRow(`SELECT i, s, f, b, ts, blb, small FROM t ORDER BY i DESC LIMIT 1`).
		Scan(&gotI, &gotS, &gotF, &gotB, &gotTS, &gotBlob, &gotSmall))
	last := rows - 1
	require.Equal(t, int64(last), gotI)
	require.Equal(t, "row-"+string(rune('a'+last%26)), gotS)
	require.Equal(t, float64(last)/4, gotF)
	require.Equal(t, last%2 == 0, gotB)
	require.Equal(t, ts.Add(time.Duration(last)*time.Second).UTC(), gotTS.UTC())
	require.Equal(t, []byte{byte(last), 0x00, 0xff}, gotBlob)
	require.Equal(t, int32(last%1000), gotSmall)
}

// TestAppenderTypedNullAndErrors covers AppendNull and the two ways the typed
// setters reject a row.
func TestAppenderTypedNullAndErrors(t *testing.T) {
	db, raw := typedRowConn(t, `CREATE TABLE t (i BIGINT, s VARCHAR)`)
	a, err := NewAppender(raw, "", "main", "t")
	require.NoError(t, err)

	require.NoError(t, a.BeginRow())
	require.NoError(t, AppendValue(a, 0, int64(1)))
	require.NoError(t, AppendNull(a, 1))
	a.EndRow()

	// Out-of-range column indices are refused rather than corrupting memory.
	require.NoError(t, a.BeginRow())
	require.Error(t, AppendValue(a, 2, int64(1)))
	require.Error(t, AppendValue(a, -1, int64(1)))
	require.Error(t, AppendNull(a, 2))
	// A value the column cannot hold is refused too.
	require.Error(t, AppendValue(a, 0, struct{ X int }{1}))
	// The row was never committed, so its slot is reused by the next BeginRow.

	require.NoError(t, a.BeginRow())
	require.NoError(t, AppendValue(a, 0, int64(2)))
	require.NoError(t, AppendValue(a, 1, "two"))
	a.EndRow()

	require.NoError(t, a.Close())

	rows, err := db.Query(`SELECT i, s FROM t ORDER BY i`)
	require.NoError(t, err)
	defer rows.Close()
	var got []string
	for rows.Next() {
		var i int64
		var s *string
		require.NoError(t, rows.Scan(&i, &s))
		if s == nil {
			got = append(got, "NULL")
			continue
		}
		got = append(got, *s)
	}
	require.NoError(t, rows.Err())
	require.Equal(t, []string{"NULL", "two"}, got)

	require.Error(t, a.BeginRow(), "the typed path must refuse a closed appender")
	require.Error(t, AppendValue(a, 0, int64(3)))
	require.Error(t, AppendNull(a, 0))
}

// TestAppenderStringNotRetained is the guard on setBytes handing DuckDB a pointer
// into Go memory instead of a copy.
//
// duckdb_vector_assign_string_element_len copies into the vector's own string
// heap (StringVector::AddStringOrBlob -> StringHeap::AddBlob: memcpy into the
// inlined bytes below 13 bytes, memcpy into the vector's arena above). If that
// reading is wrong and DuckDB retains the pointer, the values are Go strings
// whose only reference is dropped here, so the GC is free to reuse the memory
// and the reads below see something other than what was written.
//
// The strings are built at runtime, are unique per row, and straddle the 12-byte
// inline boundary so both branches of AddBlob are exercised. The 13+ byte ones
// are the case that matters: those are the only ones DuckDB could store by
// pointer.
func TestAppenderStringNotRetained(t *testing.T) {
	db, raw := typedRowConn(t, `CREATE TABLE t (i BIGINT, s VARCHAR)`)
	a, err := NewAppender(raw, "", "main", "t")
	require.NoError(t, err)

	const rows = 4096
	want := func(i int) string {
		// Row i is i%40 + 1 characters long, spanning 1..40 bytes.
		b := make([]byte, i%40+1)
		for j := range b {
			b[j] = byte('a' + (i+j)%26)
		}
		return string(b)
	}
	for i := range rows {
		require.NoError(t, a.BeginRow())
		require.NoError(t, AppendValue(a, 0, int64(i)))
		// A fresh heap string per row, unreferenced the moment AppendValue
		// returns.
		require.NoError(t, AppendValue(a, 1, want(i)))
		a.EndRow()
	}

	// Drop every Go string written above, then churn the heap through the same
	// size classes, so a retained pointer would be reading overwritten memory by
	// the time the rows are read back.
	runtime.GC()
	churn := make([][]byte, 0, 4*rows)
	for i := range 4 * rows {
		b := make([]byte, i%40+1)
		for j := range b {
			b[j] = 0x5a
		}
		churn = append(churn, b)
	}
	runtime.GC()
	runtime.KeepAlive(churn)

	require.NoError(t, a.Close())

	got, err := db.Query(`SELECT i, s FROM t ORDER BY i`)
	require.NoError(t, err)
	defer got.Close()
	n := 0
	for got.Next() {
		var i int64
		var s string
		require.NoError(t, got.Scan(&i, &s))
		require.Equal(t, want(int(i)), s, "row %d came back changed: DuckDB kept the Go pointer", i)
		n++
	}
	require.NoError(t, got.Err())
	require.Equal(t, rows, n)
}

// TestVectorAssignStringCopies is the direct evidence for the setBytes change:
// duckdb_vector_assign_string_element_len copies the bytes before it returns, so
// handing it a pointer into Go memory is safe and the []byte(string) copy
// setBytes used to make first was redundant.
//
// Both branches of StringHeap::AddBlob are covered. Below 13 bytes the value is
// memcpy'd into the string_t itself; at 13 bytes and above the string_t holds a
// pointer, and that pointer is the only one that could be made to alias the
// caller's buffer.
func TestVectorAssignStringCopies(t *testing.T) {
	_, raw := typedRowConn(t, `CREATE TABLE t (s VARCHAR)`)
	a, err := NewAppender(raw, "", "main", "t")
	require.NoError(t, err)
	defer a.Close()
	vec := a.chunk.columns[0].vec

	heaped := []byte("a string comfortably past the inline limit")
	inlined := []byte("short")
	require.Greater(t, len(heaped), 12)
	require.LessOrEqual(t, len(inlined), 12)

	mapping.VectorAssignStringElementLen(vec, 0, heaped)
	mapping.VectorAssignStringElementLen(vec, 1, inlined)

	// Poison the source buffers. An aliasing DuckDB reads these bytes.
	want := []string{string(heaped), string(inlined)}
	for _, b := range [][]byte{heaped, inlined} {
		for i := range b {
			b[i] = 'z'
		}
	}
	// Non-vacuity: the poison must actually have landed on the memory whose
	// address DuckDB was handed, or the assertions below prove nothing.
	require.Equal(t, strings.Repeat("z", len(heaped)), string(heaped))
	require.Equal(t, strings.Repeat("z", len(inlined)), string(inlined))

	for row, w := range want {
		got, err := a.chunk.GetValue(0, row)
		require.NoError(t, err)
		require.Equal(t, w, got, "DuckDB read the caller's buffer instead of its own copy")
	}

	// And the same through setBytes' string path, whose backing array this test
	// deliberately mutates afterwards — the one thing unsafe.String forbids, and
	// exactly the poison a retained pointer would pick up.
	src := []byte("another value that cannot inline into a string_t")
	require.NoError(t, setBytes(&a.chunk.columns[0], 2, unsafe.String(&src[0], len(src))))
	original := string(src)
	for i := range src {
		src[i] = 'z'
	}
	require.Equal(t, strings.Repeat("z", len(src)), string(src))
	got, err := a.chunk.GetValue(0, 2)
	require.NoError(t, err)
	require.Equal(t, original, got)
}
