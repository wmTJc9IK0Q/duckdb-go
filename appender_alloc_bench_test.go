package duckdb

import (
	"database/sql"
	"database/sql/driver"
	"path/filepath"
	"testing"
	"time"

	"github.com/duckdb/duckdb-go/v2/mapping"
)

// The per-row allocations these benchmarks price are per DATA POINT on an ingest
// path, so no amount of batching amortises them. They are reported per row.

const benchSeriesInsert = `INSERT INTO head.metric_numeric_series_wal BY NAME
(SELECT * FROM appended_data src
 WHERE NOT EXISTS (SELECT 1 FROM head.metric_numeric_series_wal t WHERE t.series_id = src.series_id))`

// benchResourceAttrs and benchAttrs are the shape of the JSON a real series row
// carries: one long value that cannot inline into a string_t, one short one.
const (
	benchResourceAttrs = `{"service.name":"checkoutservice","service.namespace":"shop","host.name":"ip-10-0-3-114","cloud.region":"us-east-1","k8s.pod.name":"checkoutservice-7d9f8c-abcde"}`
	benchAttrs         = `{"le":"0.5","status_code":"200","method":"GET","endpoint":"/api/v1/items"}`
)

func benchRawConn(b *testing.B, ddl ...string) driver.Conn {
	b.Helper()
	db, err := sql.Open("duckdb", filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { db.Close() })
	for _, stmt := range ddl {
		if _, err := db.Exec(stmt); err != nil {
			b.Fatal(err)
		}
	}
	c, err := db.Conn(b.Context())
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { c.Close() })
	var raw driver.Conn
	if err := c.Raw(func(dc any) error { raw = dc.(driver.Conn); return nil }); err != nil {
		b.Fatal(err)
	}
	return raw
}

func benchSampleAppender(b *testing.B) *Appender {
	b.Helper()
	raw := benchRawConn(b, `CREATE TABLE t (series_id BIGINT, time TIMESTAMP, value DOUBLE)`)
	a, err := NewAppender(raw, "", "main", "t")
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { a.Close() })
	return a
}

func benchSeriesAppender(b *testing.B) *Appender {
	b.Helper()
	raw := benchRawConn(b,
		`CREATE SCHEMA head`,
		`CREATE TABLE head.metric_numeric_series_wal (
			series_id BIGINT NOT NULL PRIMARY KEY, metric_name VARCHAR NOT NULL,
			metric_type VARCHAR NOT NULL, unit VARCHAR, description VARCHAR,
			agg_temporality VARCHAR, is_monotonic BOOLEAN NOT NULL,
			resource_attributes VARCHAR, scope_name VARCHAR, scope_version VARCHAR,
			scope_attributes VARCHAR, attributes VARCHAR,
			min_timestamp TIMESTAMP NOT NULL)`)
	varchar, _ := NewTypeInfo(TYPE_VARCHAR)
	bigint, _ := NewTypeInfo(TYPE_BIGINT)
	boolean, _ := NewTypeInfo(TYPE_BOOLEAN)
	tstamp, _ := NewTypeInfo(TYPE_TIMESTAMP)
	a, err := NewQueryAppender(raw, benchSeriesInsert, "",
		[]TypeInfo{bigint, varchar, varchar, varchar, varchar, varchar, boolean,
			varchar, varchar, varchar, varchar, varchar, tstamp},
		[]string{"series_id", "metric_name", "metric_type", "unit", "description",
			"agg_temporality", "is_monotonic", "resource_attributes", "scope_name",
			"scope_version", "scope_attributes", "attributes", "min_timestamp"})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { a.Close() })
	return a
}

// BenchmarkAllocSampleRow prices one (BIGINT, TIMESTAMP, DOUBLE) sample row,
// which is the row microtel appends once per ingested data point.
func BenchmarkAllocSampleRow(b *testing.B) {
	ts := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	b.Run("AppendRow", func(b *testing.B) {
		a := benchSampleAppender(b)
		b.ReportAllocs()
		b.ResetTimer()
		for i := range b.N {
			if err := a.AppendRow(int64(i), ts, float64(i)); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Typed", func(b *testing.B) {
		a := benchSampleAppender(b)
		b.ReportAllocs()
		b.ResetTimer()
		for i := range b.N {
			if err := a.BeginRow(); err != nil {
				b.Fatal(err)
			}
			if err := AppendValue(a, 0, int64(i)); err != nil {
				b.Fatal(err)
			}
			if err := AppendValue(a, 1, ts); err != nil {
				b.Fatal(err)
			}
			if err := AppendValue(a, 2, float64(i)); err != nil {
				b.Fatal(err)
			}
			a.EndRow()
		}
	})
}

// BenchmarkAllocSeriesRow prices one series row: 13 values of which 10 are
// VARCHAR, each of which pays a []byte(string) copy inside setBytes.
func BenchmarkAllocSeriesRow(b *testing.B) {
	ts := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	b.Run("AppendRow", func(b *testing.B) {
		a := benchSeriesAppender(b)
		b.ReportAllocs()
		b.ResetTimer()
		for i := range b.N {
			err := a.AppendRow(int64(i), "http_server_duration_seconds", "gauge", "s",
				"desc", "delta", true, benchResourceAttrs, "scope", "1.0",
				`{"library":"otel-go"}`, benchAttrs, ts)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Typed", func(b *testing.B) {
		a := benchSeriesAppender(b)
		b.ReportAllocs()
		b.ResetTimer()
		for i := range b.N {
			if err := a.BeginRow(); err != nil {
				b.Fatal(err)
			}
			err := AppendValue(a, 0, int64(i))
			chk(b, err)
			chk(b, AppendValue(a, 1, "http_server_duration_seconds"))
			chk(b, AppendValue(a, 2, "gauge"))
			chk(b, AppendValue(a, 3, "s"))
			chk(b, AppendValue(a, 4, "desc"))
			chk(b, AppendValue(a, 5, "delta"))
			chk(b, AppendValue(a, 6, true))
			chk(b, AppendValue(a, 7, benchResourceAttrs))
			chk(b, AppendValue(a, 8, "scope"))
			chk(b, AppendValue(a, 9, "1.0"))
			chk(b, AppendValue(a, 10, `{"library":"otel-go"}`))
			chk(b, AppendValue(a, 11, benchAttrs))
			chk(b, AppendValue(a, 12, ts))
			a.EndRow()
		}
	})
}

// BenchmarkAllocStringAssign isolates the two ways the bindings can hand a Go
// string to a DuckDB vector: with a []byte(string) copy, and without one.
// Both call duckdb_vector_assign_string_element_len, so the only difference
// measured is the Go-side copy.
func BenchmarkAllocStringAssign(b *testing.B) {
	raw := benchRawConn(b, `CREATE TABLE t (s VARCHAR)`)
	a, err := NewAppender(raw, "", "main", "t")
	if err != nil {
		b.Fatal(err)
	}
	defer a.Close()
	vec := a.chunk.columns[0].vec
	capacity := mapping.IdxT(GetDataChunkCapacity())

	b.Run("LenWithByteConv", func(b *testing.B) {
		b.ReportAllocs()
		for i := range b.N {
			mapping.VectorAssignStringElementLen(vec, mapping.IdxT(i)%capacity, []byte(benchResourceAttrs))
		}
	})
	b.Run("ZeroCopyString", func(b *testing.B) {
		b.ReportAllocs()
		for i := range b.N {
			mapping.VectorAssignStringElement(vec, mapping.IdxT(i)%capacity, benchResourceAttrs)
		}
	})
}

// chk keeps the 13-column typed row readable without an error check between
// every line. It costs a call and a nil compare per column, which is the same
// branch the inline form would have.
func chk(b *testing.B, err error) {
	if err != nil {
		b.Fatal(err)
	}
}
