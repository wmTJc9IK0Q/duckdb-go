package duckdb

import (
	"errors"

	"github.com/duckdb/duckdb-go/v2/mapping"
)

// The typed row API is AppendRow without driver.Value.
//
// AppendRow's signature is `...driver.Value`, so every row allocates a variadic
// slice, and every value in it is converted to an interface whose data pointer
// escapes into an indirect setter call — which is what forces the box onto the
// heap. That cost is per ROW, so batching never amortises it, and on an ingest
// path a row is a data point.
//
// A caller that knows its column types at compile time writes them straight into
// the appender's data chunk instead:
//
//	if err := a.BeginRow(); err != nil {
//		return err
//	}
//	if err := duckdb.AppendValue(a, 0, seriesID); err != nil {
//		return err
//	}
//	// ... remaining columns ...
//	a.EndRow()
//
// AppendValue is a function rather than a method because Go has no generic
// methods.
//
// Two things AppendRow does that this does not. It checks that a row supplies
// exactly one value per column: here a column left unset keeps whatever the
// previous occupant of that chunk slot wrote, so a caller must set every column
// of every row. And it frames the row: BeginRow must precede the setters,
// because it is what makes room when the current chunk fills up, and EndRow must
// follow them. A row whose setters failed is abandoned by not calling EndRow,
// and the next BeginRow reuses its slot.

// BeginRow opens a row for AppendValue, appending the current data chunk first
// if it is full.
func (a *Appender) BeginRow() error {
	if a.closed {
		return getError(errAppenderAppendAfterClose, nil)
	}
	if a.rowCount == a.chunkCap {
		if err := a.appendDataChunk(); err != nil {
			return getError(errAppenderAppendRow, err)
		}
	}
	return nil
}

// EndRow commits the row opened by BeginRow.
func (a *Appender) EndRow() {
	a.rowCount++
}

// AppendValue writes val to column colIdx of the row that BeginRow opened.
//
// It accepts the same Go types per column type as AppendRow, and costs no
// allocation for any value whose type the compiler can see through: the switch
// that picks the destination setter is over the column's DuckDB type, not over
// the value's dynamic type, so val never needs a heap box to be inspected.
// Values are ignored for columns the chunk does not project.
func AppendValue[T any](a *Appender, colIdx int, val T) error {
	if a.closed {
		return getError(errAppenderAppendAfterClose, nil)
	}

	colIdx, err := a.chunk.verifyAndRewriteColIdx(colIdx)
	if err != nil {
		if errors.Is(err, errUnprojectedColumn) {
			return nil
		}
		return getError(errAPI, err)
	}

	// Deliberately not wrapped with setValueError: handing val to an error
	// constructor makes it escape on every call, not only on the failing one,
	// which is the allocation this path exists to avoid.
	return setVectorVal(&a.chunk.columns[colIdx], mapping.IdxT(a.rowCount), val)
}

// AppendNull writes SQL NULL to column colIdx of the row that BeginRow opened.
// It is the typed path's equivalent of a nil driver.Value, which AppendValue
// cannot express.
func AppendNull(a *Appender, colIdx int) error {
	if a.closed {
		return getError(errAppenderAppendAfterClose, nil)
	}

	colIdx, err := a.chunk.verifyAndRewriteColIdx(colIdx)
	if err != nil {
		if errors.Is(err, errUnprojectedColumn) {
			return nil
		}
		return getError(errAPI, err)
	}

	a.chunk.columns[colIdx].setNull(mapping.IdxT(a.rowCount))
	return nil
}
