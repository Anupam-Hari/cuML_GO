package matrix

import "fmt"

type Dense struct {
	Data []float32

	Rows int
	Cols int
}

func From2D(x [][]float32) (*Dense, error) {
	if len(x) == 0 {
		return nil, fmt.Errorf("empty matrix")
	}

	rows := len(x)
	cols := len(x[0])

	data := make([]float32, 0, rows*cols)

	for i := range x {
		if len(x[i]) != cols {
			return nil, fmt.Errorf("ragged matrix")
		}

		data = append(data, x[i]...)
	}

	return &Dense{
		Data: data,
		Rows: rows,
		Cols: cols,
	}, nil
}
