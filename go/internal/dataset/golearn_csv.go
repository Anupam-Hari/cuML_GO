package dataset

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

func WriteGoLearnCSV(path string, X [][]float32, y []int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	header := make([]string, len(X[0])+1)
	for i := range X[0] {
		header[i] = fmt.Sprintf("feature_%d", i)
	}
	header[len(header)-1] = "class"

	if err := w.Write(header); err != nil {
		return err
	}

	for i := range X {
		row := make([]string, len(X[i])+1)

		for j, v := range X[i] {
			row[j] = strconv.FormatFloat(float64(v), 'f', -1, 32)
		}

		row[len(row)-1] = strconv.Itoa(y[i])

		if err := w.Write(row); err != nil {
			return err
		}
	}

	return w.Error()
}