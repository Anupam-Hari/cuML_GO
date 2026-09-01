package dataset

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

const labelColumn = "Label"

var featureColumns = []string{
	"data_rate_bps",
	"sta_rxrate_score",
	"wtp_bandwidth_rx",
	"sta_bandwidth_tx",
	"sta_txrate",
	"sta_txrate_score",
	"snr",
	"sta_rxrate",
	"channel_utilization_percent",
}

func LoadCSV(path string, maxRows int) ([][]float32, []int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)

	rows, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	if len(rows) < 2 {
		return nil, nil, fmt.Errorf("empty dataset")
	}

	header := rows[0]

	// Find column indexes.
	columnIndexes := make(map[string]int, len(header))

	for i, column := range header {
		columnIndexes[column] = i
	}

	// Validate required columns.
	labelIdx, ok := columnIndexes[labelColumn]
	if !ok {
		return nil, nil, fmt.Errorf(
			"label column %q not found",
			labelColumn,
		)
	}

	featureIndexes := make([]int, len(featureColumns))

	for i, column := range featureColumns {
		idx, ok := columnIndexes[column]

		if !ok {
			return nil, nil, fmt.Errorf(
				"feature column %q not found",
				column,
			)
		}

		featureIndexes[i] = idx
	}

	// Determine number of rows to load.
	dataRows := rows[1:]

	if maxRows > 0 && maxRows < len(dataRows) {
		dataRows = dataRows[:maxRows]
	}

	X := make([][]float32, 0, len(dataRows))
	y := make([]int, 0, len(dataRows))

	for rowIndex, row := range dataRows {

		if len(row) != len(header) {
			return nil, nil, fmt.Errorf(
				"row %d has %d columns, expected %d",
				rowIndex+2,
				len(row),
				len(header),
			)
		}

		features := make([]float32, len(featureIndexes))

		for i, columnIndex := range featureIndexes {
			value, err := strconv.ParseFloat(row[columnIndex], 32)

			if err != nil {
				return nil, nil, fmt.Errorf(
					"invalid value %q in column %q at row %d: %w",
					row[columnIndex],
					featureColumns[i],
					rowIndex+2,
					err,
				)
			}

			features[i] = float32(value)
		}

		label, err := strconv.Atoi(row[labelIdx])
		if err != nil {
			return nil, nil, fmt.Errorf(
				"invalid label %q at row %d: %w",
				row[labelIdx],
				rowIndex+2,
				err,
			)
		}

		X = append(X, features)
		y = append(y, label)
	}

	return X, y, nil
}