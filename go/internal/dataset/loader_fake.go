package dataset

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

var fakeFeatureColumns = []string{
	"log_duration",
	"log_orig_bytes",
	"log_resp_bytes",
	"log_missed_bytes",
	"log_orig_pkts",
	"log_orig_ip_bytes",
	"log_resp_pkts",
	"log_resp_ip_bytes",
	"id.orig_p",
}

func parseFeature(value string) (float32, error) {
	switch value {
	case "True":
		return 1.0, nil
	case "False":
		return 0.0, nil
	}

	f, err := strconv.ParseFloat(value, 32)
	if err != nil {
		return 0, err
	}

	return float32(f), nil
}

func LoadFakeCSV(path string, labelColumn string, maxRows int) ([][]float32, []int, error) {
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

	// Build column-name -> index mapping.
	columnIdx := make(map[string]int, len(header))
	for i, h := range header {
		columnIdx[h] = i
	}

	// Verify all required fake features exist.
	featureIdx := make([]int, len(fakeFeatureColumns))

	for i, feature := range fakeFeatureColumns {
		idx, ok := columnIdx[feature]
		if !ok {
			return nil, nil, fmt.Errorf(
				"feature column %q not found",
				feature,
			)
		}

		featureIdx[i] = idx
	}

	// Find target column.
	labelIdx, ok := columnIdx[labelColumn]
	if !ok {
		return nil, nil, fmt.Errorf(
			"label column %q not found",
			labelColumn,
		)
	}

	var X [][]float32
	var y []int

	for i, row := range rows[1:] {

		if maxRows > 0 && i >= maxRows {
			break
		}

		// Make exactly 9 features, in exactly the same order
		// as fakeFeatureColumns.
		features := make([]float32, len(featureIdx))

		for j, idx := range featureIdx {
			if idx >= len(row) {
				return nil, nil, fmt.Errorf(
					"row %d is missing column %q",
					i+1,
					fakeFeatureColumns[j],
				)
			}

			f, err := parseFeature(row[idx])
			if err != nil {
				return nil, nil, fmt.Errorf(
					"row %d, feature %q: %w",
					i+1,
					fakeFeatureColumns[j],
					err,
				)
			}

			features[j] = f
		}

		// Parse target.
		if labelIdx >= len(row) {
			return nil, nil, fmt.Errorf(
				"row %d is missing label column %q",
				i+1,
				labelColumn,
			)
		}

		switch row[labelIdx] {
		case "True":
			y = append(y, 1)
		case "False":
			y = append(y, 0)
		default:
			label, err := strconv.Atoi(row[labelIdx])
			if err != nil {
				return nil, nil, fmt.Errorf(
					"row %d, label %q: %w",
					i+1,
					labelColumn,
					err,
				)
			}

			y = append(y, label)
		}

		X = append(X, features)
	}

	return X, y, nil
}