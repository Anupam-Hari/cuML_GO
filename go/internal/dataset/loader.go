package dataset

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

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

func LoadCSV(path string, labelColumn string, maxRows int) ([][]float32, []int, error) {
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

	labelIdx := -1
	for i, h := range header {
		if h == labelColumn {
			labelIdx = i
			break
		}
	}

	otherLabelIdx := -1

	for i, h := range header {
		switch {
		case labelColumn == "is_malicious" && h == "attack_type":
			otherLabelIdx = i
		case labelColumn == "attack_type" && h == "is_malicious":
			otherLabelIdx = i
		}
	}

	if labelIdx == -1 {
		return nil, nil, fmt.Errorf("label column %q not found", labelColumn)
	}

	var X [][]float32
	var y []int

	for i, row := range rows[1:] {

		if maxRows > 0 && i >= maxRows {
			break
		}

		features := make([]float32, 0, len(row)-2)

		for j, val := range row {

			if j == labelIdx {
				switch val {
				case "True":
					y = append(y, 1)
				case "False":
					y = append(y, 0)
				default:
					label, err := strconv.Atoi(val)
					if err != nil {
						return nil, nil, err
					}
					y = append(y, label)
				}
				continue
			}

			if j == otherLabelIdx {
				continue
			}

			f, err := parseFeature(val)
			if err != nil {
				return nil, nil, err
			}

			features = append(features, float32(f))
		}

		X = append(X, features)
	}

	return X, y, nil
}