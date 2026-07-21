package main

import (
	"encoding/csv"
	"fmt"
	"os"
)

func WriteResultsCSV(
	filename string,
	results []BenchmarkResult,
) error {

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	header := []string{
		"Model",
		"TrainRows",
		"TestRows",
		"TrainTime(ms)",
		"PredictionThroughput(ops)",
		"TotalTime(ms)",
		"GPUAvg",
		"GPUPeak",
	}

	if err := writer.Write(header); err != nil {
		return err
	}

	for _, r := range results {

		record := []string{
			r.Model,
			fmt.Sprintf("%d", r.TrainRows),
			fmt.Sprintf("%d", r.TestRows),
			fmt.Sprintf("%.3f", r.TrainTimeMS),
			fmt.Sprintf("%.3f", r.PredictionThroughput),
			fmt.Sprintf("%.3f", r.TotalTimeMS),
			fmt.Sprintf("%.3f", r.GPUAvg),
			fmt.Sprintf("%.3f", r.GPUPeak),
		}

		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return writer.Error()
}