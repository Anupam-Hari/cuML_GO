package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
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
		"PredictRows",
		"Backend",
		"Runs",
		"AvgAccuracy",
		"AvgPredictionTime(ms)",
		"AvgThroughput(samples/sec)",
		"AvgCPU(%)",
	}

	if err := writer.Write(header); err != nil {
		return err
	}

	for _, r := range results {

		record := []string{
			r.Model,
			strconv.Itoa(r.PredictRows),
			r.Backend,
			strconv.Itoa(r.Runs),
			fmt.Sprintf("%.6f", r.AvgAccuracy),
			fmt.Sprintf("%.6f", r.AvgPredictionTimeMS),
			fmt.Sprintf("%.6f", r.AvgThroughput),
			fmt.Sprintf("%.6f", r.AvgCPU),
		}

		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return writer.Error()
}
