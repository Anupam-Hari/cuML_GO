package main

import (
	"fmt"
	"log"
	"time"
	"path/filepath"
	"strings"
)

func main() {

	config := Config{
		ModelPath: "go/inference_compare/models/rf_model.tl",

		DatasetPath: "benchmark/data/processed_network_traffic.csv",

		PredictRows: 1000,

		Repeats: 3,

		WarmupRuns: 1,

		CPUCores: 8,

		SampleInterval: 20 * time.Millisecond,
	}

	predictRows := []int{
		1000,
	}

	var results []BenchmarkResult

	for _, rows := range predictRows {

		cfg := config
		cfg.PredictRows = rows// or see note below

		result, err := BenchmarkRFInference(cfg)
		if err != nil {
			log.Fatal(err)
		}

		results = append(results, result)

		fmt.Printf(
			"Completed benchmark for %d prediction samples\n",
			rows,
		)
	}

	timestamp := time.Now().Format("020106150405")

	resultsDir := "go/inference_compare/results"

	resultsFile := filepath.Join(
		resultsDir,
		fmt.Sprintf("inference_results_%s.csv", timestamp),
	)

	if err := WriteResultsCSV(resultsFile, results); err != nil {
		log.Fatal(err)
	}

	summaryFile := strings.TrimSuffix(resultsFile, ".csv") + "_summary.csv"

	if err := WriteSummaryCSV(resultsFile, summaryFile); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("\nResults written to %s\n", resultsFile)
	fmt.Printf("Summary written to %s\n", summaryFile)

	for _, r := range results {

		fmt.Printf(
			"%6d rows | GPU %.2f M samples/s | CPU %.2f M samples/s\n",
			r.PredictRows,
			r.GPUThroughput/1e6,
			r.CPUThroughput/1e6,
		)
	}
}