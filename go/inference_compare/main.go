package main

import (
	"fmt"
	"log"
	"time"
	"path/filepath"
	"strings"
	"github.com/Anupam-Hari/cuml-go/go/internal/dataset"
	randomforest "github.com/Anupam-Hari/cuml-go/go/random_forest"
)

func main() {

	config := Config{
		ModelPath: "go/inference_compare/models/rf_model.tl",

		DatasetPath: "benchmark/data/processed_network_traffic.csv",

		PredictRows: 1000,

		Repeats: 20,

		WarmupRuns: 3,

		CPUCores: 0,

		SampleInterval: 20 * time.Millisecond,
	}


	predictRows := []int{
		1000,
		10000,
		20000,
		50000,
		80000,
	}

	maxRows := 0
	for _, rows := range predictRows {
		if rows > maxRows {
			maxRows = rows
		}
	}

	X, y, err := dataset.LoadCSV(
		config.DatasetPath,
		"is_malicious",
		maxRows,
	)
	if err != nil {
		log.Fatal(err)
	}

	rf, err := randomforest.Load(config.ModelPath)
	if err != nil {
		log.Fatal(err)
	}
	defer rf.Close()

	if config.CPUCores > 0 {
		randomforest.SetCPUThreads(config.CPUCores)
	}

	var results []BenchmarkResult

	for _, rows := range predictRows {

		if rows > len(X) {
			log.Fatalf(
				"predict rows (%d) exceeds dataset size (%d)",
				rows,
				len(X),
			)
		}

		start := len(X) - rows

		X_ := X[start:]
		y_ := y[start:]

		cfg := config
		cfg.PredictRows = rows

		for run := 0; run < config.Repeats; run++ {

			result, err := BenchmarkRFInference(
				rf,
				X_,
				y_,
				cfg,
			)
			if err != nil {
				log.Fatal(err)
			}

			result.Run = run

			results = append(results, result)
		}
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