package main

import (
	"flag"
	"fmt"
	"log"
	"strconv"
	"strings"
	"time"

	"github.com/Anupam-Hari/cuml-go/go/internal/dataset"
)

var rowsFlag = flag.String(
	"rows",
	"-1",
	"Comma-separated dataset row counts (e.g. 1000,5000,10000,-1)",
)

var repeatsFlag = flag.Int(
	"repeats",
	1,
	"Number of benchmark repetitions",
)

func parseRows(s string) ([]int, error) {
	parts := strings.Split(s, ",")
	rows := make([]int, 0, len(parts))

	for _, p := range parts {
		n, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil {
			return nil, fmt.Errorf("invalid row count %q", p)
		}
		rows = append(rows, n)
	}

	return rows, nil
}

func printResult(r BenchmarkResult) {
	fmt.Printf("\n=== %s ===\n", r.Model)
	fmt.Printf("Train Rows            : %d\n", r.TrainRows)
	fmt.Printf("Test Rows             : %d\n", r.TestRows)
	fmt.Printf("Training Time         : %.6f ms\n", r.TrainTimeMS)
	fmt.Printf("Accuracy              : %.2f%%\n", r.Accuracy*100)

	fmt.Printf("\nGPU Inference\n")
	fmt.Printf("Prediction Time       : %.6f ms\n", r.GPUPredictionTimeMS)
	fmt.Printf("Throughput            : %.6f samples/sec\n", r.GPUThroughput)

	fmt.Printf("\nCPU Inference\n")
	fmt.Printf("Prediction Time       : %.6f ms\n", r.CPUPredictionTimeMS)
	fmt.Printf("Throughput            : %.6f samples/sec\n", r.CPUThroughput)
}

func main() {
	flag.Parse()

	rowSizes, err := parseRows(*rowsFlag)
	if err != nil {
		log.Fatal(err)
	}

	maxRows := 0
	loadAll := false

	for _, r := range rowSizes {
		if r == -1 {
			loadAll = true
			break
		}

		if r > maxRows {
			maxRows = r
		}
	}

	rowsToLoad := maxRows
	if loadAll {
		rowsToLoad = -1
	}

	// Load the complete dataset once.
	X, y, err := dataset.LoadCSV(
		"benchmark/data/processed_network_traffic.csv",
		"is_malicious",
		rowsToLoad,
	)
	if err != nil {
		log.Fatal(err)
	}

	fullRows := len(X)
	fullCols := len(X[0])

	fmt.Printf("Dataset Loaded\n")
	fmt.Printf("Total Rows : %d\n", fullRows)
	fmt.Printf("Cols       : %d\n", fullCols)

	allResults := []BenchmarkResult{}

	for _, maxRows := range rowSizes {

		rows := maxRows
		if rows < 0 || rows > fullRows {
			rows = fullRows
		}

		ds := Dataset{
			X:    X[:rows],
			Y:    y[:rows],
			Rows: rows,
			Cols: fullCols,
		}

		for run := 1; run <= *repeatsFlag; run++ {

			fmt.Printf("\n==============================\n")
			fmt.Printf("Run %d/%d\n", run, *repeatsFlag)
			fmt.Printf("Running benchmark for %d rows\n", rows)
			fmt.Printf("==============================\n")

			rf, err := BenchmarkRFInference(ds)
			if err != nil {
				log.Fatal(err)
			}

			rf.Run = run

			allResults = append(allResults, rf)

			printResult(rf)

			// rf, err := BenchmarkRandomForest(ds)
			// if err != nil {
			// 	log.Fatal(err)
			// }
			// rf.Run = run
			// allResults = append(allResults, rf)
			// printResult(rf)

			// rfCPU, err := BenchmarkGoLearnRandomForest(ds)
			// if err != nil {
			// 	log.Fatal(err)
			// }
			// rfCPU.Run = run
			// allResults = append(allResults, rfCPU)
			// printResult(rfCPU)

			// knn, err := BenchmarkKNN(ds)
			// if err != nil {
			// 	log.Fatal(err)
			// }
			// knn.Run = run
			// allResults = append(allResults, knn)
			// printResult(knn)

			// km, err := BenchmarkKMeans(ds)
			// if err != nil {
			// 	log.Fatal(err)
			// }
			// km.Run = run
			// allResults = append(allResults, km)
			// printResult(km)
		}
	}

	timestamp := time.Now().Format("020106150405")
	filename := fmt.Sprintf(
		"go/results/go_run_cpugpu_%s.csv",
		timestamp,
	)

	if err := WriteResultsCSV(filename, allResults); err != nil {
		log.Fatal(err)
	}

	summaryFile := strings.TrimSuffix(filename, ".csv") + "_summary.csv"

	if err := WriteSummaryCSV(filename, summaryFile); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("\nResults written to %s\n", filename)
	fmt.Printf("Summary written to %s\n", summaryFile)
}