package main

import (
	"flag"
	"fmt"
	"log"
	"strconv"
	"strings"
	// "time"

	"github.com/Anupam-Hari/cuml-go/go/internal/dataset"
)

var rowsFlag = flag.String(
	"rows",
	"-1",
	"Comma-separated dataset row counts (e.g. 1000,5000,10000,-1)",
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
	fmt.Printf("Training Time         : %.3f ms\n", r.TrainTimeMS)
	fmt.Printf("Total Time            : %.3f ms\n", r.TotalTimeMS)
	fmt.Printf("Accuracy              : %.2f%%\n", r.Accuracy*100)
	fmt.Printf("Prediction Throughput : %.2f samples/sec\n",
		r.PredictionThroughput)
}

func main() {
	flag.Parse()

	rowSizes, err := parseRows(*rowsFlag)
	if err != nil {
		log.Fatal(err)
	}

	allResults := []BenchmarkResult{}

	for _, maxRows := range rowSizes {
		fmt.Printf("\n==============================\n")
		fmt.Printf("Running benchmark for %d rows\n", maxRows)
		fmt.Printf("==============================\n")

		X, y, err := dataset.LoadCSV(
			"benchmark/data/processed_network_traffic.csv",
			"is_malicious",
			maxRows,
		)
		if err != nil {
			log.Fatal(err)
		}

		ds := Dataset{
			X:    X,
			Y:    y,
			Rows: len(X),
			Cols: len(X[0]),
		}

		fmt.Printf("Dataset Loaded\n")
		fmt.Printf("Rows : %d\n", ds.Rows)
		fmt.Printf("Cols : %d\n", ds.Cols)

		rf, err := BenchmarkRandomForest(ds)
		if err != nil {
			log.Fatal(err)
		}
		allResults = append(allResults, rf)
		printResult(rf)

		knn, err := BenchmarkKNN(ds)
		if err != nil {
			log.Fatal(err)
		}
		allResults = append(allResults, knn)
		printResult(knn)

		km, err := BenchmarkKMeans(ds)
		if err != nil {
			log.Fatal(err)
		}
		allResults = append(allResults, km)
		printResult(km)
	}

	// timestamp := time.Now().Format("020106150405")
	// filename := fmt.Sprintf(
	// 	"go/results/go_run_%s.csv",
	// 	timestamp,
	// )

	// if err := WriteResultsCSV(filename, allResults); err != nil {
	// 	log.Fatal(err)
	// }

	// fmt.Printf("\nResults written to %s\n", filename)
}