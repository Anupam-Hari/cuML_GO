package main

import (
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/Anupam-Hari/cuml-go/go/internal/dataset"
)

var maxRows = flag.Int(
	"rows",
	-1,
	"Maximum number of dataset rows",
)

func printResult(r BenchmarkResult) {
	fmt.Printf("\n=== %s ===\n", r.Model)
	fmt.Printf("Train Rows            : %d\n", r.TrainRows)
	fmt.Printf("Test Rows             : %d\n", r.TestRows)
	fmt.Printf("Training Time         : %.3f ms\n", r.TrainTimeMS)
	fmt.Printf("Total Time            : %.3f ms\n", r.TotalTimeMS)
	fmt.Printf("Prediction Throughput : %.2f samples/sec\n",
		r.PredictionThroughput)
}

func main() {

	flag.Parse()

	X, y, err := dataset.LoadCSV(
		"benchmark/data/processed_network_traffic.csv",
		"is_malicious",
		*maxRows,
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

	results := []BenchmarkResult{}

	fmt.Printf("Dataset Loaded\n")
	fmt.Printf("Rows : %d\n", ds.Rows)
	fmt.Printf("Cols : %d\n", ds.Cols)

	rf, err := BenchmarkRandomForest(ds)
	if err != nil {
		log.Fatal(err)
	}
	results = append(results, rf)
	printResult(rf)

	knn, err := BenchmarkKNN(ds)
	if err != nil {
		log.Fatal(err)
	}
	results = append(results, rf)
	printResult(knn)

	km, err := BenchmarkKMeans(ds)
	if err != nil {
		log.Fatal(err)
	}
	results = append(results, rf)
	printResult(km)

	timestamp := time.Now().Format("020106150405")
	filename := fmt.Sprintf(
		"go/results/go_run_%s.csv",
		timestamp,
	)

	if err := WriteResultsCSV(filename,	results); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("\nResults written to %s\n", filename)
}