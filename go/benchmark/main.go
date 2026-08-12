package main

import (
	"flag"
	"fmt"
	"log"
	"strconv"
	"strings"
	"time"

	benchmarkutil "github.com/Anupam-Hari/cuml-go/go/internal/benchmark"
	"github.com/Anupam-Hari/cuml-go/go/internal/dataset"
)

const TRAIN_ROWS = 1_000_000

var rowsFlag = flag.String(
	"rows",
	"1000,10000,50000,80000",
	"Comma-separated prediction row counts",
)

var repeatsFlag = flag.Int(
	"repeats",
	5,
	"Number of inference repetitions",
)

func parseRows(s string) ([]int, error) {
	parts := strings.Split(s, ",")

	rows := make([]int, 0, len(parts))

	for _, p := range parts {
		n, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil {
			return nil, fmt.Errorf(
				"invalid row count %q",
				p,
			)
		}

		if n <= 0 {
			return nil, fmt.Errorf(
				"prediction row count must be greater than 0, got %d",
				n,
			)
		}

		rows = append(rows, n)
	}

	return rows, nil
}

func main() {
	flag.Parse()

	// ---------------------------------------------------------
	// Prediction sample sizes
	// ---------------------------------------------------------

	predictSizes, err := parseRows(*rowsFlag)
	if err != nil {
		log.Fatal(err)
	}

	// ---------------------------------------------------------
	// Determine how many rows need to be loaded.
	//
	// We always need 1M training rows.
	// We also need enough rows for the largest prediction size.
	// ---------------------------------------------------------

	maxPredictRows := 0

	for _, rows := range predictSizes {
		if rows > maxPredictRows {
			maxPredictRows = rows
		}
	}

	rowsToLoad := TRAIN_ROWS

	if maxPredictRows > rowsToLoad {
		rowsToLoad = maxPredictRows
	}

	// ---------------------------------------------------------
	// Load dataset once
	// ---------------------------------------------------------

	X, y, err := dataset.LoadCSV(
		"benchmark/data/processed_network_traffic.csv",
		"is_malicious",
		rowsToLoad,
	)
	if err != nil {
		log.Fatal(err)
	}

	if len(X) == 0 {
		log.Fatal("dataset is empty")
	}

	fullRows := len(X)
	fullCols := len(X[0])

	if fullRows < TRAIN_ROWS {
		log.Fatalf(
			"dataset has %d rows, but %d training rows are required",
			fullRows,
			TRAIN_ROWS,
		)
	}

	fmt.Println("Dataset Loaded")
	fmt.Printf("Rows : %d\n", fullRows)
	fmt.Printf("Cols : %d\n", fullCols)

	fmt.Printf(
		"Training Rows : %d\n",
		TRAIN_ROWS,
	)

	fmt.Printf(
		"Prediction Sizes : %v\n",
		predictSizes,
	)

	fmt.Printf(
		"Repeats : %d\n",
		*repeatsFlag,
	)

	// ---------------------------------------------------------
	// Create fixed training dataset.
	//
	// This is passed to each model benchmark.
	// Each benchmark function trains its model ONCE.
	// ---------------------------------------------------------

	trainDataset := benchmarkutil.Dataset{
		X:    X[:TRAIN_ROWS],
		Y:    y[:TRAIN_ROWS],
		Rows: TRAIN_ROWS,
		Cols: fullCols,
	}

	// ---------------------------------------------------------
	// Run benchmarks
	// ---------------------------------------------------------

	allResults := []BenchmarkResult{}

	// Random Forest
	rfResults, err := BenchmarkRandomForest(
		trainDataset,
		X,
		y,
		predictSizes,
		*repeatsFlag,
	)
	if err != nil {
		log.Fatal(err)
	}

	allResults = append(
		allResults,
		rfResults...,
	)

	// KNN
	knnResults, err := BenchmarkKNN(
		trainDataset,
		X,
		y,
		predictSizes,
		*repeatsFlag,
	)
	if err != nil {
		log.Fatal(err)
	}

	allResults = append(
		allResults,
		knnResults...,
	)

	// KMeans
	kmeansResults, err := BenchmarkKMeans(
		trainDataset,
		X,
		y,
		predictSizes,
		*repeatsFlag,
	)

	if err != nil {
		log.Fatal(err)
	}

	allResults = append(
		allResults,
		kmeansResults...,
	)

	// ---------------------------------------------------------
	// Write results
	// ---------------------------------------------------------

	timestamp := time.Now().Format(
		"020106150405",
	)

	filename := fmt.Sprintf(
		"go/benchmark/results/go_gpu_inference_%s.csv",
		timestamp,
	)

	if err := WriteResultsCSV(
		filename,
		allResults,
	); err != nil {
		log.Fatal(err)
	}

	fmt.Printf(
		"\nResults written to %s\n",
		filename,
	)

}
