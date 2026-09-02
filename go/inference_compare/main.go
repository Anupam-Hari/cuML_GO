package main

import (
	"fmt"
	"log"
	"time"
	"path/filepath"
	"strings"
	"github.com/Anupam-Hari/cuml-go/go/internal/dataset"
	randomforest "github.com/Anupam-Hari/cuml-go/go/random_forest"
	knn "github.com/Anupam-Hari/cuml-go/go/knn"
	kmeans "github.com/Anupam-Hari/cuml-go/go/kmeans"
)

func TrainRF(
	X [][]float32,
	y []int,
) (
	rfGPU *randomforest.RandomForest,
	rfCPU *randomforest.RandomForest,
	err error,
) {

	// ---------------- GPU ----------------

	rfGPU, err = randomforest.New(
		randomforest.WithEstimators(50),
		randomforest.WithMaxDepth(20),
		randomforest.WithBackend(randomforest.BackendGPU),
	)
	if err != nil {
		return nil, nil, err
	}

	if err := rfGPU.Fit(X, y); err != nil {
		rfGPU.Close()
		return nil, nil, err
	}

	// ---------------- CPU ----------------

	rfCPU, err = randomforest.New(
		randomforest.WithEstimators(50),
		randomforest.WithMaxDepth(20),
		randomforest.WithBackend(randomforest.BackendCPU),
	)
	if err != nil {
		rfGPU.Close()
		return nil, nil, err
	}

	if err := rfCPU.Fit(X, y); err != nil {
		rfGPU.Close()
		rfCPU.Close()
		return nil, nil, err
	}

	return rfGPU, rfCPU, nil
}

func TrainKNN(
	X [][]float32,
	y []int,
) (
	knnGPU *knn.KNN,
	knnCPU *knn.KNN,
	err error,
) {

	knnGPU, err = knn.New(
		knn.WithK(10),
		knn.WithBackend(knn.BackendGPU),
	)
	if err != nil {
		return nil, nil, err
	}

	if err := knnGPU.Fit(X, y); err != nil {
		knnGPU.Close()
		return nil, nil, err
	}

	knnCPU, err = knn.New(
		knn.WithK(10),
		knn.WithBackend(knn.BackendCPU),
	)
	if err != nil {
		return nil, nil, err
	}

	if err := knnCPU.Fit(X, y); err != nil {
		knnCPU.Close()
		return nil, nil, err
	}

	if err != nil {
		knnGPU.Close()
		knnCPU.Close()

		return nil, nil, err
	}

	return knnGPU, knnCPU, nil
}

func TrainKMeans(
	X [][]float32,
) (
	kmeansGPU *kmeans.KMeans,
	kmeansCPU *kmeans.KMeans,
	err error,
) {

	kmeansGPU, err = kmeans.New(
		kmeans.WithBackend(kmeans.BackendGPU),
		kmeans.WithNClusters(5),
	)
	if err != nil {
		return nil, nil, err
	}

	if err := kmeansGPU.Fit(X); err != nil {
		kmeansGPU.Close()
		return nil, nil, err
	}

	kmeansCPU, err = kmeans.New(
		kmeans.WithBackend(kmeans.BackendCPU),
		kmeans.WithNClusters(5),
	)

	if err != nil { 
		kmeansGPU.Close() 
		return nil, nil, err 
	} 
	if err := kmeansCPU.Fit(X); err != nil { 
		kmeansGPU.Close() 
		kmeansCPU.Close() 
		return nil, nil, err 
	}
	if err != nil {
		kmeansGPU.Close()
		kmeansCPU.Close()

		return nil, nil, err
	}

	return kmeansGPU, kmeansCPU, nil
}

func containsModel(
	models []string,
	target string,
) bool {

	for _, model := range models {
		if model == target {
			return true
		}
	}

	return false
}

func loadDataset(
	config Config,
) ([][]float32, []int, error) {

	switch config.DatasetLoader {

	case "csv":
		return dataset.LoadCSV(
			config.DatasetPath,
			0,
		)

	case "fake":
		return dataset.LoadFakeCSV(
			config.DatasetPath,
			0,
		)

	default:
		return nil, nil, fmt.Errorf(
			"unsupported dataset loader: %s",
			config.DatasetLoader,
		)
	}
}

func main() {

	config := Config{
		ModelPath: "go/inference_compare/models/rf_model.tl",

		DatasetPath: "benchmark/data/processed_network_traffic.csv",

		DatasetLoader: "csv",

		PredictRows: 1000,

		Repeats: 1,

		WarmupRuns: 0,

		CPUCores: 16,

		SampleInterval: 20 * time.Millisecond,
	}

	const trainingRows = 500_000

	predictRows := []int{
		1000,
		10000,
		30000,
		50000,
		70000,
	}

	// -------------------------------------------------
	// SELECT MODELS TO RUN
	// -------------------------------------------------
	//
	// Comment/uncomment models as needed.
	//
	modelsToRun := []string{
		"random_forest",
		"knn",
		"kmeans",
	}

	X, y, err := loadDataset(config)
	
	if err != nil {
		log.Fatal(err)
	}

	if config.CPUCores > 0 {
		randomforest.SetCPUThreads(config.CPUCores)
	}

	var results []BenchmarkResult

	XTrain := X[:trainingRows]
	yTrain := y[:trainingRows]

	// -------------------------------------------------
	// Random Forest
	// -------------------------------------------------

	if containsModel(modelsToRun, "random_forest") {

		fmt.Println("Starting RF training...")

		rfGPU, rfCPU, err := TrainRF(
			XTrain,
			yTrain,
		)
		if err != nil {
			log.Fatal(err)
		}

		defer rfGPU.Close()
		defer rfCPU.Close()

		for _, rows := range predictRows {

			fmt.Printf("RF benchmark: %d rows\n", rows)

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

			benchmarkResults, err := BenchmarkRFInference(
				rfGPU,
				rfCPU,
				X_,
				y_,
				cfg,
			)
			if err != nil {
				log.Fatal(err)
			}

			results = append(
				results,
				benchmarkResults...,
			)
		}

		fmt.Println("All RF benchmarks complete")
	}

	// -------------------------------------------------
	// KNN
	// -------------------------------------------------

	if containsModel(modelsToRun, "knn") {

		fmt.Println("Starting KNN training...")

		knnGPU, knnCPU, err := TrainKNN(
			XTrain,
			yTrain,
		)
		if err != nil {
			log.Fatal(err)
		}

		defer knnGPU.Close()
		defer knnCPU.Close()

		for _, rows := range predictRows {

			fmt.Printf("KNN benchmark: %d rows\n", rows)

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

			benchmarkResults, err := BenchmarkKNNInference(
				knnGPU,
				knnCPU,
				X_,
				y_,
				cfg,
			)
			if err != nil {
				log.Fatal(err)
			}

			results = append(
				results,
				benchmarkResults...,
			)
		}

		fmt.Println("All KNN benchmarks complete")
	}

	// -------------------------------------------------
	// KMeans
	// -------------------------------------------------

	if containsModel(modelsToRun, "kmeans") {

		fmt.Println("Starting KMeans training...")

		kmeansGPU, kmeansCPU, err := TrainKMeans(
			XTrain,
		)
		if err != nil {
			log.Fatal(err)
		}

		defer kmeansGPU.Close()
		defer kmeansCPU.Close()

		for _, rows := range predictRows {

			fmt.Printf("KMeans benchmark: %d rows\n", rows)

			if rows > len(X) {
				log.Fatalf(
					"predict rows (%d) exceeds dataset size (%d)",
					rows,
					len(X),
				)
			}

			start := len(X) - rows

			X_ := X[start:]

			cfg := config
			cfg.PredictRows = rows

			benchmarkResults, err := BenchmarkKMeansInference(
				kmeansGPU,
				kmeansCPU,
				X_,
				cfg,
			)
			if err != nil {
				log.Fatal(err)
			}

			results = append(
				results,
				benchmarkResults...,
			)
		}

		fmt.Println("All KMeans benchmarks complete")
	}

	// -------------------------------------------------
	// Write results
	// -------------------------------------------------

	if len(results) == 0 {
		log.Fatal("no models selected for benchmarking")
	}

	timestamp := time.Now().Format("020106150405")

	resultsDir := "go/inference_compare/results"

	resultsFile := filepath.Join(
		resultsDir,
		fmt.Sprintf(
			"inference_results_%s.csv",
			timestamp,
		),
	)

	if err := WriteResultsCSV(
		resultsFile,
		results,
	); err != nil {
		log.Fatal(err)
	}

	summaryFile := strings.TrimSuffix(
		resultsFile,
		".csv",
	) + "_summary.csv"

	if err := WriteSummaryCSV(
		resultsFile,
		summaryFile,
	); err != nil {
		log.Fatal(err)
	}

	fmt.Printf(
		"\nResults written to %s\n",
		resultsFile,
	)

	fmt.Printf(
		"Summary written to %s\n",
		summaryFile,
	)

	comparePythonGoThroughput(summaryFile)
}

// func main() {
//     comparePythonGoThroughput(
//         "/home/anupam/projects/cuml-go/go/inference_compare/results/inference_results_010926172842_summary.csv",
//     )
// }