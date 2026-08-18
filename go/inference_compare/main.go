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

func TrainKNN(
	X [][]float32,
	y []int,
) (
	knnGPU *knn.KNN,
	knnCPU *knn.KNN,
	knnONNX *knn.KNN,
	err error,
) {

	knnGPU, err = knn.New(
		knn.WithK(5),
		knn.WithBackend(knn.BackendGPU),
	)
	if err != nil {
		return nil, nil, nil, err
	}

	if err := knnGPU.Fit(X, y); err != nil {
		knnGPU.Close()
		return nil, nil, nil, err
	}

	knnCPU, err = knn.New(
		knn.WithK(5),
		knn.WithBackend(knn.BackendCPU),
	)
	if err != nil {
		return nil, nil, nil, err
	}

	if err := knnCPU.Fit(X, y); err != nil {
		knnCPU.Close()
		return nil, nil, nil, err
	}

	knnONNX, err = knn.New(
		knn.WithK(5),
		knn.WithBackend(knn.BackendGPU),
	)
	if err != nil {
		knnGPU.Close()
		knnCPU.Close()

		return nil, nil, nil, err
	}

	err = knnONNX.LoadONNX(
		"exported_models/knn_100000_n_neighbors-5_repeat-1.onnx",
	)
	if err != nil {
		knnGPU.Close()
		knnCPU.Close()
		knnONNX.Close()

		return nil, nil, nil, err
	}

	return knnGPU, knnCPU, knnONNX, nil
}

func TrainKMeans(
	X [][]float32,
) (
	kmeansGPU *kmeans.KMeans,
	kmeansCPU *kmeans.KMeans,
	kmeansONNX *kmeans.KMeans,
	err error,
) {

	kmeansGPU, err = kmeans.New(
		kmeans.WithBackend(kmeans.BackendGPU),
		kmeans.WithNClusters(8),
	)
	if err != nil {
		return nil, nil, nil, err
	}

	if err := kmeansGPU.Fit(X); err != nil {
		kmeansGPU.Close()
		return nil, nil, nil, err
	}

	kmeansCPU, err = kmeans.New(
		kmeans.WithBackend(kmeans.BackendCPU),
		kmeans.WithNClusters(8),
	)

	if err != nil { 
		kmeansGPU.Close() 
		return nil, nil, nil, err 
	} 
	if err := kmeansCPU.Fit(X); err != nil { 
		kmeansGPU.Close() 
		kmeansCPU.Close() 
		return nil, nil, nil, err 
	}
	kmeansONNX, err = kmeans.New(
		kmeans.WithBackend(kmeans.BackendGPU),
		kmeans.WithNClusters(8),
	)
	if err != nil {
		kmeansGPU.Close()
		kmeansCPU.Close()

		return nil, nil, nil, err
	}

	err = kmeansONNX.LoadONNX(
		"exported_models/kmeans_100000_n_clusters-8_repeat-1.onnx",
	)
	if err != nil {
		kmeansGPU.Close()
		kmeansCPU.Close()
		kmeansONNX.Close()

		return nil, nil, nil, err
	}

	return kmeansGPU, kmeansCPU, kmeansONNX, nil
}

func main() {

	config := Config{
		ModelPath: "go/inference_compare/models/rf_model.tl",

		DatasetPath: "benchmark/data/processed_network_traffic.csv",

		PredictRows: 1000,

		Repeats: 1,

		WarmupRuns: 1,

		CPUCores: 16,

		SampleInterval: 20 * time.Millisecond,
	}

	const trainingRows = 1_000_000


	predictRows := []int{
		1000,
		10000,
		30000,
		50000,
		70000,
	}

	maxRows := trainingRows
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

	fmt.Println("Loading RF model...")

	rf, err := randomforest.Load(config.ModelPath)
	if err != nil {
		log.Fatal(err)
	}
	defer rf.Close()

	rfONNX, err := randomforest.New(
		randomforest.WithEstimators(100),
		randomforest.WithMaxDepth(10),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer rfONNX.Close()

	err = rfONNX.LoadONNX(
		"exported_models/random_forest_100000_n_estimators-100_max_depth-10_repeat-1.onnx",
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("RF model imported")

	if config.CPUCores > 0 {
		randomforest.SetCPUThreads(config.CPUCores)
	}

	var results []BenchmarkResult

	fmt.Println("Starting RF benchmarks...")

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
			rf,
			rfONNX,
			X_,
			y_,
			cfg,
		)
		if err != nil {
			log.Fatal(err)
		}

		results = append(results, benchmarkResults...)
		fmt.Printf("RF benchmark complete: %d rows\n", rows)
	}
	fmt.Println("All RF benchmarks complete")

	XTrain := X[:trainingRows]
	// yTrain := y[:trainingRows]

	// fmt.Println("Starting KNN training...")

	// knnGPU, knnCPU, knnONNX, err := TrainKNN(
	// 	XTrain,
	// 	yTrain,
	// )
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// defer knnGPU.Close()
	// defer knnCPU.Close()
	// defer knnONNX.Close()

	// fmt.Println("KNN training complete")

	// for _, rows := range predictRows {

	// 	fmt.Printf("KNN benchmark: %d rows\n", rows)

	// 	if rows > len(X) {
	// 		log.Fatalf(
	// 			"predict rows (%d) exceeds dataset size (%d)",
	// 			rows,
	// 			len(X),
	// 		)
	// 	}

	// 	start := len(X) - rows

	// 	X_ := X[start:]
	// 	y_ := y[start:]

	// 	cfg := config
	// 	cfg.PredictRows = rows

	// 	benchmarkResults, err := BenchmarkKNNInference(
	// 		knnGPU,
	// 		knnCPU,
	// 		knnONNX,
	// 		X_,
	// 		y_,
	// 		cfg,
	// 	)
	// 	if err != nil {
	// 		log.Fatal(err)
	// 	}

	// 	results = append(results, benchmarkResults...)
	// 	fmt.Printf("KNN benchmark complete: %d rows\n", rows)
	// }
	// fmt.Printf("All KNN benchmark completed\n")

	fmt.Println("Starting KMeans training...")

	kmeansGPU, kmeansCPU, kmeansONNX, err := TrainKMeans(
		XTrain,
	)
	if err != nil {
		log.Fatal(err)
	}

	defer kmeansGPU.Close()
	defer kmeansCPU.Close()
	defer kmeansONNX.Close()

	fmt.Println("KMeans training complete")

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
			kmeansONNX,
			X_,
			cfg,
		)
		if err != nil {
			log.Fatal(err)
		}

		results = append(results, benchmarkResults...)
		fmt.Printf("KMeans benchmark complete: %d rows\n", rows)
	}
	fmt.Println("All KMeans benchmarks complete")

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
			"%25s | %4s | %6d rows | %.2f M samples/s\n",
			r.Model,
			r.Backend,
			r.PredictRows,
			r.Throughput/1e6,
		)
	}
}