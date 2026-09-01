package main

import (
	// "fmt"
	// "log"
	// "time"
	// "path/filepath"
	// "strings"
	// "github.com/Anupam-Hari/cuml-go/go/internal/dataset"
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

// func main() {

// 	config := Config{
// 		ModelPath: "go/inference_compare/models/rf_model.tl",

// 		DatasetPath: "benchmark/data/processed_network_traffic.csv",

// 		PredictRows: 1000,

// 		Repeats: 10,

// 		WarmupRuns: 2,

// 		CPUCores: 16,

// 		SampleInterval: 20 * time.Millisecond,
// 	}

// 	const trainingRows = 500_000


// 	predictRows := []int{
// 		1000,
// 		10000,
// 		30000,
// 		50000,
// 		70000,
// 	}

// 	// maxRows := trainingRows
// 	// for _, rows := range predictRows {
// 	// 	if rows > maxRows {
// 	// 		maxRows = rows
// 	// 	}
// 	// }

// 	X, y, err := dataset.LoadCSV(
// 		config.DatasetPath,
// 		0,
// 	)
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	if config.CPUCores > 0 {
// 		randomforest.SetCPUThreads(config.CPUCores)
// 	}

// 	var results []BenchmarkResult

// 	XTrain := X[:trainingRows]
// 	yTrain := y[:trainingRows]

// 	// -------------------------------------------------
// 	// Random Forest training
// 	// -------------------------------------------------

// 	fmt.Println("Starting RF training...")

// 	rfGPU, rfCPU, err := TrainRF(
// 		XTrain,
// 		yTrain,
// 	)
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	defer rfGPU.Close()
// 	defer rfCPU.Close()

// 	// fmt.Println("RF training complete")

// 	// -------------------------------------------------
// 	// RF benchmarks
// 	// -------------------------------------------------

// 	for _, rows := range predictRows {

// 		fmt.Printf("RF benchmark: %d rows\n", rows)

// 		if rows > len(X) {
// 			log.Fatalf(
// 				"predict rows (%d) exceeds dataset size (%d)",
// 				rows,
// 				len(X),
// 			)
// 		}

// 		start := len(X) - rows

// 		X_ := X[start:]
// 		y_ := y[start:]

// 		cfg := config
// 		cfg.PredictRows = rows

// 		benchmarkResults, err := BenchmarkRFInference(
// 			rfGPU,
// 			rfCPU,
// 			X_,
// 			y_,
// 			cfg,
// 		)
// 		if err != nil {
// 			log.Fatal(err)
// 		}

// 		results = append(results, benchmarkResults...)

// 		// fmt.Printf(
// 		// 	"RF benchmark complete: %d rows\n",
// 		// 	rows,
// 		// )
// 	}

// 	fmt.Println("All RF benchmarks complete")

// 	fmt.Println("Starting KNN training...")

// 	knnGPU, knnCPU, err := TrainKNN(
// 		XTrain,
// 		yTrain,
// 	)
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	defer knnGPU.Close()
// 	defer knnCPU.Close()

// 	// fmt.Println("KNN training complete")

// 	for _, rows := range predictRows {

// 		fmt.Printf("KNN benchmark: %d rows\n", rows)

// 		if rows > len(X) {
// 			log.Fatalf(
// 				"predict rows (%d) exceeds dataset size (%d)",
// 				rows,
// 				len(X),
// 			)
// 		}

// 		start := len(X) - rows

// 		X_ := X[start:]
// 		y_ := y[start:]

// 		cfg := config
// 		cfg.PredictRows = rows

// 		benchmarkResults, err := BenchmarkKNNInference(
// 			knnGPU,
// 			knnCPU,
// 			X_,
// 			y_,
// 			cfg,
// 		)
// 		if err != nil {
// 			log.Fatal(err)
// 		}

// 		results = append(results, benchmarkResults...)
// 		// fmt.Printf("KNN benchmark complete: %d rows\n", rows)
// 	}
// 	fmt.Printf("All KNN benchmark completed\n")

// 	fmt.Println("Starting KMeans training...")

// 	kmeansGPU, kmeansCPU, err := TrainKMeans(
// 		XTrain,
// 	)
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	defer kmeansGPU.Close()
// 	defer kmeansCPU.Close()

// 	// fmt.Println("KMeans training complete")

// 	for _, rows := range predictRows {
// 		fmt.Printf("KMeans benchmark: %d rows\n", rows)

// 		if rows > len(X) {
// 			log.Fatalf(
// 				"predict rows (%d) exceeds dataset size (%d)",
// 				rows,
// 				len(X),
// 			)
// 		}

// 		start := len(X) - rows

// 		X_ := X[start:]

// 		cfg := config
// 		cfg.PredictRows = rows

// 		benchmarkResults, err := BenchmarkKMeansInference(
// 			kmeansGPU,
// 			kmeansCPU,
// 			X_,
// 			cfg,
// 		)
// 		if err != nil {
// 			log.Fatal(err)
// 		}

// 		results = append(results, benchmarkResults...)
// 		// fmt.Printf("KMeans benchmark complete: %d rows\n", rows)
// 	}
// 	fmt.Println("All KMeans benchmarks complete")

// 	timestamp := time.Now().Format("020106150405")

// 	resultsDir := "go/inference_compare/results"

// 	resultsFile := filepath.Join(
// 		resultsDir,
// 		fmt.Sprintf("inference_results_%s.csv", timestamp),
// 	)

// 	if err := WriteResultsCSV(resultsFile, results); err != nil {
// 		log.Fatal(err)
// 	}

// 	summaryFile := strings.TrimSuffix(resultsFile, ".csv") + "_summary.csv"

// 	if err := WriteSummaryCSV(resultsFile, summaryFile); err != nil {
// 		log.Fatal(err)
// 	}

// 	fmt.Printf("\nResults written to %s\n", resultsFile)
// 	fmt.Printf("Summary written to %s\n", summaryFile)

// 	comparePythonGoThroughput(summaryFile)
// }

func main() {
    comparePythonGoThroughput(
        "/home/anupam/projects/cuml-go/go/inference_compare/results/inference_results_010926172842_summary.csv",
    )
}