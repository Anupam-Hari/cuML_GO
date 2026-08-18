package main

import (
	"fmt"

	benchmark "github.com/Anupam-Hari/cuml-go/go/internal/benchmark"
	knn "github.com/Anupam-Hari/cuml-go/go/knn"
	kmeans "github.com/Anupam-Hari/cuml-go/go/kmeans"
	randomforest "github.com/Anupam-Hari/cuml-go/go/random_forest"
)

func computeAccuracy(
	predictions []int,
	labels []int,
) float64 {

	if len(predictions) == 0 ||
		len(predictions) != len(labels) {
		return 0
	}

	correct := 0

	for i := range predictions {
		if predictions[i] == labels[i] {
			correct++
		}
	}

	return float64(correct) / float64(len(labels))
}

func calculateThroughput(
	samples int,
	avgPredictionTimeMS float64,
) float64 {

	if avgPredictionTimeMS <= 0 {
		return 0
	}

	return float64(samples) /
		(avgPredictionTimeMS / 1000.0)
}

// benchmarkBackend contains the common benchmarking procedure.
// The model-specific Predict() call is supplied by the caller.
func benchmarkBackend(
	predict func() ([]int, error),
	samples int,
	repeats int,
	warmupRuns int,
) (
	pred []int,
	avgTimeMS float64,
	throughput float64,
	cpuAvg float64,
	err error,
) {

	fmt.Printf(
		"Benchmark backend: warmup=%d repeats=%d samples=%d\n",
		warmupRuns,
		repeats,
		samples,
	)

	//-------------------------------------------------
	// Warmup
	//-------------------------------------------------

	for i := 0; i < warmupRuns; i++ {

		fmt.Printf("Warmup %d/%d\n", i+1, warmupRuns)

		_, err = predict()
		if err != nil {
			return
		}
	}

	fmt.Println("Warmup complete")

	//-------------------------------------------------
	// Metrics
	//-------------------------------------------------

	fmt.Println("Creating metrics")

	metrics, err := benchmark.NewMetrics()
	if err != nil {
		return
	}
	defer metrics.Close()

	fmt.Println("Metrics created")

	//-------------------------------------------------
	// Benchmark
	//-------------------------------------------------

	timer := benchmark.Timer{}

	fmt.Println("Starting metrics")

	metrics.Start()

	fmt.Println("Starting timer")

	timer.Start()

	for i := 0; i < repeats; i++ {

		fmt.Printf("Prediction %d/%d\n", i+1, repeats)

		pred, err = predict()
		if err != nil {
			metrics.Stop()
			return
		}
	}

	fmt.Println("All predictions complete")

	avgTimeMS = timer.Stop() / float64(repeats)

	metrics.Stop()

	fmt.Println("Metrics stopped")

	throughput = calculateThroughput(
		samples,
		avgTimeMS,
	)

	cpuAvg = metrics.CPUAverage()

	return
}

func createBenchmarkResult(
	model string,
	backend string,
	predictRows int,
	accuracy float64,
	predictionTimeMS float64,
	throughput float64,
	cpuAvg float64,
) BenchmarkResult {

	return BenchmarkResult{
		Model:            model,
		Backend:          backend,
		PredictRows:      predictRows,
		Accuracy:         accuracy,
		PredictionTimeMS: predictionTimeMS,
		Throughput:       throughput,
		CPUAvg:           cpuAvg,
	}
}

func comparePredictions(
	cpuPred []int,
	gpuPred []int,
) error {

	if len(cpuPred) != len(gpuPred) {
		return fmt.Errorf(
			"CPU/GPU prediction length mismatch: CPU=%d GPU=%d",
			len(cpuPred),
			len(gpuPred),
		)
	}

	for i := range gpuPred {

		if gpuPred[i] != cpuPred[i] {
			return fmt.Errorf(
				"CPU/GPU prediction mismatch at sample %d: CPU=%d GPU=%d",
				i,
				cpuPred[i],
				gpuPred[i],
			)
		}
	}

	return nil
}

//-------------------------------------------------
// Random Forest
//-------------------------------------------------

func BenchmarkRFInference(
	rf *randomforest.RandomForest,
	rfONNX *randomforest.RandomForest,
	X [][]float32,
	y []int,
	config Config,
) ([]BenchmarkResult, error) {

	const modelName = "Random Forest Inference"

	//-------------------------------------------------
	// GPU
	//-------------------------------------------------

	gpuPred,
		gpuTime,
		gpuThroughput,
		gpuCPUAvg,
		err := benchmarkBackend(
		func() ([]int, error) {
			return rf.Predict(
				X,
				randomforest.BackendGPU,
			)
		},
		len(X),
		config.Repeats,
		config.WarmupRuns,
	)

	if err != nil {
		return nil, err
	}

	//-------------------------------------------------
	// CPU
	//-------------------------------------------------

	cpuPred,
		cpuTime,
		cpuThroughput,
		cpuCPUAvg,
		err := benchmarkBackend(
		func() ([]int, error) {
			return rf.Predict(
				X,
				randomforest.BackendCPU,
			)
		},
		len(X),
		config.Repeats,
		config.WarmupRuns,
	)

	if err != nil {
		return nil, err
	}

	//-------------------------------------------------
	// ONNX
	//-------------------------------------------------

	onnxPred,
		onnxTime,
		onnxThroughput,
		onnxCPUAvg,
		err := benchmarkBackend(
		func() ([]int, error) {
			return rfONNX.PredictONNX(X)
		},
		len(X),
		config.Repeats,
		config.WarmupRuns,
	)

	if err != nil {
		return nil, err
	}

	//-------------------------------------------------
	// Verify prediction lengths
	//-------------------------------------------------

	if len(cpuPred) != len(gpuPred) {
		return nil,
			fmt.Errorf(
				"prediction length mismatch CPU=%d GPU=%d",
				len(cpuPred),
				len(gpuPred),
			)
	}

	if len(cpuPred) != len(onnxPred) {
		return nil,
			fmt.Errorf(
				"prediction length mismatch CPU=%d ONNX=%d",
				len(cpuPred),
				len(onnxPred),
			)
	}

	//-------------------------------------------------
	// Create results
	//-------------------------------------------------

	gpuResult := createBenchmarkResult(
		modelName,
		"gpu",
		config.PredictRows,
		computeAccuracy(gpuPred, y),
		gpuTime,
		gpuThroughput,
		gpuCPUAvg,
	)

	cpuResult := createBenchmarkResult(
		modelName,
		"cpu",
		config.PredictRows,
		computeAccuracy(cpuPred, y),
		cpuTime,
		cpuThroughput,
		cpuCPUAvg,
	)

	onnxResult := createBenchmarkResult(
		modelName,
		"onnx",
		config.PredictRows,
		computeAccuracy(onnxPred, y),
		onnxTime,
		onnxThroughput,
		onnxCPUAvg,
	)

	return []BenchmarkResult{
		gpuResult,
		cpuResult,
		onnxResult,
	}, nil
}

//-------------------------------------------------
// KNN
//-------------------------------------------------

func BenchmarkKNNInference(
	gpuModel *knn.KNN,
	cpuModel *knn.KNN,
	onnxModel *knn.KNN,
	X [][]float32,
	y []int,
	config Config,
) ([]BenchmarkResult, error) {

	const modelName = "KNN Inference"

	//-------------------------------------------------
	// GPU
	//-------------------------------------------------

	gpuPred,
		gpuTime,
		gpuThroughput,
		gpuCPUAvg,
		err := benchmarkBackend(
		func() ([]int, error) {
			return gpuModel.Predict(X)
		},
		len(X),
		config.Repeats,
		config.WarmupRuns,
	)

	if err != nil {
		return nil, err
	}

	//-------------------------------------------------
	// CPU
	//-------------------------------------------------

	cpuPred,
		cpuTime,
		cpuThroughput,
		cpuCPUAvg,
		err := benchmarkBackend(
		func() ([]int, error) {
			return cpuModel.Predict(X)
		},
		len(X),
		config.Repeats,
		config.WarmupRuns,
	)

	if err != nil {
		return nil, err
	}

	//-------------------------------------------------
	// ONNX
	//-------------------------------------------------

	onnxPred,
		onnxTime,
		onnxThroughput,
		onnxCPUAvg,
		err := benchmarkBackend(
		func() ([]int, error) {
			return onnxModel.PredictONNX(X)
		},
		len(X),
		config.Repeats,
		config.WarmupRuns,
	)

	if err != nil {
		return nil, err
	}

	//-------------------------------------------------
	// Results
	//-------------------------------------------------

	gpuResult := createBenchmarkResult(
		modelName,
		"gpu",
		config.PredictRows,
		computeAccuracy(gpuPred, y),
		gpuTime,
		gpuThroughput,
		gpuCPUAvg,
	)

	cpuResult := createBenchmarkResult(
		modelName,
		"cpu",
		config.PredictRows,
		computeAccuracy(cpuPred, y),
		cpuTime,
		cpuThroughput,
		cpuCPUAvg,
	)

	onnxResult := createBenchmarkResult(
		modelName,
		"onnx",
		config.PredictRows,
		computeAccuracy(onnxPred, y),
		onnxTime,
		onnxThroughput,
		onnxCPUAvg,
	)

	return []BenchmarkResult{
		gpuResult,
		cpuResult,
		onnxResult,
	}, nil
}

//-------------------------------------------------
// KMeans
//-------------------------------------------------

func BenchmarkKMeansInference(
	gpuModel *kmeans.KMeans,
	cpuModel *kmeans.KMeans,
	onnxModel *kmeans.KMeans,
	X [][]float32,
	config Config,
) ([]BenchmarkResult, error) {

	const modelName = "KMeans Inference"

	//-------------------------------------------------
	// GPU
	//-------------------------------------------------

	_,
		gpuTime,
		gpuThroughput,
		gpuCPUAvg,
		err := benchmarkBackend(
		func() ([]int, error) {
			return gpuModel.Predict(X)
		},
		len(X),
		config.Repeats,
		config.WarmupRuns,
	)

	if err != nil {
		return nil, err
	}

	//-------------------------------------------------
	// CPU
	//-------------------------------------------------

	_,
		cpuTime,
		cpuThroughput,
		cpuCPUAvg,
		err := benchmarkBackend(
		func() ([]int, error) {
			return cpuModel.Predict(X)
		},
		len(X),
		config.Repeats,
		config.WarmupRuns,
	)

	if err != nil {
		return nil, err
	}

	//-------------------------------------------------
	// ONNX
	//-------------------------------------------------

	_,
		onnxTime,
		onnxThroughput,
		onnxCPUAvg,
		err := benchmarkBackend(
		func() ([]int, error) {
			return onnxModel.PredictONNX(X)
		},
		len(X),
		config.Repeats,
		config.WarmupRuns,
	)

	if err != nil {
		return nil, err
	}

	//-------------------------------------------------
	// Results
	//-------------------------------------------------

	gpuResult := createBenchmarkResult(
		modelName,
		"gpu",
		config.PredictRows,
		0,
		gpuTime,
		gpuThroughput,
		gpuCPUAvg,
	)

	cpuResult := createBenchmarkResult(
		modelName,
		"cpu",
		config.PredictRows,
		0,
		cpuTime,
		cpuThroughput,
		cpuCPUAvg,
	)

	onnxResult := createBenchmarkResult(
		modelName,
		"onnx",
		config.PredictRows,
		0,
		onnxTime,
		onnxThroughput,
		onnxCPUAvg,
	)

	return []BenchmarkResult{
		gpuResult,
		cpuResult,
		onnxResult,
	}, nil
}