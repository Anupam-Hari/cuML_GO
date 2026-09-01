package main

import (
	"fmt"
	"math"

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

func computeSilhouetteScore(
	features [][]float32,
	labels []int,
	sampleSize int,
) float64 {
	if len(features) == 0 ||
		len(features) != len(labels) {
		return 0
	}

	if sampleSize > len(features) {
		sampleSize = len(features)
	}

	// Use the first sampleSize points.
	// Since the inference dataset is already fixed,
	// this makes the metric deterministic.
	features = features[:sampleSize]
	labels = labels[:sampleSize]

	// Check that at least two clusters exist.
	uniqueClusters := make(map[int]struct{})

	for _, label := range labels {
		uniqueClusters[label] = struct{}{}
	}

	if len(uniqueClusters) < 2 {
		return 0
	}

	totalScore := 0.0

	for i := 0; i < len(features); i++ {
		currentLabel := labels[i]

		intraDistance := 0.0
		intraCount := 0

		clusterDistances := make(map[int]float64)
		clusterCounts := make(map[int]int)

		for j := 0; j < len(features); j++ {
			if i == j {
				continue
			}

			distance := euclideanDistance(
				features[i],
				features[j],
			)

			if labels[j] == currentLabel {
				intraDistance += distance
				intraCount++
			} else {
				clusterDistances[labels[j]] += distance
				clusterCounts[labels[j]]++
			}
		}

		if intraCount == 0 {
			continue
		}

		a := intraDistance / float64(intraCount)

		b := math.MaxFloat64

		for cluster, distanceSum := range clusterDistances {
			count := clusterCounts[cluster]

			if count == 0 {
				continue
			}

			averageDistance := distanceSum / float64(count)

			if averageDistance < b {
				b = averageDistance
			}
		}

		if a == 0 && b == 0 {
			continue
		}

		denominator := math.Max(a, b)

		if denominator > 0 {
			totalScore += (b - a) / denominator
		}
	}

	return totalScore / float64(len(features))
}

func euclideanDistance(a, b []float32) float64 {
	sum := 0.0

	for i := range a {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}

	return math.Sqrt(sum)
}

func maxFloat64(a, b float64) float64 {
	if a > b {
		return a
	}

	return b
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

	// fmt.Printf(
	// 	"Benchmark backend: warmup=%d repeats=%d samples=%d\n",
	// 	warmupRuns,
	// 	repeats,
	// 	samples,
	// )

	//-------------------------------------------------
	// Warmup
	//-------------------------------------------------

	for i := 0; i < warmupRuns; i++ {

		// fmt.Printf("Warmup %d/%d\n", i+1, warmupRuns)

		_, err = predict()
		if err != nil {
			return
		}
	}

	// fmt.Println("Warmup complete")

	//-------------------------------------------------
	// Metrics
	//-------------------------------------------------

	// fmt.Println("Creating metrics")

	metrics, err := benchmark.NewMetrics()
	if err != nil {
		return
	}
	defer metrics.Close()

	// fmt.Println("Metrics created")

	//-------------------------------------------------
	// Benchmark
	//-------------------------------------------------

	timer := benchmark.Timer{}

	// fmt.Println("Starting metrics")

	metrics.Start()

	// fmt.Println("Starting timer")

	timer.Start()

	for i := 0; i < repeats; i++ {

		// fmt.Printf("Prediction %d/%d\n", i+1, repeats)

		pred, err = predict()
		if err != nil {
			metrics.Stop()
			return
		}
	}

	// fmt.Println("All predictions complete")

	avgTimeMS = timer.Stop() / float64(repeats)

	metrics.Stop()

	// fmt.Println("Metrics stopped")

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
	gpuModel *randomforest.RandomForest,
	cpuModel *randomforest.RandomForest,
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

	return []BenchmarkResult{
		gpuResult,
		cpuResult,
	}, nil
}

//-------------------------------------------------
// KNN
//-------------------------------------------------

func BenchmarkKNNInference(
	gpuModel *knn.KNN,
	cpuModel *knn.KNN,
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

	return []BenchmarkResult{
		gpuResult,
		cpuResult,
	}, nil
}

//-------------------------------------------------
// KMeans
//-------------------------------------------------

func BenchmarkKMeansInference(
	gpuModel *kmeans.KMeans,
	cpuModel *kmeans.KMeans,
	X [][]float32,
	config Config,
) ([]BenchmarkResult, error) {

	const modelName = "KMeans Inference"

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
	// Results
	//-------------------------------------------------

	gpuMetric := computeSilhouetteScore(
		X,
		gpuPred,
		5000,
	)

	gpuResult := createBenchmarkResult(
		modelName,
		"gpu",
		config.PredictRows,
		gpuMetric,
		gpuTime,
		gpuThroughput,
		gpuCPUAvg,
	)

	cpuMetric := computeSilhouetteScore(
		X,
		cpuPred,
		5000,
	)

	cpuResult := createBenchmarkResult(
		modelName,
		"cpu",
		config.PredictRows,
		cpuMetric,
		cpuTime,
		cpuThroughput,
		cpuCPUAvg,
	)

	return []BenchmarkResult{
		gpuResult,
		cpuResult,
	}, nil
}