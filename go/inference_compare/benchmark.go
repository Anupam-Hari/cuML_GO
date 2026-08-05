package main

import (
	"fmt"

	benchmark "github.com/Anupam-Hari/cuml-go/go/internal/benchmark"
	"github.com/Anupam-Hari/cuml-go/go/internal/dataset"
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

	return float64(correct) /
		float64(len(labels))
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

func benchmarkCPU(
	rf *randomforest.RandomForest,
	X [][]float32,
	repeats int,
) (
	pred []int,
	avgTimeMS float64,
	throughput float64,
	avgUtil float64,
	peakUtil float64,
	avgRAM float64,
	peakRAM float64,
	err error,
) {

	metrics, err := benchmark.NewMetrics()
	if err != nil {
		return
	}
	defer metrics.Close()

	timer := benchmark.Timer{}

	metrics.Start()

	var totalTime float64

	for i := 0; i < repeats; i++ {

		timer.Start()

		pred, err = rf.Predict(
			X,
			randomforest.BackendCPU,
		)
		if err != nil {
			metrics.Stop()
			return
		}

		totalTime += timer.Stop()
	}

	metrics.Stop()

	avgTimeMS = totalTime / float64(repeats)

	throughput = calculateThroughput(
		len(X),
		avgTimeMS,
	)

	avgUtil = metrics.CPUAverage()
	peakUtil = metrics.CPUPeak()

	avgRAM = metrics.CPUMemoryAverage()
	peakRAM = metrics.CPUMemoryPeak()

	return
}

func benchmarkGPU(
	rf *randomforest.RandomForest,
	X [][]float32,
	repeats int,
) (
	pred []int,
	avgTimeMS float64,
	throughput float64,
	avgUtil float64,
	peakUtil float64,
	avgVRAM float64,
	peakVRAM float64,
	err error,
) {

	metrics, err := benchmark.NewMetrics()
	if err != nil {
		return
	}
	defer metrics.Close()

	timer := benchmark.Timer{}

	metrics.Start()

	var totalTime float64

	for i := 0; i < repeats; i++ {

		timer.Start()

		pred, err = rf.Predict(
			X,
			randomforest.BackendGPU,
		)
		if err != nil {
			metrics.Stop()
			return
		}

		totalTime += timer.Stop()
	}

	metrics.Stop()

	avgTimeMS = totalTime / float64(repeats)

	throughput = calculateThroughput(
		len(X),
		avgTimeMS,
	)

	avgUtil = metrics.GPUAverage()
	peakUtil = metrics.GPUPeak()

	avgVRAM = metrics.GPUMemoryAverage()
	peakVRAM = metrics.GPUMemoryPeak()

	return
}

func BenchmarkRFInference(
	config Config,
) (BenchmarkResult, error) {

	result := BenchmarkResult{
		Model:       "Random Forest Inference",
		PredictRows: config.PredictRows,
		CPUCores:    config.CPUCores,
	}

	//-------------------------------------------------
	// Load dataset
	//-------------------------------------------------

	X, y, err := dataset.LoadCSV(
		config.DatasetPath,
		"is_malicious",
		0, // load entire dataset
	)
	if err != nil {
		return result, err
	}

	if config.PredictRows > len(X) {
		return result, fmt.Errorf(
			"predict rows (%d) exceeds dataset size (%d)",
			config.PredictRows,
			len(X),
		)
	}

	start := len(X) - config.PredictRows

	X = X[start:]
	y = y[start:]

	//-------------------------------------------------
	// Load trained model
	//-------------------------------------------------

	rf, err := randomforest.Load(config.ModelPath)
	if err != nil {
		return result, err
	}
	defer rf.Close()

	//-------------------------------------------------
	// Configure CPU inference threads
	//-------------------------------------------------

	if config.CPUCores > 0 {
		randomforest.SetCPUThreads(config.CPUCores)
	}

	var gpuPred []int
	var cpuPred []int

	//-------------------------------------------------
	// GPU warmup
	//-------------------------------------------------

	for i := 0; i < config.WarmupRuns; i++ {

		_, err = rf.Predict(
			X,
			randomforest.BackendGPU,
		)
		if err != nil {
			return result, err
		}
	}

	//-------------------------------------------------
	// CPU warmup
	//-------------------------------------------------

	for i := 0; i < config.WarmupRuns; i++ {

		_, err = rf.Predict(
			X,
			randomforest.BackendCPU,
		)
		if err != nil {
			return result, err
		}
	}

	//-------------------------------------------------
	// Benchmark GPU
	//-------------------------------------------------

	gpuPred,
		result.GPUPredictionTimeMS,
		result.GPUThroughput,
		result.GPUAvg,
		result.GPUPeak,
		result.GPUVRAMAvgMB,
		result.GPUVRAMPeakMB,
		err = benchmarkGPU(
		rf,
		X,
		config.Repeats,
	)

	if err != nil {
		return result, err
	}

	//-------------------------------------------------
	// Benchmark CPU
	//-------------------------------------------------

	cpuPred,
		result.CPUPredictionTimeMS,
		result.CPUThroughput,
		result.CPUAvg,
		result.CPUPeak,
		result.CPUMemoryAvgMB,
		result.CPUMemoryPeakMB,
		err = benchmarkCPU(
		rf,
		X,
		config.Repeats,
	)

	if err != nil {
		return result, err
	}

	//-------------------------------------------------
	// Verify predictions
	//-------------------------------------------------

	for i := range gpuPred {

		if gpuPred[i] != cpuPred[i] {

			return result, fmt.Errorf(
				"CPU/GPU prediction mismatch at sample %d",
				i,
			)
		}
	}

	//-------------------------------------------------
	// Accuracy
	//-------------------------------------------------

	result.Accuracy = computeAccuracy(
		gpuPred,
		y,
	)

	return result, nil
}