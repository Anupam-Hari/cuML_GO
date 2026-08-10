package main

import (
	"fmt"

	benchmark "github.com/Anupam-Hari/cuml-go/go/internal/benchmark"
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
	cpuAvg float64,
    cpuPeak float64,
	err error,
) {

	metrics, err := benchmark.NewMetrics()
	if err != nil {
		return
	}
	defer metrics.Close()

	timer := benchmark.Timer{}

	metrics.Start()

	timer.Start()

	for i := 0; i < repeats; i++ {

		pred, err = rf.Predict(
			X,
			randomforest.BackendCPU,
		)
		if err != nil {
			metrics.Stop()
			return
		}
	}

	avgTimeMS = timer.Stop() / float64(repeats)

	metrics.Stop()

	throughput = calculateThroughput(
		len(X),
		avgTimeMS,
	)

	cpuAvg = metrics.CPUAverage()
	cpuPeak = metrics.CPUPeak()
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
	cpuAvg float64,
    cpuPeak float64,
	err error,
) {

	metrics, err := benchmark.NewMetrics()
	if err != nil {
		return
	}
	defer metrics.Close()

	timer := benchmark.Timer{}

	metrics.Start()

	timer.Start()

	for i := 0; i < repeats; i++ {

		pred, err = rf.Predict(
			X,
			randomforest.BackendGPU,
		)
		if err != nil {
			metrics.Stop()
			return
		}
	}

	avgTimeMS = timer.Stop() / float64(repeats)

	metrics.Stop()

	throughput = calculateThroughput(
		len(X),
		avgTimeMS,
	)

	cpuAvg = metrics.CPUAverage()
	cpuPeak = metrics.CPUPeak()

	return
}

func BenchmarkRFInference(
	rf *randomforest.RandomForest,
	X [][]float32,
	y []int,
	config Config,
) (BenchmarkResult, error) {

	result := BenchmarkResult{
		Model:       "Random Forest Inference",
		PredictRows: config.PredictRows,
		CPUCores:    config.CPUCores,
	}

	var err error

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
		result.GPURunCPUAvg,
    	result.GPURunCPUPeak,
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
		result.CPURunCPUAvg,
    	result.CPURunCPUPeak,
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