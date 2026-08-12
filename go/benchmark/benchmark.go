package main

import (
	"os"
	"math"
	"fmt"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/ensemble"
	"github.com/sjwhitworth/golearn/evaluation"

	"github.com/Anupam-Hari/cuml-go/go/internal/capi"
	datasetpkg "github.com/Anupam-Hari/cuml-go/go/internal/dataset"
	benchmarkutil "github.com/Anupam-Hari/cuml-go/go/internal/benchmark"
	"github.com/Anupam-Hari/cuml-go/go/kmeans"
	"github.com/Anupam-Hari/cuml-go/go/knn"
	"github.com/Anupam-Hari/cuml-go/go/random_forest"
)

func BenchmarkRandomForest(dataset benchmarkutil.Dataset) (BenchmarkResult, error) {

	split := benchmarkutil.SplitDataset(dataset, 0.8)

	result := BenchmarkResult{
		Model:     "Random Forest",
		TrainRows: split.TrainRows,
		TestRows:  split.TestRows,
	}

	rf, err := randomforest.New(
		randomforest.WithEstimators(100),
		randomforest.WithMaxDepth(10),
		randomforest.WithMaxFeatures(1.0),
		randomforest.WithMaxLeaves(-1),
		randomforest.WithMaxSamples(1.0),
	)
	if err != nil {
		return result, err
	}
	defer rf.Close()

	//-------------------------------------------------
	// TRAIN
	//-------------------------------------------------

	trainTime, trainCPUAvg, trainCPUPeak, err :=
		BenchmarkRFTrain(rf, split.XTrain, split.YTrain)
	if err != nil {
		return result, err
	}

	result.TrainTimeMS = trainTime
	result.CPUAvg = trainCPUAvg
	result.CPUPeak = trainCPUPeak

	//-------------------------------------------------
	// GPU warmup
	//-------------------------------------------------

	for i := 0; i < 3; i++ {
		_, err = rf.Predict(split.XTest, randomforest.BackendGPU)
		if err != nil {
			return result, err
		}
	}

	//-------------------------------------------------
	// CPU warmup
	//-------------------------------------------------

	for i := 0; i < 3; i++ {
		_, err = rf.Predict(split.XTest, randomforest.BackendCPU)
		if err != nil {
			return result, err
		}
	}

	//-------------------------------------------------
	// GPU benchmark
	//-------------------------------------------------

	gpuPred,
		gpuTime,
		gpuThroughput,
		gpuCPUAvg,
		gpuCPUPeak,
		err := BenchmarkRFGPU(
		rf,
		split.XTest,
		5,
	)
	if err != nil {
		return result, err
	}

	//-------------------------------------------------
	// CPU benchmark
	//-------------------------------------------------

	cpuPred,
		cpuTime,
		cpuThroughput,
		cpuCPUAvg,
		cpuCPUPeak,
		err := BenchmarkRFCPU(
		rf,
		split.XTest,
		5,
	)
	if err != nil {
		return result, err
	}

	//-------------------------------------------------
	// VERIFY
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
	// ACCURACY
	//-------------------------------------------------

	result.Accuracy = computeAccuracy(
		gpuPred,
		split.YTest,
	)


	result.CPUPredictionTimeMS = cpuTime
	result.CPUThroughput = cpuThroughput
	result.CPURunCPUAvg = cpuCPUAvg

	//-------------------------------------------------
	// TOTAL
	//-------------------------------------------------

	result.TotalTimeMS =
		result.TrainTimeMS +
			result.GPUPredictionTimeMS // or CPU if you prefer baseline

	return result, nil
}

func BenchmarkRFTrain(
	rf *randomforest.RandomForest,
	X [][]float32,
	y []int,
) (
	trainTimeMS float64,
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

	err = rf.Fit(X, y)
	if err != nil {
		metrics.Stop()
		return
	}

	trainTimeMS = timer.Stop()
	metrics.Stop()

	cpuAvg = metrics.CPUAverage()
	cpuPeak = metrics.CPUPeak()

	return
}

func BenchmarkRFGPU(
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

	throughput = calculateThroughput(len(X), avgTimeMS)

	cpuAvg = metrics.CPUAverage()
	cpuPeak = metrics.CPUPeak()

	return
}

func BenchmarkRFCPU(
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

	throughput = calculateThroughput(len(X), avgTimeMS)

	cpuAvg = metrics.CPUAverage()
	cpuPeak = metrics.CPUPeak()

	return
}

func BenchmarkKNN(
    X [][]float32,
    y []int,
    config Config,
) (BenchmarkResult, error) {

    result := BenchmarkResult{
        Model:       "KNN",
        PredictRows: config.PredictRows,
        CPUCores:    config.CPUCores,
    }

    //-------------------------------------------------
    // CREATE MODELS
    //-------------------------------------------------

    knnGPU, err := knn.New(
        knn.WithBackend(knn.BackendGPU),
        knn.WithK(config.K),
    )
    if err != nil {
        return result, err
    }
    defer knnGPU.Close()

    knnCPU, err := knn.New(
        knn.WithBackend(knn.BackendCPU),
        knn.WithK(config.K),
    )
    if err != nil {
        return result, err
    }
    defer knnCPU.Close()

    //-------------------------------------------------
    // TRAIN GPU
    //-------------------------------------------------

    timer := benchmark.Timer{}
    timer.Start()

    if err := knnGPU.Fit(X, y); err != nil {
        return result, err
    }

    result.GPUTrainTimeMS = timer.Stop()

    //-------------------------------------------------
    // TRAIN CPU
    //-------------------------------------------------

    timer.Start()

    if err := knnCPU.Fit(X, y); err != nil {
        return result, err
    }

    result.CPUTrainTimeMS = timer.Stop()

    //-------------------------------------------------
    // GPU warmup
    //-------------------------------------------------

    for i := 0; i < config.WarmupRuns; i++ {
        _, err = knnGPU.Predict(X)
        if err != nil {
            return result, err
        }
    }

    //-------------------------------------------------
    // CPU warmup
    //-------------------------------------------------

    for i := 0; i < config.WarmupRuns; i++ {
        _, err = knnCPU.Predict(X)
        if err != nil {
            return result, err
        }
    }

    //-------------------------------------------------
    // Benchmark GPU
    //-------------------------------------------------

    var gpuPred []int

    gpuPred,
        result.GPUPredictionTimeMS,
        result.GPUThroughput,
        result.GPURunCPUAvg,
        result.GPURunCPUPeak,
        err = benchmarkKNNGPU(
        knnGPU,
        X,
        config.Repeats,
    )
    if err != nil {
        return result, err
    }

    //-------------------------------------------------
    // Benchmark CPU
    //-------------------------------------------------

    var cpuPred []int

    cpuPred,
        result.CPUPredictionTimeMS,
        result.CPUThroughput,
        result.CPURunCPUAvg,
        result.CPURunCPUPeak,
        err = benchmarkKNNCPU(
        knnCPU,
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

func benchmarkKNNCPU(
    knn *knn.KNN,
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
        pred, err = knn.Predict(X)
        if err != nil {
            metrics.Stop()
            return
        }
    }

    avgTimeMS = timer.Stop() / float64(repeats)
    metrics.Stop()

    throughput = calculateThroughput(len(X), avgTimeMS)

    cpuAvg = metrics.CPUAverage()
    cpuPeak = metrics.CPUPeak()

    return
}

func benchmarkKNNGPU(
    knn *knn.KNN,
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
        pred, err = knn.Predict(X)
        if err != nil {
            metrics.Stop()
            return
        }
    }

    avgTimeMS = timer.Stop() / float64(repeats)
    metrics.Stop()

    throughput = calculateThroughput(len(X), avgTimeMS)

    cpuAvg = metrics.CPUAverage()
    cpuPeak = metrics.CPUPeak()

    return
}

func BenchmarkKMeans(
    X [][]float32,
    config Config,
) (BenchmarkResult, error) {

    result := BenchmarkResult{
        Model:       "KMeans",
        PredictRows: config.PredictRows,
        CPUCores:    config.CPUCores,
    }

    //-------------------------------------------------
    // CREATE MODELS
    //-------------------------------------------------

    kmGPU, err := kmeans.New(
        kmeans.WithBackend(kmeans.BackendGPU),
        kmeans.WithNClusters(config.NClusters),
    )
    if err != nil {
        return result, err
    }
    defer kmGPU.Close()

    kmCPU, err := kmeans.New(
        kmeans.WithBackend(kmeans.BackendCPU),
        kmeans.WithNClusters(config.NClusters),
    )
    if err != nil {
        return result, err
    }
    defer kmCPU.Close()

    //-------------------------------------------------
    // TRAIN GPU
    //-------------------------------------------------

    timer := benchmark.Timer{}
    timer.Start()

    if err := kmGPU.Fit(X); err != nil {
        return result, err
    }

    result.GPUTrainTimeMS = timer.Stop()

    //-------------------------------------------------
    // TRAIN CPU
    //-------------------------------------------------

    timer.Start()

    if err := kmCPU.Fit(X); err != nil {
        return result, err
    }

    result.CPUTrainTimeMS = timer.Stop()

    //-------------------------------------------------
    // WARMUP GPU
    //-------------------------------------------------

    for i := 0; i < config.WarmupRuns; i++ {
        _, err = kmGPU.Predict(X)
        if err != nil {
            return result, err
        }
    }

    //-------------------------------------------------
    // WARMUP CPU
    //-------------------------------------------------

    for i := 0; i < config.WarmupRuns; i++ {
        _, err = kmCPU.Predict(X)
        if err != nil {
            return result, err
        }
    }

    //-------------------------------------------------
    // GPU BENCH
    //-------------------------------------------------

    result.GPUPredictionTimeMS,
        result.GPUThroughput,
        result.GPURunCPUAvg,
        result.GPURunCPUPeak,
        err = benchmarkKMeansGPU(
        kmGPU,
        X,
        config.Repeats,
    )
    if err != nil {
        return result, err
    }

    //-------------------------------------------------
    // CPU BENCH
    //-------------------------------------------------

    result.CPUPredictionTimeMS,
        result.CPUThroughput,
        result.CPURunCPUAvg,
        result.CPURunCPUPeak,
        err = benchmarkKMeansCPU(
        kmCPU,
        X,
        config.Repeats,
    )
    if err != nil {
        return result, err
    }

    return result, nil
}

func benchmarkKMeansCPU(
    km *kmeans.KMeans,
    X [][]float32,
    repeats int,
) (
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
        _, err = km.Predict(X)
        if err != nil {
            metrics.Stop()
            return
        }
    }

    avgTimeMS = timer.Stop() / float64(repeats)
    metrics.Stop()

    throughput = calculateThroughput(len(X), avgTimeMS)

    cpuAvg = metrics.CPUAverage()
    cpuPeak = metrics.CPUPeak()

    return
}

func benchmarkKMeansGPU(
    km *kmeans.KMeans,
    X [][]float32,
    repeats int,
) (
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
        _, err = km.Predict(X)
        if err != nil {
            metrics.Stop()
            return
        }
    }

    avgTimeMS = timer.Stop() / float64(repeats)
    metrics.Stop()

    throughput = calculateThroughput(len(X), avgTimeMS)

    cpuAvg = metrics.CPUAverage()
    cpuPeak = metrics.CPUPeak()

    return
}