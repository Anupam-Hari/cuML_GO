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
	"github.com/Anupam-Hari/cuml-go/go/kmeans"
	"github.com/Anupam-Hari/cuml-go/go/knn"
	"github.com/Anupam-Hari/cuml-go/go/random_forest"
)

func BenchmarkRandomForest(dataset Dataset) (BenchmarkResult, error) {

	split := SplitDataset(dataset, 0.8)

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

	cpuMonitor, _ := capi.NewCPUMonitor()
	defer cpuMonitor.Close()

	gpumonitor, _ := capi.NewGPUMonitor()
	defer gpumonitor.Close()

	timer := Timer{}

	gpumonitor.Start()
	cpuMonitor.Start()

	timer.Start()

	err = rf.Fit(
		split.XTrain,
		split.YTrain,
	)
	if err != nil {
		return result, err
	}

	result.TrainTimeMS = timer.Stop()

	timer.Start()

	pred, err := rf.Predict(split.XTest, 1)
	if err != nil {
		return result, err
	}

	predictTimeMS := timer.Stop()
	gpumonitor.Stop()
	cpuMonitor.Stop()

	correct := 0
	for i := range pred {
		if pred[i] == split.YTest[i] {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(split.YTest))
	result.Accuracy = accuracy

	result.GPUAvg = gpumonitor.Average()
	result.GPUPeak = gpumonitor.Peak()

	result.CPUAvg = cpuMonitor.Average()
	result.CPUPeak = cpuMonitor.Peak()

	result.PredictionThroughput =
		float64(result.TestRows) /
			(predictTimeMS / 1000.0)

	result.TotalTimeMS =
		result.TrainTimeMS +
			predictTimeMS

	return result, nil
}

func BenchmarkGoLearnRandomForest(dataset Dataset) (BenchmarkResult, error) {

	split := SplitDataset(dataset, 0.8)

	result := BenchmarkResult{
		Model:     "GoLearn Random Forest",
		TrainRows: split.TrainRows,
		TestRows:  split.TestRows,
	}

	trainFile := "golearn_train_tmp.csv"
	testFile := "golearn_test_tmp.csv"

	err := datasetpkg.WriteGoLearnCSV(
		trainFile,
		split.XTrain,
		split.YTrain,
	)
	if err != nil {
		return result, err
	}

	err = datasetpkg.WriteGoLearnCSV(
		testFile,
		split.XTest,
		split.YTest,
	)
	if err != nil {
		return result, err
	}

	defer os.Remove(trainFile)
	defer os.Remove(testFile)

	classAttr := base.NewCategoricalAttribute()
	classAttr.SetName("class")

	overrides := map[int]base.Attribute{
		49: classAttr,
	}

	classGroups := map[string]string{
		"class": "CLASS",
	}

	trainData, err := base.ParseCSVToInstancesWithAttributeGroups(
		trainFile,
		nil,
		classGroups,
		overrides,
		true,
	)
	if err != nil {
		return result, err
	}

	testData, err := base.ParseCSVToInstancesWithAttributeGroups(
		testFile,
		nil,
		classGroups,
		overrides,
		true,
	)
	if err != nil {
		return result, err
	}

	mtry := int(math.Sqrt(float64(len(split.XTrain[0]))))

	rf := ensemble.NewRandomForest(
		100, // trees
		mtry,   // features considered per split
	)
	cpuMonitor, _ := capi.NewCPUMonitor()
	defer cpuMonitor.Close()

	timer := Timer{}
	cpuMonitor.Start()

	timer.Start()

	err = rf.Fit(trainData)
	if err != nil {
		return result, err
	}

	result.TrainTimeMS = timer.Stop()

	timer.Start()

	predictions, err := rf.Predict(testData)
	if err != nil {
		return result, err
	}

	predictTimeMS := timer.Stop()

	cpuMonitor.Stop()

	cm, err := evaluation.GetConfusionMatrix(
		testData,
		predictions,
	)
	if err != nil {
		return result, err
	}

	result.CPUAvg = cpuMonitor.Average()
	result.CPUPeak = cpuMonitor.Peak()

	result.Accuracy = evaluation.GetAccuracy(cm)

	result.PredictionThroughput =
		float64(result.TestRows) /
		(predictTimeMS / 1000.0)

	result.TotalTimeMS =
		result.TrainTimeMS +
		predictTimeMS

	return result, nil
}

func BenchmarkRFInference(dataset Dataset) (BenchmarkResult, error) {

	split := SplitDataset(dataset, 0.8)

	result := BenchmarkResult{
		Model:     "Random Forest Inference",
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

	timer := Timer{}

	//-------------------------------------------------
	// Train once
	//-------------------------------------------------

	timer.Start()

	err = rf.Fit(
		split.XTrain,
		split.YTrain,
	)
	if err != nil {
		return result, err
	}

	result.TrainTimeMS = timer.Stop()

	//-------------------------------------------------
	// GPU warmup
	//-------------------------------------------------

	_, err = rf.Predict(
		split.XTest,
		randomforest.BackendGPU,
	)
	if err != nil {
		return result, err
	}

	//-------------------------------------------------
	// GPU benchmark
	//-------------------------------------------------

	timer.Start()

	gpuPred, err := rf.Predict(
		split.XTest,
		randomforest.BackendGPU,
	)
	if err != nil {
		return result, err
	}

	result.GPUPredictionTimeMS = timer.Stop()

	result.GPUThroughput =
		float64(result.TestRows) /
			(result.GPUPredictionTimeMS / 1000.0)

	//-------------------------------------------------
	// CPU warmup
	//-------------------------------------------------

	_, err = rf.Predict(
		split.XTest,
		randomforest.BackendCPU,
	)
	if err != nil {
		return result, err
	}

	//-------------------------------------------------
	// CPU benchmark
	//-------------------------------------------------

	timer.Start()

	cpuPred, err := rf.Predict(
		split.XTest,
		randomforest.BackendCPU,
	)
	if err != nil {
		return result, err
	}

	result.CPUPredictionTimeMS = timer.Stop()

	result.CPUThroughput =
		float64(result.TestRows) /
			(result.CPUPredictionTimeMS / 1000.0)

	//-------------------------------------------------
	// Accuracy
	//-------------------------------------------------

	correct := 0

	for i := range gpuPred {

		if gpuPred[i] == split.YTest[i] {
			correct++
		}

		if gpuPred[i] != cpuPred[i] {
			return result, fmt.Errorf(
				"CPU and GPU predictions differ at sample %d",
				i,
			)
		}
	}

	result.Accuracy =
		float64(correct) /
			float64(len(split.YTest))

	return result, nil
}

func BenchmarkKNN(dataset Dataset) (BenchmarkResult, error) {

	split := SplitDataset(dataset, 0.8)

	result := BenchmarkResult{
		Model:     "KNN",
		TrainRows: split.TrainRows,
		TestRows:  split.TestRows,
	}

	knn, err := knn.New(
		knn.WithK(5),
	)
	if err != nil {
		return result, err
	}
	defer knn.Close()

	timer := Timer{}
	gpumonitor, _ := capi.NewGPUMonitor()
	defer gpumonitor.Close()

	cpuMonitor, _ := capi.NewCPUMonitor()
    defer cpuMonitor.Close()

	gpumonitor.Start()
	cpuMonitor.Start()
	timer.Start()

	err = knn.Fit(
		split.XTrain,
		split.YTrain,
	)
	if err != nil {
		return result, err
	}

	result.TrainTimeMS = timer.Stop()

	timer.Start()

	pred, err := knn.Predict(split.XTest)
	if err != nil {
		return result, err
	}

	predictTimeMS := timer.Stop()

	gpumonitor.Stop()
	cpuMonitor.Stop()

	correct := 0
	for i := range pred {
		if pred[i] == split.YTest[i] {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(split.YTest))

	result.Accuracy = accuracy

	result.GPUAvg = gpumonitor.Average()
	result.GPUPeak = gpumonitor.Peak()

	result.CPUAvg = cpuMonitor.Average()
	result.CPUPeak = cpuMonitor.Peak()

	result.PredictionThroughput =
		float64(result.TestRows) /
			(predictTimeMS / 1000.0)

	result.TotalTimeMS =
		result.TrainTimeMS +
			predictTimeMS

	return result, nil
}

func BenchmarkKMeans(dataset Dataset) (BenchmarkResult, error) {

	split := SplitDataset(dataset, 0.8)

	result := BenchmarkResult{
		Model:     "KMeans",
		TrainRows: split.TrainRows,
		TestRows:  split.TestRows,
	}

	nClasses, err := NumClasses(dataset)
	if err != nil {
		return result, err
	}

	kmeans, err := kmeans.New(
		kmeans.WithNClusters(nClasses),
		kmeans.WithMaxIters(300),
		kmeans.WithTolerance(1e-4),
	)
	if err != nil {
		return result, err
	}
	defer kmeans.Close()

	timer := Timer{}

	cpuMonitor, _ := capi.NewCPUMonitor()
	defer cpuMonitor.Close()

	gpumonitor, _ := capi.NewGPUMonitor()
	defer gpumonitor.Close()

	cpuMonitor.Start()
	gpumonitor.Start()
	

	timer.Start()

	err = kmeans.Fit(split.XTrain)
	if err != nil {
		return result, err
	}

	result.TrainTimeMS = timer.Stop()

	timer.Start()

	_, err = kmeans.Predict(split.XTest)
	if err != nil {
		return result, err
	}

	predictTimeMS := timer.Stop()

	gpumonitor.Stop()
	cpuMonitor.Stop()

	result.GPUAvg = gpumonitor.Average()
	result.GPUPeak = gpumonitor.Peak()

	result.CPUAvg = cpuMonitor.Average()
	result.CPUPeak = cpuMonitor.Peak()

	result.PredictionThroughput =
		float64(result.TestRows) /
			(predictTimeMS / 1000.0)

	result.TotalTimeMS =
		result.TrainTimeMS +
			predictTimeMS

	return result, nil
}