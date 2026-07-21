package main

import (
	"github.com/Anupam-Hari/cuml-go/go/random_forest"
	"github.com/Anupam-Hari/cuml-go/go/knn"
	"github.com/Anupam-Hari/cuml-go/go/kmeans"
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

	timer := Timer{}

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

	_, err = rf.Predict(split.XTest)
	if err != nil {
		return result, err
	}

	predictTimeMS := timer.Stop()

	result.PredictionThroughput =
		float64(result.TestRows) /
			(predictTimeMS / 1000.0)

	result.TotalTimeMS =
		result.TrainTimeMS +
			predictTimeMS

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

	_, err = knn.Predict(split.XTest)
	if err != nil {
		return result, err
	}

	predictTimeMS := timer.Stop()

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
		kmeans.WithMaxIters(8),
		kmeans.WithTolerance(1e-4),
	)
	if err != nil {
		return result, err
	}
	defer kmeans.Close()

	timer := Timer{}

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

	result.PredictionThroughput =
		float64(result.TestRows) /
			(predictTimeMS / 1000.0)

	result.TotalTimeMS =
		result.TrainTimeMS +
			predictTimeMS

	return result, nil
}