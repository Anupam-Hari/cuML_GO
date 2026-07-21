package main

import (
	"github.com/Anupam-Hari/cuml-go/go/internal/matrix"
)

type Dataset struct {
	X    [][]float32
	Y    []int
	Rows int
	Cols int
}

type TrainTestSplit struct {
	XTrain [][]float32
	YTrain []int

	XTest [][]float32
	YTest []int

	TrainRows int
	TestRows  int
}

func SplitDataset(
	dataset Dataset,
	trainRatio float32,
) TrainTestSplit {

	trainRows := int(float32(dataset.Rows) * trainRatio)
	testRows := dataset.Rows - trainRows

	return TrainTestSplit{
		XTrain: dataset.X[:trainRows],
		YTrain: dataset.Y[:trainRows],

		XTest: dataset.X[trainRows:],
		YTest: dataset.Y[trainRows:],

		TrainRows: trainRows,
		TestRows:  testRows,
	}
}

func NumClasses(dataset Dataset) (int, error) {
	return matrix.NumClasses(dataset.Y)
}