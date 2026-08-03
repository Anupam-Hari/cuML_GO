package main

import (
	"math/rand"
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

	rng := rand.New(rand.NewSource(42))

	perm := rng.Perm(dataset.Rows)

	xTrain := make([][]float32, trainRows)
	yTrain := make([]int, trainRows)

	xTest := make([][]float32, testRows)
	yTest := make([]int, testRows)

	for i := 0; i < trainRows; i++ {
		idx := perm[i]
		xTrain[i] = dataset.X[idx]
		yTrain[i] = dataset.Y[idx]
	}

	for i := 0; i < testRows; i++ {
		idx := perm[trainRows+i]
		xTest[i] = dataset.X[idx]
		yTest[i] = dataset.Y[idx]
	}

	return TrainTestSplit{
		XTrain: xTrain,
		YTrain: yTrain,

		XTest: xTest,
		YTest: yTest,

		TrainRows: trainRows,
		TestRows:  testRows,
	}
}

func NumClasses(dataset Dataset) (int, error) {
	return matrix.NumClasses(dataset.Y)
}