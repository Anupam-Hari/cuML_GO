package main

type BenchmarkResult struct {
	Model string
	Run   int

	TrainRows int
	TestRows  int

	TrainTimeMS float64
	TotalTimeMS float64

	PredictionThroughput float64

	Accuracy float64

	// New inference benchmark fields
	CPUPredictionTimeMS float64
	GPUPredictionTimeMS float64

	CPUThroughput float64
	GPUThroughput float64

	CPUAvg  float64
	CPUPeak float64

	GPUAvg  float64
	GPUPeak float64
}