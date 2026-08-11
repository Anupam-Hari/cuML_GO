package main

type BenchmarkResult struct {
	Model string
	Run int

	Backend string
	PredictRows int

	Accuracy float64

	PredictionTimeMS float64
	Throughput       float64

	CPUAvg  float64
}