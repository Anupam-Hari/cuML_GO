package main

type BenchmarkResult struct {
	Model string

	Run int

	PredictRows int

	Accuracy float64

	CPUPredictionTimeMS float64
	GPUPredictionTimeMS float64

	CPUThroughput float64
	GPUThroughput float64

	// CPU utilization during CPU inference
	CPURunCPUAvg  float64
	CPURunCPUPeak float64

	// CPU utilization during GPU inference
	GPURunCPUAvg  float64
	GPURunCPUPeak float64

	CPUCores int
}