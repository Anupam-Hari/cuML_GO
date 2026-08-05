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

	CPUAvg  float64
	CPUPeak float64

	GPUAvg  float64
	GPUPeak float64

	GPUVRAMAvgMB  float64
	GPUVRAMPeakMB float64

	CPUMemoryAvgMB  float64
	CPUMemoryPeakMB float64

	CPUCores int
}