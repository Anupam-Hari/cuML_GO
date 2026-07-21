package main

type BenchmarkResult struct {
	Model                  string
	TrainRows              int
	TestRows               int
	TrainTimeMS            float64
	TotalTimeMS            float64
	PredictionThroughput   float64
	GPUAvg  float64
	GPUPeak float64
}