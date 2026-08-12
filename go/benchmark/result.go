package main

type BenchmarkResult struct {
	Model                 string
	PredictRows           int
	Backend               string
	Runs                  int
	AvgAccuracy           float64
	AvgPredictionTimeMS   float64
	AvgThroughput         float64
	AvgCPU                float64
}