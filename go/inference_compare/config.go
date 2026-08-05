package main

import "time"

type Config struct {
	// Saved model to benchmark
	ModelPath string

	DatasetPath string

	// Number of samples to predict
	PredictRows int

	// Number of timed benchmark runs
	Repeats int

	// Number of warmup runs before timing
	WarmupRuns int

	// Number of CPU threads to use (OpenMP)
	// 0 = use OpenMP default
	CPUCores int

	// Sampling interval for CPU/GPU utilization
	SampleInterval time.Duration
}