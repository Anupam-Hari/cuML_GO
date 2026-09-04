package randomforest

import (
	"flag"
	"os"
	"testing"
	"time"

	"github.com/Anupam-Hari/cuml-go/go/internal/dataset"
)

var maxRows = flag.Int(
	"rows",
	-1,
	"Maximum number of dataset rows",
)

func TestRandomForest_BackendComparison(t *testing.T) {

	wd, _ := os.Getwd()
	t.Log("Working directory:", wd)

	X, y, err := dataset.LoadCSV(
		"../../benchmark/data/processed_network_traffic_real.csv",
		*maxRows,
	)
	if err != nil {
		t.Fatal(err)
	}

	// ============================================================
	// GPU
	// ============================================================

	rfGPU, err := New(
		WithEstimators(50),
		WithMaxDepth(20),
		WithMaxFeatures(1.0),
		WithMaxLeaves(-1),
		WithMaxSamples(1.0),
		WithBackend(BackendGPU),
	)
	if err != nil {
		t.Fatal(err)
	}

	gpuTrainStart := time.Now()

	if err := rfGPU.Fit(X, y); err != nil {
		rfGPU.Close()
		t.Fatal(err)
	}

	gpuTrainTime := time.Since(gpuTrainStart)

	// Save GPU model.
	gpuModelFile := "../models/random_forest_gpu.model"

	if err := rfGPU.Save(gpuModelFile); err != nil {
		rfGPU.Close()
		t.Fatal(err)
	}

	// Destroy the trained model to make sure Load() actually works.
	rfGPU.Close()

	// Load GPU model.
	rfGPU, err = Load(
		gpuModelFile,
		WithBackend(BackendGPU),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer rfGPU.Close()

	// Warmup.
	for i := 0; i < 2; i++ {
		_, err := rfGPU.Predict(X)
		if err != nil {
			t.Fatal(err)
		}
	}

	gpuPredictStart := time.Now()

	predGPU, err := rfGPU.Predict(X)
	if err != nil {
		t.Fatal(err)
	}

	gpuPredictTime := time.Since(gpuPredictStart)

	accGPU := computeAccuracy(predGPU, y)

	// ============================================================
	// CPU
	// ============================================================

	rfCPU, err := New(
		WithEstimators(50),
		WithMaxDepth(20),
		WithMaxFeatures(1.0),
		WithMaxLeaves(-1),
		WithMaxSamples(1.0),
		WithBackend(BackendCPU),
	)
	if err != nil {
		t.Fatal(err)
	}

	cpuTrainStart := time.Now()

	if err := rfCPU.Fit(X, y); err != nil {
		rfCPU.Close()
		t.Fatal(err)
	}

	cpuTrainTime := time.Since(cpuTrainStart)

	// Save CPU model.
	cpuModelFile := "../models/random_forest_cpu.bin"

	if err := rfCPU.Save(cpuModelFile); err != nil {
		rfCPU.Close()
		t.Fatal(err)
	}

	// Destroy the trained model to make sure Load() actually works.
	rfCPU.Close()

	// Load CPU model.
	rfCPU, err = Load(
		cpuModelFile,
		WithBackend(BackendCPU),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer rfCPU.Close()

	cpuPredictStart := time.Now()

	predCPU, err := rfCPU.Predict(X)
	if err != nil {
		t.Fatal(err)
	}

	cpuPredictTime := time.Since(cpuPredictStart)

	accCPU := computeAccuracy(predCPU, y)

	// ============================================================
	// Logs
	// ============================================================

	t.Logf("Samples         : %d", len(y))

	t.Logf(
		"GPU Train Time  : %v",
		gpuTrainTime,
	)

	t.Logf(
		"GPU Predict Time: %v",
		gpuPredictTime,
	)

	t.Logf(
		"GPU Accuracy    : %.2f%%",
		accGPU*100,
	)

	t.Logf(
		"CPU Train Time  : %v",
		cpuTrainTime,
	)

	t.Logf(
		"CPU Predict Time: %v",
		cpuPredictTime,
	)

	t.Logf(
		"CPU Accuracy    : %.2f%%",
		accCPU*100,
	)

	// ============================================================
	// Sanity
	// ============================================================

	if len(predGPU) != len(predCPU) {
		t.Fatalf(
			"prediction length mismatch GPU=%d CPU=%d",
			len(predGPU),
			len(predCPU),
		)
	}

	if len(predGPU) != len(y) {
		t.Fatalf(
			"GPU prediction length mismatch predictions=%d labels=%d",
			len(predGPU),
			len(y),
		)
	}

	if len(predCPU) != len(y) {
		t.Fatalf(
			"CPU prediction length mismatch predictions=%d labels=%d",
			len(predCPU),
			len(y),
		)
	}

	// Clean up serialized models.
	// os.Remove(gpuModelFile)
	// os.Remove(gpuModelFile + ".meta")

	// os.Remove(cpuModelFile)
	// os.Remove(cpuModelFile + ".meta")
}

func computeAccuracy(pred, y []int) float64 {

	if len(y) == 0 {
		return 0
	}

	correct := 0

	for i := range pred {
		if pred[i] == y[i] {
			correct++
		}
	}

	return float64(correct) / float64(len(y))
}