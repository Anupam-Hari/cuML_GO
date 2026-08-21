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

const onnxModel = "../../exported_models/random_forest_100000_n_estimators-100_max_depth-10_repeat-1.onnx"

func TestRandomForest_BackendComparison(t *testing.T) {

	wd, _ := os.Getwd()
	t.Log("Working directory:", wd)

	X, y, err := dataset.LoadCSV(
		"../../benchmark/data/processed_network_traffic.csv",
		"is_malicious",
		*maxRows,
	)
	if err != nil {
		t.Fatal(err)
	}

	rf, err := New(
		WithEstimators(100),
		WithMaxDepth(16),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer rf.Close()

	if err := rf.Fit(X, y); err != nil {
		t.Fatal(err)
	}

	// ---------------- GPU ----------------

	start := time.Now()

	predGPU, err := rf.Predict(
		X,
		BackendGPU,
	)
	if err != nil {
		t.Fatal(err)
	}

	gpuDuration := time.Since(start)

	accGPU := computeAccuracy(
		predGPU,
		y,
	)

	// ---------------- CPU ----------------

	start = time.Now()

	predCPU, err := rf.Predict(
		X,
		BackendCPU,
	)
	if err != nil {
		t.Fatal(err)
	}

	cpuDuration := time.Since(start)

	accCPU := computeAccuracy(
		predCPU,
		y,
	)

	// ---------------- ONNX ----------------

	if err := rf.LoadONNX(onnxModel); err != nil {
		t.Fatal(err)
	}

	start = time.Now()

	predONNX, err := rf.PredictONNX(X)
	if err != nil {
		t.Fatal(err)
	}

	onnxDuration := time.Since(start)

	accONNX := computeAccuracy(
		predONNX,
		y,
	)

	// ---------------- Results ----------------

	t.Logf("Samples        : %d", len(y))


	t.Logf("GPU Time       : %v", gpuDuration)
	t.Logf("CPU Time       : %v", cpuDuration)
	t.Logf("ONNX Time      : %v", onnxDuration)

	t.Logf("GPU Accuracy   : %.2f%%", accGPU*100)
	t.Logf("CPU Accuracy   : %.2f%%", accCPU*100)
	t.Logf("ONNX Accuracy  : %.2f%%", accONNX*100)
	if len(predGPU) != len(predCPU) {
		t.Fatalf(
			"CPU/GPU prediction length mismatch: %d vs %d",
			len(predCPU),
			len(predGPU),
		)
	}

	if len(predGPU) != len(predONNX) {
		t.Fatalf(
			"GPU/ONNX prediction length mismatch: %d vs %d",
			len(predGPU),
			len(predONNX),
		)
	}
}

func computeAccuracy(
	pred []int,
	y []int,
) float64 {

	correct := 0

	for i := range pred {
		if pred[i] == y[i] {
			correct++
		}
	}

	return float64(correct) / float64(len(y))
}