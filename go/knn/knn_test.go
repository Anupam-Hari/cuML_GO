package knn

import (
	"flag"
	"os"
	"testing"

	"github.com/Anupam-Hari/cuml-go/go/internal/dataset"
)

var maxRows = flag.Int(
	"rows",
	-1,
	"Maximum number of dataset rows",
)

func TestKNN_BackendComparison(t *testing.T) {

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

	// ---------------- GPU ----------------
	knnGPU, err := New(
		WithK(5),
		WithBackend(BackendGPU),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer knnGPU.Close()

	if err := knnGPU.Fit(X, y); err != nil {
		t.Fatal(err)
	}

	predGPU, err := knnGPU.Predict(X)
	if err != nil {
		t.Fatal(err)
	}

	accGPU := computeAccuracy(predGPU, y)

	// ---------------- CPU ----------------
	knnCPU, err := New(
		WithK(5),
		WithBackend(BackendCPU),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer knnCPU.Close()

	if err := knnCPU.Fit(X, y); err != nil {
		t.Fatal(err)
	}

	predCPU, err := knnCPU.Predict(X)
	if err != nil {
		t.Fatal(err)
	}

	accCPU := computeAccuracy(predCPU, y)

	// ---------------- Compare ----------------
	t.Logf("Samples       : %d", len(y))
	t.Logf("GPU Accuracy  : %.2f%%", accGPU*100)
	t.Logf("CPU Accuracy  : %.2f%%", accCPU*100)

	// optional strict check
	if len(predCPU) != len(predGPU) {
		t.Fatalf("prediction length mismatch CPU=%d GPU=%d", len(predCPU), len(predGPU))
	}
}

func computeAccuracy(pred, y []int) float64 {
	correct := 0
	for i := range pred {
		if pred[i] == y[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(y))
}