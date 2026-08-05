package randomforest

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

func runBackendTest(
	t *testing.T,
	backend int,
	name string,
	X [][]float32,
	y []int,
) {
	rf, err := New(
		WithEstimators(100),
		WithMaxDepth(16),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer rf.Close()

	if err := rf.Fit(X, y, backend); err != nil {
		t.Fatal(err)
	}

	pred, err := rf.Predict(X)
	if err != nil {
		t.Fatal(err)
	}

	if len(pred) != len(y) {
		t.Fatalf("expected %d predictions, got %d", len(y), len(pred))
	}

	correct := 0
	for i := range pred {
		if pred[i] == y[i] {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(y))

	t.Logf("[%s] Samples  : %d", name, len(y))
	t.Logf("[%s] Accuracy : %.2f%%", name, accuracy*100)
}

func TestRandomForestProcessedDataset(t *testing.T) {
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

	t.Run("GPU", func(t *testing.T) {
		runBackendTest(
			t,
			BackendGPU,
			"GPU",
			X,
			y,
		)
	})

	t.Run("CPU", func(t *testing.T) {
		runBackendTest(
			t,
			BackendCPU,
			"CPU",
			X,
			y,
		)
	})
}