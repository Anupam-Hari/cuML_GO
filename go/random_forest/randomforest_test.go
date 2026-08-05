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

	// Train once.
	if err := rf.Fit(X, y); err != nil {
		t.Fatal(err)
	}

	// Choose inference backend.
	pred, err := rf.Predict(X, backend)
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

	rf, err := New(
		WithEstimators(100),
		WithMaxDepth(16),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer rf.Close()

	// Train exactly once.
	if err := rf.Fit(X, y); err != nil {
		t.Fatal(err)
	}

	t.Run("GPU", func(t *testing.T) {
		pred, err := rf.Predict(X, BackendGPU)
		if err != nil {
			t.Fatal(err)
		}

		correct := 0
		for i := range pred {
			if pred[i] == y[i] {
				correct++
			}
		}

		t.Logf("[GPU] Samples  : %d", len(y))
		t.Logf("[GPU] Accuracy : %.2f%%",
			100*float64(correct)/float64(len(y)))
	})

	t.Run("CPU", func(t *testing.T) {
		pred, err := rf.Predict(X, BackendCPU)
		if err != nil {
			t.Fatal(err)
		}

		correct := 0
		for i := range pred {
			if pred[i] == y[i] {
				correct++
			}
		}

		t.Logf("[CPU] Samples  : %d", len(y))
		t.Logf("[CPU] Accuracy : %.2f%%",
			100*float64(correct)/float64(len(y)))
	})
}