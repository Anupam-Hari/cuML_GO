package knn

import (
	"testing"
	"os"
	"flag"

	"github.com/Anupam-Hari/cuml-go/go/internal/dataset"
)

var maxRows = flag.Int(
	"rows",
	-1,
	"Maximum number of dataset rows",
)

func TestKNNProcessedDataset(t *testing.T) {

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

	knn, err := New(
		WithK(5),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer knn.Close()

	if err := knn.Fit(X, y); err != nil {
		t.Fatal(err)
	}

	pred, err := knn.Predict(X)
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

	t.Logf("Samples  : %d", len(y))
	t.Logf("Accuracy : %.2f%%", accuracy*100)
}