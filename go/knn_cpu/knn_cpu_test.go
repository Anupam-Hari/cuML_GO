package knn_cpu

import (
	"testing"
	"os"
	"flag"

	"github.com/Anupam-Hari/cuml-go/go/internal/dataset"
	"github.com/Anupam-Hari/cuml-go/go/internal/capi"
)

var maxRows = flag.Int(
	"rows",
	-1,
	"Maximum number of dataset rows",
)

func TestKNNCPUProcessedDataset(t *testing.T) {

	wd, _ := os.Getwd()
	t.Log("Working directory:", wd)

	X, yInt, err := dataset.LoadCSV(
		"../../benchmark/data/processed_network_traffic.csv",
		"is_malicious",
		*maxRows,
	)
	if err != nil {
		t.Fatal(err)
	}

	nSamples := len(X)
	if nSamples == 0 {
		t.Fatal("empty dataset")
	}
	nFeatures := len(X[0])

	// flatten X
	Xflat := make([]float32, 0, nSamples*nFeatures)
	for _, row := range X {
		Xflat = append(Xflat, row...)
	}

	// convert labels to int32
	y := make([]int32, len(yInt))
	maxLabel := int32(0)
	for i, v := range yInt {
		y[i] = int32(v)
		if y[i] > maxLabel {
			maxLabel = y[i]
		}
	}
	nClasses := int(maxLabel) + 1

	model := capi.NewKNNCPU(
		Xflat,
		y,
		nSamples,
		nFeatures,
		nClasses,
	)
	defer model.Free()

	k := 5

	pred := model.Predict(
		Xflat,
		nSamples,
		k,
	)

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