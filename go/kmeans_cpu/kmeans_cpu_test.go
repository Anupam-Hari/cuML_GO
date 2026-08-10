package kmeans_cpu

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

func TestKMeansCPUProcessedDataset(t *testing.T) {

	wd, _ := os.Getwd()
	t.Log("Working directory:", wd)

	X, _, err := dataset.LoadCSV(
		"../../benchmark/data/processed_network_traffic.csv",
		"is_malicious",
		*maxRows,
	)
	if err != nil {
		t.Fatal(err)
	}

	// Flatten 2D → 1D (same as internal matrix does)
	nSamples := len(X)
	if nSamples == 0 {
		t.Fatal("empty dataset")
	}
	nFeatures := len(X[0])

	data := make([]float32, 0, nSamples*nFeatures)
	for _, row := range X {
		data = append(data, row...)
	}

	kmeans := New(
		8,     // nClusters
		300,   // maxIters
		1e-4,  // tolerance
	)
	defer kmeans.Free()

	kmeans.Fit(
		data,
		nSamples,
		nFeatures,
	)

	labels := kmeans.Predict(
		data,
		nSamples,
	)

	if len(labels) != nSamples {
		t.Fatalf("expected %d labels, got %d", nSamples, len(labels))
	}

	clusterCount := make(map[int]int)
	for _, label := range labels {
		clusterCount[int(label)]++
	}

	t.Logf("Samples        : %d", nSamples)
	t.Logf("Clusters found : %d", len(clusterCount))

	for c, count := range clusterCount {
		t.Logf("Cluster %d : %d samples", c, count)
	}
}