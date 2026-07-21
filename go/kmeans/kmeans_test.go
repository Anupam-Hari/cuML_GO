package kmeans

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

func TestKMeansProcessedDataset(t *testing.T) {

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

	kmeans, err := New(
		WithNClusters(8),
		WithMaxIters(300),
		WithTolerance(1e-4),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer kmeans.Close()

	if err := kmeans.Fit(X); err != nil {
		t.Fatal(err)
	}

	labels, err := kmeans.Predict(X)
	if err != nil {
		t.Fatal(err)
	}

	if len(labels) != len(X) {
		t.Fatalf("expected %d cluster labels, got %d", len(X), len(labels))
	}

	clusterCount := make(map[int]int)
	for _, label := range labels {
		clusterCount[label]++
	}

	t.Logf("Samples          : %d", len(X))
	t.Logf("Clusters found   : %d", len(clusterCount))

	for cluster, count := range clusterCount {
		t.Logf("Cluster %d : %d samples", cluster, count)
	}
}