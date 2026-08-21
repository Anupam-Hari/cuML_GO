package randomforest_cpu

import (
	"flag"
	"os"
	"testing"
	"time"

	"github.com/Anupam-Hari/cuml-go/go/internal/capi"
	"github.com/Anupam-Hari/cuml-go/go/internal/dataset"
)

var maxRowsCPU = flag.Int(
	"rows",
	-1,
	"Maximum number of dataset rows",
)

func TestRandomForestCPU(t *testing.T) {

	wd, _ := os.Getwd()
	t.Log("Working directory:", wd)

	X, y, err := dataset.LoadCSV(
		"../../benchmark/data/processed_network_traffic.csv",
		"is_malicious",
		*maxRowsCPU,
	)
	if err != nil {
		t.Fatal(err)
	}

	if len(X) == 0 {
		t.Fatal("empty dataset")
	}

	// Flatten X.
	rows := len(X)
	cols := len(X[0])

	Xflat := make([]float32, 0, rows*cols)

	for _, row := range X {
		if len(row) != cols {
			t.Fatal("inconsistent number of features")
		}

		Xflat = append(Xflat, row...)
	}

	// Convert labels.
	labels := make([]int32, len(y))

	for i, v := range y {
		labels[i] = int32(v)
	}

	nClasses := 0
	for _, v := range labels {
		if int(v)+1 > nClasses {
			nClasses = int(v) + 1
		}
	}

	// ------------------------------------------------
	// Create mlpack CPU Random Forest
	// ------------------------------------------------

	rf, err := capi.RFCPUCreate(
		100,  // n_estimators
		16,   // max_depth
		1.0,  // max_features
		-1,   // max_leaves
		1.0,  // max_samples
	)
	if err != nil {
		t.Fatal(err)
	}
	defer capi.RFCPUFree(rf)

	// ------------------------------------------------
	// Fit
	// ------------------------------------------------

	fitStart := time.Now()

	if err := capi.RFCPUFit(
		rf,
		Xflat,
		rows,
		cols,
		labels,
		nClasses,
	); err != nil {
		t.Fatal(err)
	}

	fitTime := time.Since(fitStart)

	// ------------------------------------------------
	// Predict
	// ------------------------------------------------

	predictStart := time.Now()

	predictions, err := capi.RFCPUPredict(
		rf,
		Xflat,
		rows,
		cols,
	)
	if err != nil {
		t.Fatal(err)
	}

	predictTime := time.Since(predictStart)

	// ------------------------------------------------
	// Validate
	// ------------------------------------------------

	if len(predictions) != len(y) {
		t.Fatalf(
			"prediction length mismatch: expected=%d got=%d",
			len(y),
			len(predictions),
		)
	}

	correct := 0

	for i := range predictions {
		if int(predictions[i]) == y[i] {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(y))

	// ------------------------------------------------
	// Results
	// ------------------------------------------------

	t.Logf("Samples       : %d", rows)
	t.Logf("Features      : %d", cols)
	t.Logf("Classes       : %d", nClasses)
	t.Logf("Fit Time      : %s", fitTime)
	t.Logf("Predict Time  : %s", predictTime)
	t.Logf("Accuracy      : %.2f%%", accuracy*100)
}