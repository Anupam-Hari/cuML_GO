package randomforest

import "testing"

func TestRandomForestFitPredict(t *testing.T) {
	X := [][]float32{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	y := []int{
		0,
		0,
		1,
		1,
	}

	rf, err := New(
		WithEstimators(10),
		WithMaxDepth(5),
	)

	if err != nil {
		t.Fatal(err)
	}
	defer rf.Close()

	if err := rf.Fit(X, y); err != nil {
		t.Fatal(err)
	}

	pred, err := rf.Predict(X)
	if err != nil {
		t.Fatal(err)
	}

	if len(pred) != len(y) {
		t.Fatalf("expected %d predictions, got %d", len(y), len(pred))
	}

	for i := range pred {
		if pred[i] != y[i] {
			t.Fatalf(
				"prediction mismatch at %d: expected %d got %d",
				i,
				y[i],
				pred[i],
			)
		}
	}
}
