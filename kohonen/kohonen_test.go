package kohonen

import (
	"testing"

	clus "github.com/Giulianos/ml-tp4/cluster"
)

func TestPredict(t *testing.T) {
	kohonen := New()

	X := []clus.Example{
		{0, 0},
		{1, 0},
		{0, 1},
		{1, 1},
		{4, 4},
		{5, 4},
		{4, 5},
		{5, 5},
	}

	kohonen.Fit(X)

	label1 := kohonen.Predict(clus.Example{0.5, 0.5})
	label2 := kohonen.Predict(clus.Example{4.5, 4.5})

	if label1 == label2 {
		t.Error("label1 should be different than label2")
	}
}
