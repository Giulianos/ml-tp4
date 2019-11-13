package kmeans

import (
	"testing"

	clus "github.com/Giulianos/ml-tp4/cluster"
)

func TestPredict(t *testing.T) {
	kmeans := New(3, 1983)

	X := []clus.Example{
		{0, 0},
		{1, 0},
		{0, 1},
		{1, 1},
		{4, 4},
		{5, 4},
		{4, 5},
		{5, 5},
		{100, 0},
		{110, 0},
		{100, 1},
		{110, 1},
	}

	kmeans.Fit(X)

	label1 := kmeans.Predict(clus.Example{0.5, 0.5})
	label2 := kmeans.Predict(clus.Example{4.5, 4.5})
	label3 := kmeans.Predict(clus.Example{101, 1.5})

	if label1 == label2 {
		t.Error("label1 should be different than label2")
	}

	if label1 == label3 {
		t.Error("label1 should be different than label3")
	}

	if label2 == label3 {
		t.Error("label2 should be different than label3", label1, label2, label3)
	}

}
