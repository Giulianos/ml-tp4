package hc

import (
	"testing"

	clus "github.com/Giulianos/ml-tp4/cluster"
)

func TestPredictDifferentClusters(t *testing.T) {
	hc := New(CentroidSimilarity, 2)

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

	hc.Fit(X)

	label1 := hc.Predict(clus.Example{0.5, 0.5})
	label2 := hc.Predict(clus.Example{4.5, 4.5})

	if label1 == label2 {
		t.Error("label1 should be different than label")
	}
}

func TestPredictSameClusters(t *testing.T) {
	hc := New(CentroidSimilarity, 2)

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

	hc.Fit(X)

	label1 := hc.Predict(clus.Example{0.5, 0.5})
	label2 := hc.Predict(clus.Example{0.3, 0.4})

	if label1 != label2 {
		t.Error("label1 should be equal to label2")
	}
}
