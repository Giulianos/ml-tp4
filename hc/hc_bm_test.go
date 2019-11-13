package hc

import (
	"testing"

	clus "github.com/Giulianos/ml-tp4/cluster"
)

func BenchmarkFit(b *testing.B) {
	for n := 0; n < b.N; n++ {
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
	}
}
