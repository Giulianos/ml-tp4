package hc

import (
	"math"

	clus "github.com/Giulianos/ml-tp4/cluster"
)

// SimilarityFunc is a function that returns the
// similarity of two groups of examples.
type SimilarityFunc func(g1, g2 []clus.Example) float64

func CentroidSimilarity(g1, g2 []clus.Example) float64 {
	// Compute centroids
	c1 := computeCentroid(g1)
	c2 := computeCentroid(g2)

	// Return euclidean distance between centroids
	var squared float64
	for i := range c1 {
		squared += math.Pow(c1[i]-c2[i], 2)
	}

	return math.Sqrt(squared)
}

func computeCentroid(examples []clus.Example) clus.Example {
	if len(examples) == 0 {
		return nil
	}
	centroid := make(clus.Example, len(examples[0]))

	for _, e := range examples {
		for k, v := range e {
			centroid[k] += v / float64(len(examples))
		}
	}

	return centroid
}
