package kmeans

import (
	"math"

	clus "github.com/Giulianos/ml-tp4/cluster"
)

// DistanceFunc is a function that returns the distance
// between two examples.
type DistanceFunc func(e1, e2 clus.Example) float64

// DistanceEuclideanAll computes the euclidean
// distance using all attributes
func DistanceEuclideanAll(e1, e2 clus.Example) float64 {
	var squared float64
	for i := range e1 {
		squared += math.Pow(e1[i]-e2[i], 2)
	}

	return math.Sqrt(squared)
}
