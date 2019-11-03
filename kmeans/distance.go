package kmeans

import (
	"math"

	clf "github.com/Giulianos/ml-tp4/classifier"
)

// DistanceFunc is a function that returns the distance
// between two examples.
type DistanceFunc func(e1, e2 clf.Example) float64

// DistanceEuclideanAll computes the euclidean
// distance using all attributes
func DistanceEuclideanAll(e1, e2 clf.Example) float64 {
	var squared float64
	for k := range e1 {
		squared += math.Pow(e1[k]-e2[k], 2)
	}

	return math.Sqrt(squared)
}
