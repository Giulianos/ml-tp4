package kmeans

import (
	"math"
	"math/rand"

	clus "github.com/Giulianos/ml-tp4/cluster"
)

// KMeans is the K-Means
// Cluster interface implementation
type KMeans struct {
	k         int
	centroids []clus.Example
	distance  DistanceFunc
	rng       *rand.Rand
}

// New creates a new KMeans clusterer
func New(k int, seed int64) KMeans {
	return KMeans{
		k:         k,
		distance:  DistanceEuclideanAll,
		centroids: make([]clus.Example, k),
		rng:       rand.New(rand.NewSource(seed)),
	}
}

// Fit trains the clusterer
func (km *KMeans) Fit(examples []clus.Example) {
	currClfs := make([]int, len(examples))
	prevClfs := make([]int, len(examples))

	// Initialize random classes
	for i := range currClfs {
		currClfs[i] = km.rng.Intn(km.k)
	}

	for classificationsDiffer(prevClfs, currClfs) {
		// Compute new centroids
		for i := 0; i < km.k; i++ {
			km.centroids[i] = computeCentroid(examples, currClfs, i)
		}
		// Save current classifications to check if it changed before next iteration
		copy(prevClfs, currClfs)
		// Reclassify
		for i, e := range examples {
			currClfs[i] = km.Predict(e)
		}
	}

}

func classificationsDiffer(prevClfs, curClfs []int) bool {
	for i := range prevClfs {
		if prevClfs[i] != curClfs[i] {
			return true
		}
	}

	return false
}

func computeCentroid(examples []clus.Example, currClfs []int, cluster int) clus.Example {
	if len(examples) == 0 {
		return nil
	}
	centroid := make(clus.Example, len(examples[0]))

	for i, e := range examples {
		if currClfs[i] != cluster {
			continue
		}
		for k, v := range e {
			centroid[k] += v / float64(len(examples))
		}
	}

	return centroid
}

// Predict predicts the label of an example
func (km KMeans) Predict(e clus.Example) int {
	// find nearest centroid
	var curDist = math.MaxFloat64
	var curCent int

	for i, c := range km.centroids {
		d := km.distance(e, c)
		if d < curDist {
			curDist = d
			curCent = i
		}
	}

	return curCent
}
