package kmeans

import (
	"log"
	"math"
	"math/rand"
	"strconv"

	clf "github.com/Giulianos/ml-tp4/classifier"
)

// KMeans is the K-Means
// Classifier interface implementation
type KMeans struct {
	k         int
	centroids map[string]clf.Example
	distance  DistanceFunc
	rng       *rand.Rand
}

// New creates a new KMeans implementation
// of the Classifier interface.
func New(k int, seed int64) KMeans {
	return KMeans{
		k:         k,
		distance:  DistanceEuclideanAll,
		centroids: make(map[string]clf.Example, k),
		rng:       rand.New(rand.NewSource(seed)),
	}
}

// Fit trains the classifier
func (km *KMeans) Fit(examples []clf.Example) {
	currClfs := make([]string, len(examples))
	prevClfs := make([]string, len(examples))

	// Initialize random classes
	for i := range currClfs {
		currClfs[i] = strconv.FormatInt(int64(km.rng.Intn(km.k)), 10)
	}

	for classificationsDiffer(prevClfs, currClfs) {
		log.Println(currClfs)
		// Compute new centroids
		for i := 0; i < km.k; i++ {
			cluster := strconv.FormatInt(int64(i), 10)
			km.centroids[cluster] = computeCentroid(examples, currClfs, cluster)
		}
		// Save current classifications to check if it changed before next iteration
		copy(prevClfs, currClfs)
		// Reclassify
		for i, e := range examples {
			currClfs[i] = km.Classify(e)
		}
	}

}

func classificationsDiffer(prevClfs, curClfs []string) bool {
	for i := range prevClfs {
		if prevClfs[i] != curClfs[i] {
			return true
		}
	}

	return false
}

func computeCentroid(examples []clf.Example, currClfs []string, cluster string) clf.Example {
	if len(examples) == 0 {
		return nil
	}
	centroid := make(clf.Example, len(examples[0]))

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

// Classify classifies an example
func (km KMeans) Classify(e clf.Example) string {
	// find nearest centroid
	var curDist = math.MaxFloat64
	var curCent string

	for i, c := range km.centroids {
		d := km.distance(e, c)
		if d < curDist {
			curDist = d
			curCent = i
		}
	}

	return curCent
}

// GetTargetAttribute returns the target
// attribute of the classifier
func (km KMeans) GetTargetAttribute() string {
	return ""
}
