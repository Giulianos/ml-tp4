package knn

import (
	"container/heap"
	"fmt"

	clf "github.com/Giulianos/ml-tp4/classifier"
)

// KNN is the K-Nearest Neighbour
// Classifier interface implementation
type KNN struct {
	distance DistanceFunc
	k        int
	examples []clf.Example
	targets  []string
}

// New creates a KNN classifier
func New(k int, distance DistanceFunc) KNN {
	return KNN{
		distance: distance,
		k:        k,
	}
}

// Fit trains the classifier
func (knn *KNN) Fit(examples []clf.Example, targets []string) error {
	if len(targets) != len(examples) {
		return fmt.Errorf("fit: examples and targets lengths must match")
	}

	knn.examples = make([]clf.Example, len(examples))
	knn.targets = make([]string, len(targets))
	copy(knn.examples, examples)
	copy(knn.targets, targets)

	return nil
}

func (knn KNN) getKNearest(example clf.Example) NeighborHeap {
	nearest := &NeighborHeap{}
	heap.Init(nearest)

	for i, n := range knn.examples {
		dist := knn.distance(example, n)
		heap.Push(nearest, neighbor{example: &knn.examples[i], distance: dist, class: &knn.targets[i]})

		// Check if we have to delete the excess
		if nearest.Len() > knn.k {
			heap.Pop(nearest)
		}
	}

	return *nearest
}

// Classify classifies an example
func (knn KNN) Classify(e clf.Example) string {
	contrib := map[string]float64{}

	nearest := knn.getKNearest(e)

	for _, n := range nearest {
		if n.distance == 0 {
			return *n.class
		}
		var weight float64 = 1
		contrib[*n.class] += weight
	}

	// Find the mode in the contrib map
	var maxClass string
	var maxContrib float64

	for class, val := range contrib {
		if val > maxContrib {
			maxClass = class
			maxContrib = val
		}
	}

	return maxClass
}
