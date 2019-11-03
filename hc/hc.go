package hc

import (
	"math"

	clus "github.com/Giulianos/ml-tp4/cluster"
)

// HC is the Hierarchical Clustering
// Classifier interface implementation
type HC struct {
	groups     [][]clus.Example
	similarity SimilarityFunc
}

// New creates a new Hierarchical Clustering clusterer
func New(similarity SimilarityFunc) HC {
	return HC{
		similarity: similarity,
	}
}

// Predict predicts the label of an example
func (hc HC) Predict(e clus.Example) int {
	// build a group with just e
	ge := []clus.Example{e}

	// find the group with minimum similarity
	minSim := math.MaxFloat64
	var minG int

	for i, g := range hc.groups {
		sim := hc.similarity(ge, g)
		if sim < minSim {
			minSim = sim
			minG = i
		}
	}

	// Return the group with minimum similarity
	return minG
}

func (hc *HC) removeGroup(i int) {
	hc.groups[i] = hc.groups[len(hc.groups)-1]
	hc.groups = hc.groups[:len(hc.groups)-1]
}

// Fit trains the clusterer
func (hc *HC) Fit(examples []clus.Example) {
	// Initialize groups
	hc.groups = make([][]clus.Example, len(examples))
	for i := range examples {
		hc.groups[i] = make([]clus.Example, 1)
		hc.groups[i][0] = examples[i]
	}

	// Main loop
	for len(hc.groups) > 2 {
		minSim := math.MaxFloat64
		var minG1, minG2 int
		for i, g1 := range hc.groups {
			for j, g2 := range hc.groups {
				if i == j {
					continue
				}
				sim := hc.similarity(g1, g2)
				if sim < minSim {
					minSim = sim
					minG1, minG2 = i, j
				}
			}
		}
		// Join groups with min similarity
		hc.groups[minG1] = append(hc.groups[minG1], hc.groups[minG2]...)
		hc.removeGroup(minG2)
	}
}
