package hc

import clus "github.com/Giulianos/ml-tp4/cluster"

// HC is the Hierarchical Clustering
// Classifier interface implementation
type HC struct {
}

// New creates a new Hierarchical Clustering clusterer
func New() HC {
	return HC{}
}

// Predict predicts the label of an example
func (hc HC) Predict(e clus.Example) string {
	return ""
}

// Fit trains the clusterer
func (hc *HC) Fit(examples []clus.Example) {

}
