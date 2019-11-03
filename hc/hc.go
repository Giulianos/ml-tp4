package hc

import clf "github.com/Giulianos/ml-tp4/classifier"

// HC is the Hierarchical Clustering
// Classifier interface implementation
type HC struct {
}

// New creates a new Hierarchical Clustering classifier
func New() HC {
	return HC{}
}

// Classify classifies an example
func (hc HC) Classify(e clf.Example) string {
	return ""
}

// Fit trains the classifier
func (hc *HC) Fit(examples []clf.Example) {

}

// GetTargetAttribute returns the target
// attribute of the classifier
func (hc HC) GetTargetAttribute() string {
	return ""
}
