package hc

import clf "github.com/Giulianos/ml-tp4/classifier"

// HC is the Hierarchical Clustering
// Classifier interface implementation
type HC struct {
}

// Classify classifies an example
func (hc HC) Classify(e clf.Example) string {
	return ""
}

// GetTargetAttribute returns the target
// attribute of the classifier
func (hc HC) GetTargetAttribute() string {
	return ""
}
