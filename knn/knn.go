package knn

import clf "github.com/Giulianos/ml-tp4/classifier"

// KNN is the K-Nearest Neighbour
// Classifier interface implementation
type KNN struct {
}

// Classify classifies an example
func (knn KNN) Classify(e clf.Example) string {
	return ""
}

// GetTargetAttribute returns the target
// attribute of the classifier
func (knn KNN) GetTargetAttribute() string {
	return ""
}
