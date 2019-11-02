package kmeans

import clf "github.com/Giulianos/ml-tp4/classifier"

// KMeans is the K-Means
// Classifier interface implementation
type KMeans struct {
}

// Classify classifies an example
func (km KMeans) Classify(e clf.Example) string {
	return ""
}

// GetTargetAttribute returns the target
// attribute of the classifier
func (km KMeans) GetTargetAttribute() string {
	return ""
}
