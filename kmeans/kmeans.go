package kmeans

import clf "github.com/Giulianos/ml-tp4/classifier"

// KMeans is the K-Means
// Classifier interface implementation
type KMeans struct {
}

// New creates a new KMeans implementation
// of the Classifier interface.
func New() KMeans {
	return KMeans{}
}

// Fits trains the classifier
func (km *KMeans) Fit(examples []clf.Example) {

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
