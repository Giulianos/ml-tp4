package kohonen

import clus "github.com/Giulianos/ml-tp4/cluster"

// Kohonen implements the Cluster interface
// using Kohonen maps for clustering
type Kohonen struct {
}

// New creates a Kohonen clusterer
func New() Kohonen {
	return Kohonen{}
}

func (k Kohonen) Fit(examples []clus.Example) {

}

// Predict predicts the cluster to which
// e belongs
func (k Kohonen) Predict(e clus.Example) int {
	return 0
}
