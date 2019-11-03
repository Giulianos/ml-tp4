package cluster

// Example represents an example
// in a clustering problem
type Example []float64

// Cluster abstracts a clustering algorithm
type Cluster interface {
	Predict(example Example) int
}
