package classifier

// Example represents an example
// in a classification problem
type Example map[string]float64

// Classifier abstracts a
// feature classifier
type Classifier interface {
	Classify(example Example) string
}
