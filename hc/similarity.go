package hc

import clf "github.com/Giulianos/ml-tp4/classifier"

// SimilarityFunc is a function that returns the
// similarity of two groups of examples.
type SimilarityFunc func(g1, g2 []clf.Example)
