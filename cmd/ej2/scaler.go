package main

import (
	"math"

	"github.com/Giulianos/ml-tp4/cluster"
)

func scaleExamples(xs []cluster.Example) []cluster.Example {
	zs := make([]float64, len(xs[0]))
	ss := make([]float64, len(xs[0]))

	// Calculate mean
	for _, x := range xs {
		for fi, f := range x {
			zs[fi] += f / float64(len(xs))
		}
	}

	// Calculate var
	for _, x := range xs {
		for fi, f := range x {
			ss[fi] += math.Pow(f-zs[fi], 2) / float64(len(xs))
		}
	}

	// Calculate std
	for si, s := range ss {
		ss[si] = math.Sqrt(s)
	}

	// Normalize examples
	newExamples := make([]cluster.Example, 0, len(xs))
	for _, x := range xs {
		newExamples = append(newExamples, normExample(x, zs, ss))
	}

	return newExamples
}

func normExample(x cluster.Example, zs, ss []float64) cluster.Example {
	for fi, f := range x {
		x[fi] = (f - zs[fi]) / ss[fi]
	}

	return x
}
