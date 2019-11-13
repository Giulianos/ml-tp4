package kohonen

import (
	"math"

	"github.com/Giulianos/ml-tp4/cluster"
)

// DecayFunc computes the decayed value of
// any parameter at iteration t.
type DecayFunc func(t int) float64

// NeighborhoodKernelFunc computes the
// kernel function between w (winner) unit position
// and n (neighbor) unit position
// at iteration t.
type NeighborhoodKernelFunc func(t int, w, n UnitPos) float64

// DifferenceFunc computes the difference (error) between the example
// e and the weights w
type DifferenceFunc func(e cluster.Example, w []float64) float64

// ExpDecay is an exponential decayed
// function starting from x0 and decaying
// at r ratio.
func ExpDecay(x0 float64, r float64) DecayFunc {
	return func(t int) float64 {
		return x0 * math.Exp(-1*float64(t)*r)
	}
}

func DefaultKernelFunc(t int, w, n UnitPos) float64 {
	d := math.Pow(float64(w.row)-float64(n.row), 2) + math.Pow(float64(w.col)-float64(n.col), 2)
	sig := 0.1
	return math.Exp(-1 * d / 2 * sig)
}
