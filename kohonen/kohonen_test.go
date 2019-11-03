package kohonen

import (
	"math"
	"math/rand"
	"os"
	"testing"

	clus "github.com/Giulianos/ml-tp4/cluster"
)

func TestPredict(t *testing.T) {

	distanceFunc := func(e clus.Example, w []float64) float64 {
		var squared float64
		for i, v := range e {
			squared += math.Pow(v-w[i], 2)
		}
		return math.Sqrt(squared)
	}

	kernelFunc := func(t int, w, n UnitPos) float64 {
		d := math.Pow(float64(w.row)-float64(n.row), 2) + math.Pow(float64(w.col)-float64(n.col), 2)
		sig := 0.1
		return math.Exp(-1 * d / 2 * sig)
	}

	kohonen := New(
		ExpDecay(0.1, 0),
		kernelFunc,
		distanceFunc,
		*rand.New(rand.NewSource(10)),
	)

	X := []clus.Example{
		{0, 0},
		{1, 0},
		{0, 1},
		{1, 1},
		{4, 4},
		{5, 4},
		{4, 5},
		{5, 5},
	}

	kohonen.Fit(X)

	kohonen.PrintWeights(os.Stdout)

	label1 := kohonen.Predict(clus.Example{0.5, 0.5})
	label2 := kohonen.Predict(clus.Example{4.5, 4.5})

	if label1 == label2 {
		t.Error("label1 should be different than label2")
	}
}
