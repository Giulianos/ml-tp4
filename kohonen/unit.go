package kohonen

import "math/rand"

// UnitPos represents the position of a unit
type UnitPos struct {
	row int
	col int
}

type unit struct {
	pos UnitPos
	w   []float64
}

func newUnit(dim int, pos UnitPos, rng rand.Rand) unit {
	u := unit{
		pos: pos,
		w:   make([]float64, dim),
	}

	// init random weights
	for i := range u.w {
		u.w[i] = 0.1 * rng.Float64()
	}

	return u
}
