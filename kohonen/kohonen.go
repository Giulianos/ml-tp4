package kohonen

import (
	"math"
	"math/rand"

	clus "github.com/Giulianos/ml-tp4/cluster"
)

// Kohonen implements the Cluster interface
// using Kohonen maps for clustering
type Kohonen struct {
	lr          DecayFunc
	h           NeighborhoodKernelFunc
	dif         DifferenceFunc
	units       map[UnitPos]unit
	latticeCols int
	latticeRows int
	rng         rand.Rand
	dim         int
}

// New creates a Kohonen clusterer
func New(rows, cols int, lr DecayFunc, h NeighborhoodKernelFunc, dif DifferenceFunc, rng rand.Rand) Kohonen {
	return Kohonen{
		lr:          lr,
		h:           h,
		dif:         dif,
		rng:         rng,
		latticeRows: rows,
		latticeCols: cols,
	}
}

// Fit trains the clusterer
func (k *Kohonen) Fit(examples []clus.Example, maxIter int) {
	// examples dimension
	k.dim = len(examples[0])

	// init lattice
	k.units = make(map[UnitPos]unit, k.latticeCols*k.latticeRows)
	for row := 0; row < k.latticeRows; row++ {
		for col := 0; col < k.latticeCols; col++ {
			pos := UnitPos{row, col}
			k.units[pos] = newUnit(k.dim, pos, k.rng)
		}
	}

	// main loop
	for t := 0; t < maxIter; t++ {
		x := examples[k.rng.Intn(len(examples))]

		// find BMU
		bmu := k.findBMU(x)

		// get neighborhood
		neighborhoodPos := k.getNeighborhood(bmu)

		// update neighborhood weights
		for _, nPos := range neighborhoodPos {
			k.updateUnitWeights(nPos, bmu, x, t)
		}
	}
}

func (k Kohonen) updateUnitWeights(uPos, bmuPos UnitPos, e clus.Example, t int) {
	for wIdx := range k.units[uPos].w {
		wn := k.units[uPos].w[wIdx]
		k.units[uPos].w[wIdx] += k.lr(t) * k.h(t, k.units[uPos].pos, bmuPos) * (e[wIdx] - wn)
	}
}

func (k Kohonen) findBMU(e clus.Example) UnitPos {
	minDif := math.MaxFloat64
	var minPos UnitPos
	for row := 0; row < k.latticeRows; row++ {
		for col := 0; col < k.latticeCols; col++ {
			pos := UnitPos{row, col}
			dif := k.dif(e, k.units[pos].w)
			if dif < minDif {
				minDif = dif
				minPos = pos
			}
		}
	}

	return minPos
}

// getNeighbors returns the list of neighbors positions for unit,
// including the position of unit itself
func (k Kohonen) getNeighborhood(unitPos UnitPos) []UnitPos {
	neighbors := make([]UnitPos, 4)
	maxDist := 1
	for drow := -1 * maxDist; drow < maxDist; drow++ {
		newRow := unitPos.row + drow

		if newRow < 0 || newRow >= k.latticeRows {
			continue
		}

		for dcol := -1 * maxDist; dcol < maxDist; dcol++ {
			newCol := unitPos.col + dcol

			if newCol < 0 || newCol >= k.latticeCols {
				continue
			}

			neighbors = append(neighbors, UnitPos{newRow, newCol})
		}
	}

	return neighbors
}

func (k Kohonen) unitPosToLabel(up UnitPos) int {
	return up.row*k.latticeCols + up.col
}

// Predict predicts the cluster to which
// e belongs
func (k Kohonen) Predict(e clus.Example) int {
	return k.unitPosToLabel(
		k.findBMU(e),
	)
}
