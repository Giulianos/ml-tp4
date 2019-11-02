package knn

import clf "github.com/Giulianos/ml-tp4/classifier"

type neighbor struct {
	example  *clf.Example
	class    *string
	distance float64
}

// NeighborHeap represents a heap data structure
// of Neighbors (of the same example). It's ordered
// by the distance of each neighbor.
type NeighborHeap []neighbor

func (nh NeighborHeap) Len() int { return len(nh) }

func (nh NeighborHeap) Less(i, j int) bool { return nh[i].distance > nh[j].distance }

func (nh NeighborHeap) Swap(i, j int) { nh[i], nh[j] = nh[j], nh[i] }

// Push adds a neighbor to the end of the underlying storage data structure
func (nh *NeighborHeap) Push(x interface{}) { *nh = append(*nh, x.(neighbor)) }

// Pop removes and retrieves a neighbor form the end of the underlying storage data structure
func (nh *NeighborHeap) Pop() interface{} {
	old := *nh
	n := len(old)
	ret := old[n-1]
	*nh = old[0 : n-1]
	return ret
}
