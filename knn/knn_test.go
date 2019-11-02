package knn

import (
	"testing"

	clf "github.com/Giulianos/ml-tp4/classifier"
)

func TestGetNearest(t *testing.T) {
	knn := New(3, Distance1D("height"))

	X := []clf.Example{
		{"height": 11.4},
		{"height": 100},
		{"height": 4.3},
		{"height": 1000},
		{"height": 15.2},
		{"height": 17},
	}

	y := []string{
		"red",
		"blue",
		"red",
		"green",
		"red",
		"red",
	}

	_ = knn.Fit(X, y)

	nearestNeighbors := knn.getKNearest(clf.Example{"height": 10})

	for _, n := range nearestNeighbors {
		if n.distance > 6 {
			t.Errorf("distance should be at most 5.7, but got %f", n.distance)
		}
	}
}

func TestClassifyNotWeighted(t *testing.T) {
	knn := New(5, Distance1D("height"))

	X := []clf.Example{
		{"height": 11.4},
		{"height": 9.3},
		{"height": 100},
		{"height": 4.3},
		{"height": 150},
		{"height": 15.2},
		{"height": 8},
	}

	y := []string{
		"red",
		"red",
		"green",
		"blue",
		"green",
		"blue",
		"blue",
	}

	_ = knn.Fit(X, y)

	expected := "blue"
	actual := knn.Classify(clf.Example{"height": 10})

	if actual != expected {
		t.Errorf("class should be %s, got %s", expected, actual)
	}

}

func TestClassifyWeighted(t *testing.T) {
	knn := New(5, Distance1D("height"))
	// Set weighted contributions
	knn.SetWeighted(true)

	X := []clf.Example{
		{"height": 11.4},
		{"height": 9.3},
		{"height": 100},
		{"height": 4.3},
		{"height": 150},
		{"height": 15.2},
		{"height": 8},
	}

	y := []string{
		"red",
		"red",
		"green",
		"blue",
		"green",
		"blue",
		"blue",
	}

	_ = knn.Fit(X, y)

	expected := "red"
	actual := knn.Classify(clf.Example{"height": 10})

	if actual != expected {
		t.Errorf("class should be %s, got %s", expected, actual)
	}

}
