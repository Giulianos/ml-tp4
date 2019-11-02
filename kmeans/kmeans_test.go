package kmeans

import (
	"testing"

	clf "github.com/Giulianos/ml-tp4/classifier"
)

func TestClassify(t *testing.T) {
	kmeans := New()

	X := []clf.Example{
		{"x": 0, "y": 0},
		{"x": 1, "y": 0},
		{"x": 0, "y": 1},
		{"x": 1, "y": 1},
		{"x": 4, "y": 4},
		{"x": 5, "y": 4},
		{"x": 4, "y": 5},
		{"x": 5, "y": 5},
	}

	kmeans.Fit(X)

	class1 := kmeans.Classify(clf.Example{"x": 0.5, "y": 0.5})
	class2 := kmeans.Classify(clf.Example{"x:": 4.5, "y": 4.5})

	if class1 == class2 {
		t.Error("class1 should be different than class2")
	}
}
