package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/Giulianos/ml-tp4/cluster"
	"github.com/Giulianos/ml-tp4/hc"
	"github.com/Giulianos/ml-tp4/kmeans"
	"github.com/Giulianos/ml-tp4/kohonen"
)

// Kohonen functions
var distanceFunc = func(e cluster.Example, w []float64) float64 {
	var squared float64
	for i, v := range e {
		squared += math.Pow(v-w[i], 2)
	}
	return math.Sqrt(squared)
}

func mainPredict() {
	textDir := flag.String("dir", "", "texts directory")
	modName := flag.String("model", "kmeans", "ml model to use")
	flag.Parse()

	xsText, ys := readDirectoryTexts(*textDir)
	xs, _ := textToFeatures(xsText)

	var model cluster.Cluster

	switch *modName {
	case "kmeans":
		log.Println("running kmeans")
		kmModel := kmeans.New(5, 10)
		kmModel.Fit(scaleExamples(xs))
		model = kmModel
	case "hc":
		log.Println("running hierarchical clustering")
		hcModel := hc.New(hc.CentroidSimilarity, 5)
		hcModel.Fit(xs)
		model = hcModel
	case "kohonen":
		log.Println("running kohonen")
		kModel := kohonen.New(
			25, 25,
			kohonen.ExpDecay(0.1, 0),
			kohonen.DefaultKernelFunc,
			distanceFunc,
			*rand.New(rand.NewSource(10)),
		)
		kModel.Fit(scaleExamples(xs), 10000)
		model = kModel
	}

	// Predict labels
	fmt.Println("author,predCluster")
	for i, x := range xs {
		p := model.Predict(x)
		fmt.Printf("%s,%d\n", ys[i], p)
	}
}

func mainFeatureExtraction() {
	textDir := flag.String("dir", "", "texts directory")
	flag.Parse()

	xsText, _ := readDirectoryTexts(*textDir)
	xs, featureLabels := textToFeatures(xsText)

	for i, v := range featureLabels {
		if i < len(featureLabels)-1 {
			fmt.Printf("%s,", v)
		} else {
			fmt.Println(v)
		}
	}

	for _, x := range xs {
		for i, v := range x {
			if i < len(x)-1 {
				fmt.Printf("%f,", v)
			} else {
				fmt.Println(v)
			}
		}
	}
}

func main() {
	mainPredict()
}
