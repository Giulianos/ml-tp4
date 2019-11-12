package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/linear"
)

func main() {
	trainFilename := flag.String("train-file", "", "training dataset filename")
	flag.Parse()

	xTrain, yTrain, err := loadCSV(*trainFilename)
	if err != nil {
		log.Fatal(err)
	}

	// Use the goml library to create the logistic regression model
	model := linear.NewLogistic(base.BatchGA, 0.0001, 0, 10000, xTrain, yTrain)
	model.Output = os.Stderr
	err = model.Learn()
	if err != nil {
		log.Fatal(err)
	}

	// Predict the probability
	p, err := model.Predict([]float64{60, 2, 199})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Probability: %f\n", p[0])

}
