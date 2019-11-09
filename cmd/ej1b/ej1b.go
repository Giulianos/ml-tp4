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
	testFilename := flag.String("test-file", "", "testing dataset filename")
	dropSex :=
		flag.Parse()

	xTrain, yTrain, err := loadCSV(trainFilename)
	if err != nil {
		log.Fatal(err)
	}
	xTest, yTest, err := loadCSV(testFilename)
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

	// Benchmark model
	var tp, tn, fp, fn int

	for i, e := range xTest {
		pred, err := model.Predict(e)

		expected := yTest[i]
		actual := pred[0] > 0.6

		if err != nil {
			log.Fatal(fmt.Errorf("benchmark: %e", err))
		}

		if actual && expected == 1 {
			tp++
		}
		if actual && expected == 0 {
			fp++
		}
		if !actual && expected == 1 {
			fn++
		}
		if !actual && expected == 0 {
			tn++
		}
	}

	// Output results
	fmt.Println("tn,tp,fp,fn")
	fmt.Printf("%d,%d,%d,%d\n", tn, tp, fp, fn)
}
