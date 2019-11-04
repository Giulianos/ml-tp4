package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"

	"github.com/Giulianos/ml-tp4/knn"

	clf "github.com/Giulianos/ml-tp4/classifier"
)

func loadDataFrame(filename string) ([]clf.Example, []string, error) {
	f, e := os.Open(filename)
	if e != nil {
		return nil, nil, e
	}
	csvReader := csv.NewReader(f)

	// If ds has header, save it
	headers, e := csvReader.Read()
	if e != nil {
		return nil, nil, e
	}

	// Create ds
	examples := make([]clf.Example, 10)
	targets := make([]string, 10)
	var i int
	for {
		// read csv record
		r, e := csvReader.Read()
		if e == io.EOF {
			break
		}
		if e != nil {
			return nil, nil, e
		}

		// create new example
		example := make(clf.Example, len(headers))
		for headerPos, header := range headers {
			if headerPos == len(headers)-1 {
				targets = append(targets, r[headerPos])
			} else {
				n, e := strconv.ParseFloat(r[headerPos], 64)
				if e != nil {
					return nil, nil, fmt.Errorf("line %d: %e", i, e)
				}
				example[header] = n
			}
		}
		examples = append(examples, example)

		// move to next example
		i++
	}

	return examples, targets, nil
}

func main() {
	trainFilename := flag.String("train-file", "", "training dataset filename")
	testFilename := flag.String("test-file", "", "testing dataset filename")
	flag.Parse()

	trainX, trainY, e := loadDataFrame(*trainFilename)
	if e != nil {
		log.Fatal("couldn't read training dataset")
	}
	testX, testY, e := loadDataFrame(*testFilename)
	if e != nil {
		log.Fatal("couldn't read training dataset")
	}

	// Train the model
	knnModel := knn.New(5, knn.DistanceEuclideanAll)
	e = knnModel.Fit(trainX, trainY)

	if e != nil {
		log.Fatalf("error training model: %e", e)
	}

	// Benchmark model
	var tp, tn, fp, fn int

	for i, e := range testX {
		expected := testY[i]
		actual := knnModel.Classify(e)

		if actual == "1" && expected == "1" {
			tp++
		}
		if actual == "1" && expected == "0" {
			fp++
		}
		if actual == "0" && expected == "1" {
			fn++
		}
		if actual == "0" && expected == "0" {
			tn++
		}
	}

	// Output results
	log.Printf("TP: %d", tp)
	log.Printf("FP: %d", tn)
	log.Printf("FN: %d", fn)
	log.Printf("TN: %d", tn)
}
