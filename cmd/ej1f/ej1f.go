package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"

	"github.com/Giulianos/ml-tp4/kmeans"

	clus "github.com/Giulianos/ml-tp4/cluster"
)

func loadDataFrame(filename string) ([]clus.Example, error) {
	f, e := os.Open(filename)
	if e != nil {
		return nil, e
	}
	csvReader := csv.NewReader(f)

	// If ds has header, save it
	headers, e := csvReader.Read()
	if e != nil {
		return nil, e
	}

	// Create ds
	examples := make([]clus.Example, 0)
	var i int
	for {
		// read csv record
		r, e := csvReader.Read()
		if e == io.EOF {
			break
		}
		if e != nil {
			return nil, e
		}

		// create new example
		example := make(clus.Example, len(headers)-1)
		for headerPos := range headers {
			if headerPos < len(headers)-1 {
				n, e := strconv.ParseFloat(r[headerPos], 64)
				if e != nil {
					return nil, fmt.Errorf("line %d: %e", i, e)
				}
				example[headerPos] = n
			}
		}
		examples = append(examples, example)

		// move to next example
		i++
	}

	return examples, nil
}

func main() {
	trainFilename := flag.String("train-file", "", "training dataset filename")
	flag.Parse()

	trainX, e := loadDataFrame(*trainFilename)
	if e != nil {
		log.Fatal("couldn't read training dataset")
	}

	lenAll := 4

	for _, e := range trainX {
		if len(e) != lenAll {
			log.Fatal("examples must be all the same length")
		}
	}

	// Train the model
	kmeansModel := kmeans.New(2, 1212)
	kmeansModel.Fit(trainX)

	// Output results
	fmt.Println("sex,age,cad.dur,choleste,cluster")
	for _, example := range trainX {
		predicted := kmeansModel.Predict(example)
		fmt.Printf("%f,%f,%f,%f,%d\n",
			example[0],
			example[1],
			example[2],
			example[3],
			predicted,
		)
	}
}
