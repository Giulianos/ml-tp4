package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"log"
	"math/rand"
	"os"

	"github.com/Giulianos/ml-tp2/classifier"

	"github.com/Giulianos/ml-tp2/marshalling"
)

func randomSplit(examples []classifier.Example, testPortion float64) (train []classifier.Example, test []classifier.Example) {
	rand.Shuffle(len(examples), func(i, j int) {
		examples[i], examples[j] = examples[j], examples[i]
	})

	testSize := int(float64(len(examples)) * testPortion)

	return examples[testSize:], examples[0:testSize]
}

func writeSet(examples []classifier.Example, filename string) error {
	// Open file to write
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	writer := bufio.NewWriter(file)
	csvWriter := csv.NewWriter(writer)
	marshalling.MarshallCSV(examples, *csvWriter)
	err = writer.Flush()
	if err != nil {
		return err
	}
	err = file.Close()
	if err != nil {
		return err
	}

	return nil
}

func main() {
	// Read flags
	dsFilename := flag.String("ds", "", "input filename for dataset to split, stdin if empty")
	trainFilename := flag.String("train", "train.csv", "output filename for train set")
	testFilename := flag.String("test", "test.csv", "output filename for test set")
	customSeed := flag.Int64("seed", 1207, "custom seed for random split")
	testSize := flag.Float64("test-size", 0.1, "percentage of examples for testing [0, 1]")
	flag.Parse()

	// Set seed
	rand.Seed(*customSeed)

	// Open file to read
	var dsFile *os.File
	if *dsFilename == "" {
		dsFile = os.Stdin
	} else {
		var err error
		dsFile, err = os.Open(*dsFilename)
		if err != nil {
			log.Fatal(err)
		}
	}
	dsReader := bufio.NewReader(dsFile)
	dsCSVReader := csv.NewReader(dsReader)

	// Load entire dataset into memory (naive implementation for relatively small files)
	examples, err := marshalling.UnmarshallCSV(*dsCSVReader)
	if err != nil {
		log.Fatal(err)
	}

	// Split set in training and test
	training, test := randomSplit(examples, *testSize)

	// Save files
	err = writeSet(training, *trainFilename)
	if err != nil {
		log.Fatal(err)
	}
	err = writeSet(test, *testFilename)
	if err != nil {
		log.Fatal(err)
	}
}
