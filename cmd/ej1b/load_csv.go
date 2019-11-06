package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

func loadCSV(filename string) ([][]float64, []float64, error) {
	_, err := os.Stat(filename)
	if err != nil {
		return nil, nil, err
	}

	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var xs [][]float64
	var ys []float64

	// save header to know fields quantity
	headers, err := reader.Read()
	if err != nil {
		return nil, nil, err
	}

	var line int
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, fmt.Errorf("line %d: %e", line, err)
		}

		x := make([]float64, len(headers)-2)
		for i, v := range record {
			f, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return nil, nil, fmt.Errorf("line %d, value: %s: %e", line, v, err)
			}
			if i == len(headers)-1 {
				ys = append(ys, f)
			} else if i != 0 {
				x[i-1] = f
			}
		}
		xs = append(xs, x)
		line++
	}

	return xs, ys, nil
}
