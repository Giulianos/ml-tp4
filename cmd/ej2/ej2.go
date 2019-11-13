package main

import "fmt"

func main() {
	texts := readDirectoryTexts("something")
	df, labels := textToFeatures(texts)
	fmt.Println(labels)
	for _, r := range df {
		fmt.Println(r)
	}
}
