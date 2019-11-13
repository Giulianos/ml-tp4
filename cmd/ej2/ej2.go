package main

import (
	"flag"
	"fmt"
)

func main() {
	textDir := flag.String("dir", "", "texts directory")
	flag.Parse()

	_, authors := readDirectoryTexts(*textDir)
	// _, _ := textToFeatures(texts)
	for _, a := range authors {
		fmt.Println(a)
	}
}
