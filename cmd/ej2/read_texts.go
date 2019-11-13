package main

import (
	"io/ioutil"
	"log"
	"os"
	"regexp"
	"strings"
)

func readDirectoryTexts(path string) ([]string, []string) {
	files, err := ioutil.ReadDir(path)
	if err != nil {
		log.Fatal(err)
	}
	texts := make([]string, 0, len(files))
	authors := make([]string, 0, len(files))
	for _, file := range files {
		if !file.IsDir() && file.Name()[0] != '.' {
			filename := path + "/" + file.Name()
			f, err := os.Open(filename)
			if err != nil {
				log.Printf("couldn't open %s: %e", filename, err)
				continue
			}
			t, err := ioutil.ReadAll(f)
			if err != nil {
				log.Printf("error reading %s: %e", filename, err)
				f.Close()
				continue
			}
			log.Printf("loaded: %s", file.Name())
			texts = append(texts, string(t))
			authors = append(authors, authorFromFilename(file.Name()))
			f.Close()
		}
	}
	return texts, authors
}

func authorFromFilename(fn string) string {
	fn = strings.ReplaceAll(fn, ".txt", "")
	re, _ := regexp.Compile("[^a-zA-Z]+")

	return re.ReplaceAllString(fn, "")
}
