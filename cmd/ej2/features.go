package main

import (
	"container/heap"
	"regexp"
	"strings"
)

func textToFeatures(texts []string) ([][]float64, []string) {
	labels := []string{
		"sentWcAvg",
		"mostRepFreqSum",
		"diffWc",
		"subordFreq",
		"coordFreq",
		"detArtFreq",
		"indetArtFreq",
	}

	df := make([][]float64, 0, len(texts))

	for _, t := range texts {
		features := make([]float64, 0, len(labels))
		features = append(features,
			averageSentenceWords(t),
			modeWordsFreqSum(t),
			relativeDifferentWords(t),
			relSubordConjOccurrences(t),
			relCoordConjOccurrences(t),
			relDetArtOccurrences(t),
			relIndetArtOccurrences(t),
		)
		df = append(df, features)
	}

	return df, labels
}

func averageSentenceWords(text string) float64 {
	// split in sentences
	sentences := strings.Split(sanitizeText(text), ".")

	// count how many words on each sentence
	wordCounts := make([]float64, 0, len(sentences))
	var textWordCount float64
	for _, sentence := range sentences {
		sentenceLen := float64(len(strings.Split(sentence, " ")))
		if sentenceLen < 2 {
			continue
		}
		wordCounts = append(wordCounts, sentenceLen)
		textWordCount += sentenceLen
	}

	// get average and make it relative to text word count
	var avg float64
	for _, wc := range wordCounts {
		avg += wc / float64(len(wordCounts)) / textWordCount
	}

	return avg
}

// WordHeap
type wordFreq struct {
	word string
	freq float64
}

type wordHeap []wordFreq

func (wh wordHeap) Len() int { return len(wh) }

func (wh wordHeap) Less(i, j int) bool { return wh[i].freq < wh[j].freq }

func (wh wordHeap) Swap(i, j int) { wh[i], wh[j] = wh[j], wh[i] }

func (wh *wordHeap) Push(x interface{}) { *wh = append(*wh, x.(wordFreq)) }

func (wh *wordHeap) Pop() interface{} {
	old := *wh
	n := len(old)
	ret := old[n-1]
	*wh = old[0 : n-1]
	return ret
}

func modeWordsFreqSum(text string) float64 {
	// calculate freq of each word
	freqs := make(map[string]float64)
	var wc float64
	procText := removePunctuation(sanitizeText(text))
	for _, word := range strings.Split(procText, " ") {
		freqs[strings.ToLower(word)] += 1
		wc++
	}

	// calculate relative frequencies
	for k := range freqs {
		freqs[k] /= wc
	}

	// get 5 most repeated
	wHeap := &wordHeap{}
	heap.Init(wHeap)

	for w, f := range freqs {
		heap.Push(wHeap, wordFreq{word: w, freq: f})

		if wHeap.Len() > 5 {
			heap.Pop(wHeap)
		}
	}

	// return the sum of the frequencies
	var sum float64
	for _, wf := range *wHeap {
		sum += wf.freq
	}

	return sum
}

// sanitizeText converts it to lowercase and removes
// accents
func sanitizeText(text string) string {
	accReplacer := strings.NewReplacer(
		"á", "a", "Á", "a",
		"é", "e", "É", "e",
		"í", "i", "Í", "i",
		"ó", "o", "Ó", "o",
		"ú", "u", "Ú", "u",
	)

	return strings.ToLower(accReplacer.Replace(text))
}

func removePunctuation(text string) string {
	reg, _ := regexp.Compile("[^a-zA-Z0-9ñ\\s]+")

	return reg.ReplaceAllString(text, "")
}

func relativeDifferentWords(text string) float64 {
	procText := removePunctuation(sanitizeText(text))

	words := make(map[string]bool)
	var wc float64
	for _, w := range strings.Split(procText, " ") {
		words[w] = true
		wc++
	}

	return float64(len(words)) / wc
}

var subordConj = []string{
	"porque",
	"pues",
	"ya que",
	"puesto que",
	"a causa de",
	"debido a",
	"luego",
	"conque",
	"asi que",
	"si",
	"para que",
	"a fin de que",
	"como",
	"que",
	"aunque",
	"aun cuando",
	"si bien",
}

var coordConj = []string{
	"ni",
	"o",
	"o bien",
	"pero aunque",
	"no obstante",
	"sin embargo",
	"sino",
	"por el contrario",
}

var detArt = []string{
	"la",
	"el",
	"los",
	"las",
}

var indetArt = []string{
	"un",
	"una",
	"unos",
	"unas",
}

func relCoordConjOccurrences(text string) float64 {
	return relativeWordsAppearence(text, coordConj)
}

func relSubordConjOccurrences(text string) float64 {
	return relativeWordsAppearence(text, subordConj)
}

func relDetArtOccurrences(text string) float64 {
	return relativeWordsAppearence(text, detArt)
}

func relIndetArtOccurrences(text string) float64 {
	return relativeWordsAppearence(text, indetArt)
}

func relativeWordsAppearence(text string, words []string) float64 {
	var twc, wc float64

	for _, tw := range strings.Split(removePunctuation(sanitizeText(text)), " ") {
		for _, w2 := range words {
			if tw == w2 {
				wc++
			}
		}
		twc++
	}

	return wc / twc
}
