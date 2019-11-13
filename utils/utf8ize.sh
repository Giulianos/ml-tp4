#!/bin/bash

## This tool converts text files
## from Windows CP1252 to UTF-8 inplace
## Author: Giuliano Scaglioni (2016)

# Convert files
for FILENAME in "$@"
do
	cat "$FILENAME"|iconv -f "CP1252" -t "UTF-8">"$FILENAME.new" 2> /dev/null &&
	mv -f "$FILENAME.new" "$FILENAME" || 
	rm -f "$FILENAME.new" 2> /dev/null
done
echo "DONE!"
