SHELL=/bin/bash

all: ./data/in-header.tsv ./data/out-header.tsv
	dvc repro --pull

./data/in-header.tsv ./data/out-header.tsv: ./data/parsed-pdfs.json
	echo "InText" >> ./data/in-header.tsv
	echo "Labels" >> ./data/out-header.tsv

./data/parsed-pdfs.json:
	dvc pull ./data/parsed-pdfs.json

clean:
	rm -f ./data/parsed-pdfs.json
	rm -f ./data/dev/*.tsv
	rm -f ./data/test/*.tsv
	rm -f ./data/train/*.tsv
	rm -f ./data/in-header.tsv
	rm -f ./data/out-header.tsv
	rm -f ./data/dev/lm.txt
	rm -f ./data/test/lm.txt
	rm -f ./data/train/lm.txt
