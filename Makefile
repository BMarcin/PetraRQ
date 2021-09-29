SHELL=/bin/bash

./data/in-header.tsv: ./data/parsed-pdfs.json
	echo -e "date    content\n" > ./data/in-header.tsv

./data/out-header.tsv: ./data/parsed-pdfs.json
	echo -e "labels\n" > ./data/out-header.tsv

./data/parsed-pdfs.json:
	dvc pull ./data/parsed-pdfs.json

clean:
	rm ./data/parsed-pdfs.json
	rm ./data/dev/*.tsv
	rm ./data/test/*.tsv
	rm ./data/train/*.tsv
	rm ./data/in-header.tsv
	rm ./data/out-header.tsv
