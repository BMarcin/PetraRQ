#!/bin/bash

mkdir -p /home/runner/work/PetraRQ/dontpeek_repo/
echo "Created dontpeek repo"

cd /home/runner/work/PetraRQ/dontpeek_repo/
git clone ssh://gitolite@gonito.net/eur-lex-documents-dont-peek
cd eur-lex-documents-dont-peek
#git init

cp /home/runner/work/PetraRQ/PetraRQ/README.md .
cp /home/runner/work/PetraRQ/PetraRQ/config.txt .

mkdir -p ./train
mkdir -p ./dev-0
mkdir -p ./test-A

cp /home/runner/work/PetraRQ/PetraRQ/data/dev/* ./dev-0/
cp /home/runner/work/PetraRQ/PetraRQ/data/test/* ./test-A/
cp /home/runner/work/PetraRQ/PetraRQ/data/train/* ./train/

#git add .
git commit -am $COMMIT_MESSAGE
#git remote add origin ssh://gitolite@gonito.net/eur-lex-documents-dont-peek
git push origin master
