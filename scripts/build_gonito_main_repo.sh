#!/bin/bash

mkdir -p /home/runner/work/PetraRQ/main_repo/
echo "Created main repo"

cd /home/runner/work/PetraRQ/main_repo/
git clone ssh://gitolite@gonito.net/eur-lex-documents
cd eur-lex-documents
#git init

cp /home/runner/work/PetraRQ/PetraRQ/README.md .
cp /home/runner/work/PetraRQ/PetraRQ/config.txt .

mkdir -p ./train
mkdir -p ./dev-0
mkdir -p ./test-A

cp /home/runner/work/PetraRQ/PetraRQ/data/dev/* ./dev-0/
cp /home/runner/work/PetraRQ/PetraRQ/data/test/* ./test-A/
cp /home/runner/work/PetraRQ/PetraRQ/data/train/* ./train/
rm ./test-A/expected.tsv

#git add .
git commit -am $COMMIT_MESSAGE
#git remote add origin ssh://gitolite@gonito.net/eur-lex-documents
git push origin master
