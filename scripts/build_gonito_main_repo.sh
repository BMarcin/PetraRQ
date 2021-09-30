#!/bin/bash
#/home/runner/work/PetraRQ/PetraRQ
mkdir -p /home/runner/work/PetraRQ/main_repo/
echo "Created main repo"

cd /home/runner/work/PetraRQ/main_repo/
#git clone ssh://gitolite@gonito.net/eur-lex-documents
git init

cp /home/runner/work/PetraRQ/PetraRQ/README.md .
cp /home/runner/work/PetraRQ/PetraRQ/config.txt .

mkdir -p ./train
mkdir -p ./dev-0
mkdir -p ./test-A

cp /home/runner/work/PetraRQ/PetraRQ/data/dev/* ./dev-0/
cp /home/runner/work/PetraRQ/PetraRQ/data/test/* ./test-A/
cp /home/runner/work/PetraRQ/PetraRQ/data/train/* ./train/

git add .
git commit -m "update files"
git remote add origin ssh://gitolite@gonito.net/eur-lex-documents-dont-peek
git push origin master
