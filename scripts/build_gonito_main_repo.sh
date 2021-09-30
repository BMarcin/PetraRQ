#!/bin/bash
#/home/runner/work/PetraRQ/PetraRQ
sudo mkdir -p /home/runner/work/main_repo
echo "Created main repo"

cd /home/runner/work/main_repo
git clone ssh://gitolite@gonito.net/eur-lex-documents

cp /home/runner/work/PetraRQ/PetraRQ/README.md .
cp /home/runner/work/PetraRQ/PetraRQ/config.txt .

mkdir -p ./train
mkdir -p ./dev-0
mkdir -p ./test-A

cp /home/runner/work/PetraRQ/PetraRQ/data/dev/* ./dev-0/
cp /home/runner/work/PetraRQ/PetraRQ/data/test/* ./test-A/
cp /home/runner/work/PetraRQ/PetraRQ/data/train/* ./train/

tree