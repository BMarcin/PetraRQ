#!/bin/bash

mkdir -p /home/runner/work/PetraRQ/main_repo/
echo "Created main repo"

cd /home/runner/work/PetraRQ/main_repo/
git clone ssh://gitolite@gonito.net/eur-lex-documents
cd eur-lex-documents

if [ ! -d "train" ]; then
  git init
fi

git switch -c "$BRANCH_NAME"

cp /home/runner/work/PetraRQ/PetraRQ/README.md .
cp /home/runner/work/PetraRQ/PetraRQ/config.txt .
cp /home/runner/work/PetraRQ/PetraRQ/.gitignore .

mkdir -p ./train
mkdir -p ./dev-0
mkdir -p ./test-A

cp /home/runner/work/PetraRQ/PetraRQ/data/dev/* ./dev-0/
cp /home/runner/work/PetraRQ/PetraRQ/data/test/* ./test-A/
cp /home/runner/work/PetraRQ/PetraRQ/data/train/* ./train/
mv /home/runner/work/PetraRQ/PetraRQ/data/in-header.tsv ./
mv /home/runner/work/PetraRQ/PetraRQ/data/out-header.tsv ./
rm ./test-A/expected.tsv

gzip ./train/in.tsv
gzip ./train/expected.tsv
gzip ./test-A/in.tsv
gzip ./dev/in.tsv
gzip ./dev/expected.tsv

geval --validate --expected-directory .

git add .
git commit -am $COMMIT_MESSAGE
#git remote add origin ssh://gitolite@gonito.net/eur-lex-documents
git push origin "$BRANCH_NAME"
