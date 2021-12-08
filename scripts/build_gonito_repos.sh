#!/bin/bash

mkdir -p /home/runner/work/PetraRQ/main_repo/
echo "Created main repo"

sudo curl -L https://gonito.net/get/bin/geval -o /usr/local/bin/geval
sudo chmod +x /usr/local/bin/geval

cd /home/runner/work/PetraRQ/main_repo/
git clone ssh://gitolite@gonito.net/eur-lex-documents
cd eur-lex-documents

git switch -c "$BRANCH_NAME"
#git branch --set-upstream-to=origin/"$BRANCH_NAME" "$BRANCH_NAME"

cp /home/runner/work/PetraRQ/PetraRQ/README.md .
cp /home/runner/work/PetraRQ/PetraRQ/config.txt .
cp /home/runner/work/PetraRQ/PetraRQ/.gitignore .

mkdir -p ./train
mkdir -p ./dev-0
mkdir -p ./test-A

if [ -f "./train/in.tsv.gz" ]; then
  rm ./train/in.tsv.gz
fi

if [ -f "./train/expected.tsv.gz" ]; then
  rm ./train/expected.tsv.gz
fi

if [ -f "./dev-0/in.tsv.gz" ]; then
  rm ./dev-0/in.tsv.gz
fi

if [ -f "./dev-0/expected.tsv.gz" ]; then
  rm ./dev-0/expected.tsv.gz
fi

if [ -f "./dev-0/out.tsv.gz" ]; then
  rm ./dev-0/out.tsv.gz
fi

if [ -f "./test-A/in.tsv.gz" ]; then
  rm ./test-A/in.tsv.gz
fi

if [ -f "./test-A/expected.tsv.gz" ]; then
  rm ./test-A/expected.tsv.gz
fi

if [ -f "./test-A/out.tsv.gz" ]; then
  rm ./test-A/out.tsv.gz
fi

#cp /home/runner/work/PetraRQ/PetraRQ/data/dev/* ./dev-0/
#cp /home/runner/work/PetraRQ/PetraRQ/data/test/* ./test-A/
#cp /home/runner/work/PetraRQ/PetraRQ/data/train/* ./train/
tr -d '\015' </home/runner/work/PetraRQ/PetraRQ/data/dev/in.tsv >./dev-0/in.tsv
tr -d '\015' </home/runner/work/PetraRQ/PetraRQ/data/dev/expected.tsv >./dev-0/expected.tsv
tr -d '\015' </home/runner/work/PetraRQ/PetraRQ/data/dev/out.tsv >./dev-0/out.tsv

tr -d '\015' </home/runner/work/PetraRQ/PetraRQ/data/test/in.tsv >./test-A/in.tsv
tr -d '\015' </home/runner/work/PetraRQ/PetraRQ/data/test/expected.tsv >./test-A/expected.tsv
tr -d '\015' </home/runner/work/PetraRQ/PetraRQ/data/test/out.tsv >./test-A/out.tsv

tr -d '\015' </home/runner/work/PetraRQ/PetraRQ/data/train/in.tsv >./train/in.tsv
tr -d '\015' </home/runner/work/PetraRQ/PetraRQ/data/train/expected.tsv >./train/expected.tsv


mv /home/runner/work/PetraRQ/PetraRQ/data/in-header.tsv ./
mv /home/runner/work/PetraRQ/PetraRQ/data/out-header.tsv ./

gzip ./train/in.tsv
gzip ./train/expected.tsv
gzip ./test-A/in.tsv
gzip ./test-A/expected.tsv
gzip ./test-A/out.tsv
gzip ./dev-0/in.tsv
gzip ./dev-0/expected.tsv
gzip ./dev-0/out.tsv

geval --validate --expected-directory .

mv ./test-A/expected.tsv.gz ../expected.tsv.gz

tree

git add .
git status
git commit -m "$COMMIT_MESSAGE"
git push -f origin "$BRANCH_NAME"

git remote rm origin
git remote add origin ssh://gitolite@gonito.net/eur-lex-documents-dont-peek

mv ../expected.tsv.gz ./test-A/expected.tsv.gz

tree

git add ./test-A/expected.tsv.gz
git status
git commit -m "$COMMIT_MESSAGE"
git push -f origin "$BRANCH_NAME"


#git commit -am $COMMIT_MESSAGE
##git remote add origin ssh://gitolite@gonito.net/eur-lex-documents
#git push origin "$BRANCH_NAME"
