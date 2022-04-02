#!/bin/bash
set -euo pipefail

cd /app

echo "Repro"
mkdir -p ./.dvc/tmp
echo $BMARCINAI_GOOGLE_CREDENTIALS > ./.dvc/tmp/gdrive-user-credentials.json
make

echo "Created main repo"

curl -L https://gonito.net/get/bin/geval -o ./geval
chmod +x ./geval

git clone ssh://gitolite@gonito.net/eur-lex-documents
cd eur-lex-documents

git switch -c "$BRANCH_NAME"

mkdir -p ./src
mkdir -p ./docker

cp /app/README.md .
cp /app/config.txt .
cp /app/.gitignore .
cp /app/gonito.yaml .
cp /app/scripts/train.sh .
cp /app/scripts/predict.sh .
cp -r /app/src/* ./src/
cp -r /app/docker/* ./docker/

chmod +x ./train.sh
chmod +x ./predict.sh

mkdir -p ./train
mkdir -p ./dev-0
mkdir -p ./test-A

# use git annex
git-annex init
git-annex add ./train/in.tsv.xz
git-annex enableremote gonito-https
git-annex sync --content

if [ -f "./train/in.tsv.xz" ]; then
  rm ./train/in.tsv.xz
fi

if [ -f "./train/expected.tsv" ]; then
  rm ./train/expected.tsv
fi

if [ -f "./dev-0/in.tsv.xz" ]; then
  rm ./dev-0/in.tsv.xz
fi

if [ -f "./dev-0/expected.tsv" ]; then
  rm ./dev-0/expected.tsv
fi

if [ -f "./dev-0/out.tsv.gz" ]; then
  rm ./dev-0/out.tsv.gz
fi

if [ -f "./test-A/in.tsv.xz" ]; then
  rm ./test-A/in.tsv.xz
fi

if [ -f "./test-A/expected.tsv" ]; then
  rm ./test-A/expected.tsv
fi

if [ -f "./test-A/out.tsv" ]; then
  rm ./test-A/out.tsv
fi

tr -d '\015' </app/data/dev/in.tsv >./dev-0/in.tsv
tr -d '\015' </app/data/dev/expected.tsv >./dev-0/expected.tsv
tr -d '\015' </app/data/dev/out.tsv >./dev-0/out.tsv

tr -d '\015' </app/data/test/in.tsv >./test-A/in.tsv
tr -d '\015' </app/data/test/out.tsv >./test-A/out.tsv

tr -d '\015' </app/data/train/in.tsv >./train/in.tsv
tr -d '\015' </app/data/train/expected.tsv >./train/expected.tsv


mv /app/data/in-header.tsv ./
mv /app/data/out-header.tsv ./

xz ./train/in.tsv
xz ./test-A/in.tsv
xz ./dev-0/in.tsv
/app/geval --validate --expected-directory .

git remote rm origin
git remote add origin ssh://gitolite@gonito.net/marcinb/eur-lex-documents

git add .
git status
git commit -m "$COMMIT_MESSAGE"
git push -f origin "$BRANCH_NAME"