#!/bin/bash
set -euo pipefail

which python
which python3

cd /app

echo "Repro"
mkdir -p /app/.dvc/tmp
echo $BMARCINAI_GOOGLE_CREDENTIALS > ./.dvc/tmp/gdrive-user-credentials.json
make

echo "Created main repo"

curl -L https://gonito.net/get/bin/geval -o ./geval
chmod +x ./geval

git clone ssh://gitolite@gonito.net/marcinb/eur-lex-documents eur-lex-documents-marcinb
git clone ssh://gitolite@gonito.net/eur-lex-documents eur-lex-documents-base
cd eur-lex-documents-base

# use git annex
git-annex init
git-annex add ./train/in.tsv.xz
git-annex enableremote gonito-https
git-annex sync --content

cd ../eur-lex-documents-marcinb

git switch -c "$BRANCH_NAME"
git pull origin "$BRANCH_NAME" || true

# use git annex
git-annex init
if [ -f "./train/in.tsv.xz" ]; then
  git-annex add ./train/in.tsv.xz || true
fi
git-annex enableremote gonito-https
git-annex sync --content


rm -r ./src || true
rm -r ./docker || true

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
cp /app/requirements.txt .
cp /app/data/labels.tsv .

chmod +x ./train.sh
chmod +x ./predict.sh


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

#tr -d '\015' </app/data/dev/in.tsv >./dev-0/in.tsv
#tr -d '\015' </app/data/dev/expected.tsv >./dev-0/expected.tsv
#tr -d '\015' </app/data/test/in.tsv >./test-A/in.tsv
#tr -d '\015' </app/data/train/in.tsv >./train/in.tsv
#tr -d '\015' </app/data/train/expected.tsv >./train/expected.tsv

cp -r ../eur-lex-documents-base/dev-0/* ./dev-0
cp -r ../eur-lex-documents-base/test-A/* ./test-A
cp -r ../eur-lex-documents-base/train/* ./train

tr -d '\015' </app/data/test/out.tsv >./test-A/out.tsv
tr -d '\015' </app/data/dev/out.tsv >./dev-0/out.tsv

cp /app/data/in-header.tsv ./
cp /app/data/out-header.tsv ./

#xz ./train/in.tsv
#xz ./test-A/in.tsv
#xz ./dev-0/in.tsv
#/app/geval --validate --expected-directory .

git-annex add ./train/in.tsv.xz
git add .
git status
echo "before commit"
git commit -m "$COMMIT_MESSAGE"
echo "after commit"
git push -f origin "$BRANCH_NAME"
echo "after push"
git-annex sync --no-content --no-pull --push --all
echo "after sync"
