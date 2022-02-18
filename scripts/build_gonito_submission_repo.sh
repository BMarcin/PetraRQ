#!/bin/bash

sudo apt-get update && sudo apt-get install -y xz-utils pacman
pacman -S git-annex

mkdir -p /home/runner/work/PetraRQ/main_repo/
echo "Created main repo"

sudo curl -L https://gonito.net/get/bin/geval -o /usr/local/bin/geval
sudo chmod +x /usr/local/bin/geval

cd /home/runner/work/PetraRQ/main_repo/

#git config --global pack.windowMemory "100m"
#git config --global pack.SizeLimit "100m"
#git config --global pack.threads "1"
#git config --global pack.window "0"

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

# use git annex
git annex init
git annex add ./train/in.tsv.xz
git annex enableremote gonito-https
git annex sync --content

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

xz ./train/in.tsv
#gzip ./train/expected.tsv
xz ./test-A/in.tsv
#gzip ./test-A/expected.tsv
#gzip ./test-A/out.tsv
xz ./dev-0/in.tsv
#gzip ./dev-0/expected.tsv
#gzip ./dev-0/out.tsv

#mv ./test-A/expected.tsv ../expected.tsv

tree

geval --validate --expected-directory .

git remote rm origin
git remote add origin ssh://gitolite@gonito.net/marcinb/eur-lex-documents

git add .
git status
git commit -m "$COMMIT_MESSAGE"
git push origin "$BRANCH_NAME"

