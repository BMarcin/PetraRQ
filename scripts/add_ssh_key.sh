#!/bin/bash

mkdir -p ~/.ssh
chmod 700 ~/.ssh
cp ./scripts/ssh_config ~/.ssh/config

ssh-keyscan -H gonito.net
ssh-keyscan -H gonito.net >> ~/.ssh/known_hosts

echo $GONITO_PRIVATE_SSH | base64 -d > ~/.ssh/gonito
chmod 600 ~/.ssh/gonito

git config --global user.email "m1.borzymowski@gmail.com"
git config --global user.name "Marcin Borzymowski"
