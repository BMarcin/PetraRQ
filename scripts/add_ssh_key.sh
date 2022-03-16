#!/bin/bash

mkdir -p ~/.ssh
chmod 700 ~/.ssh
cp ./scripts/ssh_config ~/.ssh/config

echo $GONITO_PRIVATE_SSH | base64 -d > ~/.ssh/gonito
chmod 600 ~/.ssh/gonito

ssh-keyscan -oStrictHostKeyChecking=no gonito.net >> ~/.ssh/known_hosts

git config --global user.email "m1.borzymowski@gmail.com"
git config --global user.name "Marcin Borzymowski"
