#!/bin/bash

mkdir -p ~/.ssh
chmod 700 ~/.ssh
cp ./scripts/ssh_config ~/.ssh/config
mkdir -p /root/.ssh
chmod 700 /root/.ssh
cp ./scripts/ssh_config /root/.ssh/config

echo $GONITO_PRIVATE_SSH | base64 -d > ~/.ssh/gonito
echo $GONITO_PRIVATE_SSH | base64 -d > /root/.ssh/gonito
chmod 600 ~/.ssh/gonito
chmod 600 /root/.ssh/gonito

ssh-keyscan -H gonito.net >> ~/.ssh/known_hosts
ssh-keyscan -H gonito.net >> /root/.ssh/known_hosts

git config --global user.email "m1.borzymowski@gmail.com"
git config --global user.name "Marcin Borzymowski"
