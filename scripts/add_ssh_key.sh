#!/bin/bash

whoami
pwd

#mkdir -p ~/.ssh
#chmod 700 ~/.ssh
#cp ./scripts/ssh_config ~/.ssh/config

# fix for self hosted
mkdir -p /root/.ssh
chmod 700 /root/.ssh
cp ./scripts/ssh_config /root/.ssh/config

#ssh-keyscan -H gonito.net >> ~/.ssh/known_hosts
# fix for self hosted
ssh-keyscan -H gonito.net >> /root/.ssh/known_hosts

git config --global user.email "m1.borzymowski@gmail.com"
git config --global user.name "Marcin Borzymowski"
