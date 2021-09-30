#!/bin/bash

mkdir -p ~/.ssh
cp ./scripts/ssh_config ~/.ssh/config

echo $GONITO_PRIVATE_SSH > ~/.ssh/gonito
chmod 600 ~/.ssh/gonito
