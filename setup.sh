#!/bin/bash

# Install python2.7 and python 3.5
sudo apt update
sudo apt install python python-dev python3 python3-dev

wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo python3 get-pip.py

# Install virtualenv
sudo pip install virtualenv
sudo pip install virtualenvwrapper

echo "### User specific Configs ###" >> .bashrc
echo "export LC_ALL='en_US.UTF-8'" >> .bashrc
echo "export WORKON_HOME=$HOME/.virtualenvs" >> .bashrc
echo "export PROJECT_HOME=$HOME/Devel" >> .bashrc
echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> .bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> .bashrc
echo "" >> .bashrc
source ~/.bashrc

# Install StarCraft II API binary
sudo apt-get install unzip
sudo apt-get install expect
mkdir -p .local/{lib,bin} && cd .local/lib
wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.3.16.1.zip

spawn unzip SC2.3.16.1.zip
expect "password:"
send "iagreetotheeula\n";
interact
rm SC2.3.16.1.zip

cd StarCraftII/Maps
wget http://blzdistsc2-a.akamaihd.net/MapPacks/Melee.zip
spawn unzip Melee.zip
expect "password:"
send "iagreetotheeula\n";
interact
rm -rf Melee.zip

wget https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip
unzip mini_games.zip && rm -rf mini_games.zip

cd $HOME
echo "# StarCraftII path" >> .bashrc
echo "export PATH=$HOME/.local/lib:$PATH" >> .bashrc
echo "export SC2PATH=$HOME/.local/lib/StarCraftII" >> .bashrc

# Create virtualenv
mkvirtualenv --python=python3 pysc2
pip install --user pysc2==1.2
pip install --user tensorflow

# Install git and tmux
sudo apt update
sudo apt install git-all
sudo apt-get install tmux
