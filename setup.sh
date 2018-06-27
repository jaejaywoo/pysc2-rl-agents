#!/bin/bash

# Install python2.7 and python 3.5
cd $HOME
sudo apt update
sudo apt-get install tmux
sudo apt install python python-dev python3 python3-dev

wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo python3 get-pip.py

# Install virtualenv
sudo pip install virtualenv
sudo pip install virtualenvwrapper

cd $HOME
echo "### User specific Configs ###" >> .bashrc
echo "export LC_ALL='en_US.UTF-8'" >> .bashrc
echo "export WORKON_HOME=$HOME/.virtualenvs" >> .bashrc
echo "export PROJECT_HOME=$HOME/Devel" >> .bashrc
echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> .bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> .bashrc
echo "" >> .bashrc

# Add git config
git config --global user.email "hjwoo@umich.edu"
git config --global user.name "Hyunjae Woo"

echo "Reboot your shell by typing 'source ~/.bashrc'"
