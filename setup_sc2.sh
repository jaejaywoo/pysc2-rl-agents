#!/bin/bash

# Install StarCraft II API binary
cd $HOME
sudo apt-get install unzip
mkdir -p .local/{lib,bin} && cd .local/lib
wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.1.2.60604_2018_05_16.zip
unzip SC2.4.1.2.60604_2018_05_16.zip && rm -rf SC2.4.1.2.60604_2018_05_16.zip

cd StarCraftII/Maps
wget http://blzdistsc2-a.akamaihd.net/MapPacks/Melee.zip
unzip Melee.zip && rm -rf Melee.zip

wget https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip
unzip mini_games.zip && rm -rf mini_games.zip

cd $HOME
echo "# StarCraftII path" >> .bashrc
echo "export PATH=$HOME/.local/lib:$HOME/.local/bin:$PATH" >> .bashrc
echo "export SC2PATH=$HOME/.local/lib/StarCraftII" >> .bashrc
echo "" >> .bashrc

echo "Reboot the shell by typing 'source ~/.bashrc'"
