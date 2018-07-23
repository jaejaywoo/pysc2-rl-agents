#!/bin/bash

# Clean up & preparation
rm -rf {.vim}
sudo pip3 install neovim

# Install custom dotfiles
cd $HOME
git clone --recursive https://github.com/wookayin/dotfiles.git ~/.dotfiles
cd ~/.dotfiles && python install.py
dotfiles install neovim && python install.py
