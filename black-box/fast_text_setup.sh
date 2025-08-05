#!/bin/bash

# Update system
sudo apt-get update

# Install dependencies
sudo apt-get install -y git build-essential python3-pip python3-dev

# Clone and build fastText
git clone https://github.com/facebookresearch/fastText.git
cd fastText
sudo pip3 install .

# Verify installation
python3 -c "import fasttext; print('FastText installed successfully!')"