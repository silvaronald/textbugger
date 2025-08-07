#!/bin/bash
# Setup script for remote HPC environments
# Run this once to set up the environment on a new cluster

echo "Setting up TextBugger environment on remote machine..."

# Create virtual environment
python -m venv $HOME/textbugger_env

# Activate environment
source $HOME/textbugger_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Download spacy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('wordnet')"

# Create necessary directories
mkdir -p logs results

echo "Setup completed!"
echo "Remember to:"
echo "1. Copy your .env file with API credentials"
echo "2. Upload any model files to models/external/"
echo "3. Test with: python tests/test_api_clients.py"