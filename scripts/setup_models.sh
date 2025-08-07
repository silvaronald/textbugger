#!/bin/bash
# Setup script for downloading external models and dependencies
# Run this after cloning the repository to get all required external models

set -e  # Exit on any error

echo "🚀 Setting up TextBugger external models and dependencies"
echo "========================================================="

# Create directories
echo "📁 Creating model directories..."
mkdir -p models/external/fasttext/
mkdir -p models/external/huggingface/

# Download FastText Amazon Review Polarity model
FASTTEXT_MODEL="models/external/fasttext/amazon_review_polarity.bin"
if [ ! -f "$FASTTEXT_MODEL" ]; then
    echo "📥 Downloading FastText Amazon Review Polarity model (~600MB)..."
    wget -O "$FASTTEXT_MODEL" "https://dl.fbaipublicfiles.com/fasttext/supervised-models/amazon_review_polarity.bin"
    echo "✅ FastText model downloaded"
else
    echo "✅ FastText model already exists"
fi

# Download spaCy English model
echo "📥 Downloading spaCy English model..."
python -m spacy download en_core_web_sm || echo "⚠️  spaCy model download failed - install spacy first"

# Install required Python packages
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
else
    echo "⚠️  No requirements.txt found"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo "📊 Model files location:"
echo "  - FastText: $FASTTEXT_MODEL"
echo "  - spaCy: en_core_web_sm (system-wide)"
echo ""
echo "🚀 Ready to run TextBugger experiments!"
echo "Test with: python scripts/run_attacks.py --target api --limit 1 --dataset rtmr"