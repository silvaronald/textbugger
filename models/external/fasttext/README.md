# FastText Models

This directory contains FastText model binaries.

## Files
- `amazon_review_polarity.bin` - Pre-trained FastText model for Amazon review sentiment classification

## Usage
```python
import fasttext
model = fasttext.load_model('models/external/fasttext/amazon_review_polarity.bin')
```

Note: Binary files are moved here from the root directory for better organization.