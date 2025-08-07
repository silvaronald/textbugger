# TextBugger Setup Instructions

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd textbugger
   ```

2. **Run setup script** (downloads external models)
   ```bash
   chmod +x scripts/setup_models.sh
   ./scripts/setup_models.sh
   ```

3. **Configure API credentials** (create `.env` file)
   ```bash
   cp .env.example .env  # Edit with your API keys
   ```

4. **Test the system**
   ```bash
   python scripts/run_attacks.py --target api --limit 1 --dataset rtmr
   ```

## What Gets Downloaded

### External Models (Not in Git)
- `amazon_review_polarity.bin` (~600MB) - FastText pre-trained model
- `en_core_web_sm` - spaCy English language model
- HuggingFace models (downloaded on first use)

### Research Artifacts (In Git) 
- `results/` - Your experiment results 
- `logs/` - Experiment logs
- `training/` - Your trained models
- Dataset preprocessors and tokenizers

## Manual Setup (Alternative)

If the setup script fails, install dependencies manually:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download FastText model
wget -O models/external/fasttext/amazon_review_polarity.bin \
  "https://dl.fbaipublicfiles.com/fasttext/supervised-models/amazon_review_polarity.bin"
```

## API Credentials

Create `.env` file with:
```bash
# IBM Watson
IBM_API_KEY=your_ibm_api_key
IBM_SERVICE_URL=your_ibm_service_url

# Azure Text Analytics  
AZURE_TEXT_ANALYTICS_KEY=your_azure_key
AZURE_TEXT_ANALYTICS_ENDPOINT=your_azure_endpoint

# Google Cloud NLP
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# AWS Comprehend
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1

# Attack Configuration
NUM_ATTACKS_API=10
```

## Project Structure

```
textbugger/
├── src/                    # Main source code
│   ├── attacks/           # Attack implementations
│   ├── clients/           # API clients  
│   ├── models/            # Model wrappers
│   └── utils/             # Utilities
├── scripts/               # Experiment runners
├── results/               # Experiment results (tracked)
├── logs/                  # Experiment logs (tracked)
├── training/              # Your trained models (tracked)
├── models/external/       # Downloaded models (ignored)
└── datasets/              # Datasets
```