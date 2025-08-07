#!/usr/bin/env python3
"""
Consolidated tests for API clients
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from clients.ibm_watson import IBMWatsonClassifier
from clients.azure_text import AzureTextAnalyticsClassifier  
from clients.google_nlp import GoogleNLPClassifier
from clients.aws_comprehend import AWSComprehendClassifier

# Test texts
test_texts = [
    "This film was a masterpiece. The acting was incredibly compelling.",
    "I was so disappointed by this movie. The plot was predictable.",
    "You are an absolute idiot for even suggesting that.",
    "The report indicates that the project is currently on schedule."
]

def test_ibm_watson():
    print("=" * 50)
    print("TESTING IBM WATSON")
    print("=" * 50)
    
    try:
        client = IBMWatsonClassifier()
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text[:50]}...")
            label, scores = client.classify(text)
            print(f"Label: {label}")
            print(f"Scores: {scores}")
    except Exception as e:
        print(f"Error with IBM Watson: {e}")

def test_azure():
    print("=" * 50)
    print("TESTING AZURE TEXT ANALYTICS")
    print("=" * 50)
    
    try:
        client = AzureTextAnalyticsClassifier()
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text[:50]}...")
            label, scores = client.classify(text)
            print(f"Label: {label}")
            print(f"Scores: {scores}")
    except Exception as e:
        print(f"Error with Azure: {e}")

def test_google_nlp():
    print("=" * 50)
    print("TESTING GOOGLE CLOUD NLP")
    print("=" * 50)
    
    try:
        client = GoogleNLPClassifier()
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text[:50]}...")
            label, scores = client.classify(text)
            print(f"Label: {label}")
            print(f"Scores: {scores}")
    except Exception as e:
        print(f"Error with Google NLP: {e}")

def test_aws_comprehend():
    print("=" * 50)
    print("TESTING AWS COMPREHEND")
    print("=" * 50)
    
    try:
        client = AWSComprehendClassifier()
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text[:50]}...")
            label, scores = client.classify(text)
            print(f"Label: {label}")
            print(f"Scores: {scores}")
    except Exception as e:
        print(f"Error with AWS Comprehend: {e}")

def main():
    print("Testing all API clients...")
    
    # Test working clients
    test_ibm_watson()
    test_azure()
    
    # Test Google Cloud NLP if credentials available
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        test_google_nlp()
    else:
        print("=" * 50)
        print("GOOGLE CLOUD NLP - SKIPPED")
        print("=" * 50)
        print("Set GOOGLE_APPLICATION_CREDENTIALS to test Google Cloud NLP")
    
    # Test AWS if credentials available
    if os.getenv('AWS_ACCESS_KEY_ID'):
        test_aws_comprehend()
    else:
        print("=" * 50)
        print("AWS COMPREHEND - SKIPPED")
        print("=" * 50)
        print("Set AWS credentials to test AWS Comprehend")

if __name__ == "__main__":
    main()