#!/usr/bin/env python3
"""
Test script for single word attack on specific RTMR instance
"""

import sys
import os
from pathlib import Path
import time
import pandas as pd

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from attacks.batch_blackbox import BatchBlackBoxTextBugger, BatchLevel
from models.api_models import APIModelWrapper
from clients.ibm_watson import IBMWatsonClassifier


def test_single_word_attack_instance_95():
    """Test single word attack on RTMR instance 95"""
    
    # Initialize API client
    try:
        client = IBMWatsonClassifier()
        wrapper = APIModelWrapper(client)
    except Exception as e:
        print(f"Error initializing IBM Watson: {e}")
        return
    
    # Load RTMR dataset
    try:
        df = pd.read_csv("datasets/rtmr/X_test.csv")
        if len(df) <= 95:
            print(f"Error: Dataset only has {len(df)} instances, cannot access instance 95")
            return
        
        text = df.iloc[95]["text"]  # Get instance 95 (0-indexed)
        print(f"✅ Loaded instance 95 from RTMR dataset")
        print(f"Text: {text}")
        
    except Exception as e:
        print(f"⚠️  Could not load RTMR dataset: {e}")
        return
    
    print("="*80)
    print("SINGLE WORD ATTACK - INSTANCE 95")
    print("="*80)
    
    # Test single word attack (equivalent to batch_size=1)
    print("🔸 Running single word attack...")
    
    attacker = BatchBlackBoxTextBugger(
        wrapper,
        similarity_threshold=0.8,
        batch_level=BatchLevel.SINGLE_WORD,  # Use single word approach
        batch_size=1
    )
    
    start_time = time.time()
    adv_text, label, queries, perturbs, success = attacker.attack(text)
    duration = time.time() - start_time
    
    print(f"\n📊 RESULTS:")
    print(f"Original text: {text}")
    print(f"Adversarial text: {adv_text}")
    print(f"Original label: {label}")
    print(f"Attack success: {success}")
    print(f"Total queries: {queries}")
    print(f"Total perturbations: {perturbs}")
    print(f"Time taken: {duration:.2f}s")
    
    # Show the changes made
    if text != adv_text:
        print(f"\n🔍 Changes made:")
        original_words = text.split()
        adversarial_words = adv_text.split()
        
        # Simple word-by-word comparison
        changes = []
        min_len = min(len(original_words), len(adversarial_words))
        
        for i in range(min_len):
            if original_words[i] != adversarial_words[i]:
                changes.append(f"'{original_words[i]}' → '{adversarial_words[i]}'")
        
        if len(adversarial_words) > len(original_words):
            changes.extend([f"Added: '{word}'" for word in adversarial_words[len(original_words):]])
        elif len(original_words) > len(adversarial_words):
            changes.extend([f"Removed: '{word}'" for word in original_words[len(adversarial_words):]])
        
        for change in changes:
            print(f"  - {change}")


if __name__ == "__main__":
    print("Testing single word attack on RTMR instance 95...\n")
    test_single_word_attack_instance_95()
    print("\n✅ Test completed!")