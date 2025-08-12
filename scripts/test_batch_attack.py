#!/usr/bin/env python3
"""
Test script for batch TextBugger implementation
Compares query efficiency between original and batch approaches
"""

import sys
import os
from pathlib import Path
import time

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from attacks.blackbox import BlackBoxTextBugger
from attacks.batch_blackbox import BatchBlackBoxTextBugger, BatchLevel
from models.api_models import APIModelWrapper
from clients.ibm_watson import IBMWatsonClassifier


def test_batch_implementations():
    """Test all three batch level options"""
    
    # Initialize API client
    try:
        client = IBMWatsonClassifier()
        wrapper = APIModelWrapper(client)
    except Exception as e:
        print(f"Error initializing IBM Watson: {e}")
        return
    
    # Load real examples from RTMR dataset
    import pandas as pd
    
    try:
        df = pd.read_csv("datasets/rtmr/X_test.csv")
        test_texts = df["text"].head(5).tolist()  # 5 examples for better pattern analysis
        print(f"✅ Loaded {len(test_texts)} examples from RTMR dataset")
    except Exception as e:
        print(f"⚠️  Could not load RTMR dataset: {e}")
        # Fallback to manual examples
        test_texts = [
            "occasionally funny , always very colorful and enjoyably overblown in the traditional almodóvar style .",
            "becomes the last thing you would expect from a film with this title or indeed from any plympton film : boring ."
        ]
        print("Using fallback examples")
    
    print("="*80)
    print("BATCH TEXTBUGGER COMPARISON")
    print("="*80)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}: {text}")
        print("-" * 60)
        
        results = {}
        
        # Original TextBugger
        print("🔸 Original TextBugger...")
        start_time = time.time()
        original_attacker = BlackBoxTextBugger(wrapper, similarity_threshold=0.8)
        adv_text, label, queries, perturbs, success = original_attacker.attack(text)
        duration = time.time() - start_time
        
        results['original'] = {
            'queries': queries,
            'perturbations': perturbs,
            'success': success,
            'time': duration,
            'adversarial_text': adv_text
        }
        
        # Batch implementations
        batch_configs = [
            ("Multi-Word Batch (2)", BatchLevel.MULTI_WORD, 2),
            ("Multi-Word Batch (3)", BatchLevel.MULTI_WORD, 3),
            ("Confidence-Driven Batch", BatchLevel.CONFIDENCE_DRIVEN, 3),
            ("Adaptive Batch", BatchLevel.ADAPTIVE, 4),
        ]
        
        for config_name, batch_level, batch_size in batch_configs:
            print(f"🔸 {config_name}...")
            start_time = time.time()
            
            batch_attacker = BatchBlackBoxTextBugger(
                wrapper, 
                similarity_threshold=0.8,
                batch_level=batch_level,
                batch_size=batch_size
            )
            
            adv_text, label, queries, perturbs, success = batch_attacker.attack(text)
            duration = time.time() - start_time
            
            results[config_name.lower().replace(' ', '_')] = {
                'queries': queries,
                'perturbations': perturbs,
                'success': success,
                'time': duration,
                'adversarial_text': adv_text
            }
        
        # Print comparison
        print("\n📊 RESULTS COMPARISON:")
        print(f"{'Method':<20} {'Queries':<8} {'Perturbs':<8} {'Success':<8} {'Time(s)':<8}")
        print("-" * 60)
        
        for method, stats in results.items():
            print(f"{method.replace('_', ' ').title():<20} "
                  f"{stats['queries']:<8} "
                  f"{stats['perturbations']:<8} "
                  f"{str(stats['success']):<8} "
                  f"{stats['time']:.2f}s")
        
        # Calculate efficiency gains
        original_queries = results['original']['queries']
        print(f"\n🚀 EFFICIENCY GAINS (vs Original):")
        for method, stats in results.items():
            if method != 'original':
                reduction = ((original_queries - stats['queries']) / original_queries) * 100
                print(f"  {method.replace('_', ' ').title()}: {reduction:.1f}% query reduction")
        
        print("\n" + "="*80)


def test_specific_batch_level():
    """Test specific batch level in detail"""
    
    try:
        client = IBMWatsonClassifier()
        wrapper = APIModelWrapper(client)
    except Exception as e:
        print(f"Error initializing IBM Watson: {e}")
        return
    
    text = "This movie was absolutely terrible and boring, I hated every minute of it."
    
    print("="*80)
    print("DETAILED BATCH ANALYSIS - GREEDY MULTI-WORD")
    print("="*80)
    
    # Test different batch sizes
    batch_sizes = [1, 2, 3, 5, 7]
    
    for batch_size in batch_sizes:
        print(f"\n🔸 Testing batch size: {batch_size}")
        
        attacker = BatchBlackBoxTextBugger(
            wrapper,
            similarity_threshold=0.8,
            batch_level=BatchLevel.MULTI_WORD,
            batch_size=batch_size
        )
        
        start_time = time.time()
        adv_text, label, queries, perturbs, success = attacker.attack(text)
        duration = time.time() - start_time
        
        print(f"  Queries: {queries}, Perturbations: {perturbs}, Success: {success}, Time: {duration:.2f}s")
        if success:
            print(f"  Result: {adv_text}")


if __name__ == "__main__":
    print("Starting batch TextBugger tests...\n")
    
    # Check which test to run
    if len(sys.argv) > 1 and sys.argv[1] == "--detailed":
        test_specific_batch_level()
    else:
        test_batch_implementations()
    
    print("\n✅ Batch testing completed!")