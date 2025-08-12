#!/usr/bin/env python3
"""
Test confidence-driven batch sizing
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from attacks.batch_blackbox import BatchBlackBoxTextBugger, BatchLevel
from models.api_models import APIModelWrapper
from clients.ibm_watson import IBMWatsonClassifier


def test_confidence_driven():
    """Test confidence-driven batch sizing with different confidence scenarios"""
    
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
        rtmr_texts = df["text"].head(5).tolist()
        
        # Create test cases from real RTMR data
        test_cases = []
        for i, text in enumerate(rtmr_texts):
            test_cases.append({
                "text": text,
                "description": f"RTMR example {i+1}"
            })
        print(f"✅ Loaded {len(test_cases)} real examples from RTMR dataset")
        
    except Exception as e:
        print(f"⚠️  Could not load RTMR dataset: {e}")
        # Fallback examples
        test_cases = [
            {
                "text": "occasionally funny , always very colorful and enjoyably overblown in the traditional almodóvar style .",
                "description": "Positive movie review"
            },
            {
                "text": "becomes the last thing you would expect from a film with this title or indeed from any plympton film : boring .",
                "description": "Negative movie review"
            }
        ]
        print("Using fallback examples")
    
    print("="*80)
    print("CONFIDENCE-DRIVEN BATCH SIZING TEST")
    print("="*80)
    
    for i, test_case in enumerate(test_cases, 1):
        text = test_case["text"]
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"Text: {text}")
        print("-" * 60)
        
        # First, check the actual confidence
        label, confidence = wrapper(text)
        print(f"🎯 Actual prediction: {label} (confidence: {confidence:.3f})")
        
        # Test confidence-driven batch
        attacker = BatchBlackBoxTextBugger(
            wrapper,
            similarity_threshold=0.8,
            batch_level=BatchLevel.CONFIDENCE_DRIVEN,
            batch_size=3  # This will be overridden by confidence logic
        )
        
        print(f"🚀 Running confidence-driven attack...")
        adv_text, orig_label, queries, perturbs, success = attacker.attack(text)
        
        print(f"📊 Results:")
        print(f"  - Queries used: {queries}")
        print(f"  - Perturbations applied: {perturbs}")
        print(f"  - Attack successful: {success}")
        print(f"  - Adversarial text: {adv_text}")
        
        if success:
            # Check new confidence
            new_label, new_confidence = wrapper(adv_text)
            print(f"  - New prediction: {new_label} (confidence: {new_confidence:.3f})")
        
        print("="*80)


if __name__ == "__main__":
    test_confidence_driven()