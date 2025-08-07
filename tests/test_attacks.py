#!/usr/bin/env python3
"""
Test TextBugger attack implementations
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from attacks.blackbox import BlackBoxTextBugger
from models.api_models import APIModelWrapper
from clients.ibm_watson import IBMWatsonClassifier

def test_api_attack():
    """Test TextBugger attack on API client"""
    print("🔬 Testing API Attack")
    
    # Simple test text
    text = "This movie is great."
    
    # Setup IBM Watson
    client = IBMWatsonClassifier()
    wrapper = APIModelWrapper(client)
    
    print(f"Original text: {text}")
    
    # Test original classification
    original_label, original_scores = client.classify(text)
    print(f"Original classification: {original_label} (scores: {original_scores})")
    
    # Test wrapper
    wrapped_label, wrapped_score = wrapper(text)
    print(f"Wrapped classification: {wrapped_label} (score: {wrapped_score})")
    
    # Test attack
    print(f"\n🎯 Testing attack...")
    attacker = BlackBoxTextBugger(wrapper, similarity_threshold=0.7)
    
    try:
        result = attacker.attack(text)
        adv_text, og_label, num_reqs, success = result
        
        if success:
            new_label, new_scores = client.classify(adv_text)
            print(f"✅ Attack SUCCESS!")
            print(f"Adversarial text: {adv_text}")
            print(f"New classification: {new_label}")
            print(f"API requests used: {num_reqs}")
        else:
            print(f"❌ Attack failed after {num_reqs} requests")
            print(f"Final text: {adv_text}")
            
    except Exception as e:
        print(f"💥 Error: {e}")

def main():
    print("Testing TextBugger attacks...")
    test_api_attack()

if __name__ == "__main__":
    main()