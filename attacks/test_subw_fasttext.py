from blackbox import BlackBoxAttack
import numpy as np

def test_subw_against_fasttext():
    """
    Test Sub-W (word substitution) attack specifically against FastText
    """
    
    print("="*70)
    print("TESTING SUB-W (WORD SUBSTITUTION) ATTACK AGAINST FASTTEXT")
    print("="*70)
    
    try:
        # Initialize attacker
        attacker = BlackBoxAttack("rtmr")
        fasttext_classifier = attacker.create_fasttext_classifier_wrapper("amazon_review_polarity.bin")
        
        if not fasttext_classifier:
            print("❌ FastText classifier failed to load")
            return
            
        # Test sentences with strong sentiment words that should have good substitutes
        test_cases = [
            {
                "text": "This movie is amazing and fantastic",
                "target_words": ["amazing", "fantastic"],
                "expected_sentiment": "positive"
            },
            {
                "text": "This film is terrible and awful", 
                "target_words": ["terrible", "awful"],
                "expected_sentiment": "negative"
            },
            {
                "text": "I love this wonderful movie",
                "target_words": ["love", "wonderful"],
                "expected_sentiment": "positive" 
            },
            {
                "text": "I hate this boring film",
                "target_words": ["hate", "boring"],
                "expected_sentiment": "negative"
            },
            {
                "text": "Excellent acting and superb direction",
                "target_words": ["excellent", "superb"],
                "expected_sentiment": "positive"
            }
        ]
        
        print(f"\n🔍 WORD SIMILARITY ANALYSIS")
        print("-" * 50)
        
        # First, let's see what word substitutions we can generate
        for case in test_cases:
            print(f"\n📝 Analyzing: '{case['text']}'")
            for word in case["target_words"]:
                similar_words = attacker.get_similar_words(word, top_k=5)
                print(f"   '{word}' → {similar_words}")
        
        print(f"\n🎯 FASTTEXT ATTACK TESTING")
        print("-" * 50)
        
        for i, case in enumerate(test_cases):
            text = case["text"]
            print(f"\n--- Test Case {i+1} ---")
            print(f"Original: '{text}'")
            
            # Get original prediction
            original_label = fasttext_classifier(text, return_label=True)
            original_conf = fasttext_classifier(text, return_confidence=True)
            original_name = "POSITIVE" if original_label == 1 else "NEGATIVE"
            print(f"FastText: {original_name} (label {original_label}, conf: {original_conf:.3f})")
            
            # Test manual word substitutions to see if any can flip the prediction
            print(f"🔧 Testing manual word substitutions:")
            
            words = text.split()
            successful_substitutions = []
            
            for word_idx, word in enumerate(words):
                if word.lower().replace('.', '').replace(',', '') in case["target_words"]:
                    similar_words = attacker.get_similar_words(word.lower(), top_k=5)
                    
                    for substitute in similar_words:
                        # Create candidate with word substitution
                        candidate_words = words.copy()
                        candidate_words[word_idx] = substitute
                        candidate_text = " ".join(candidate_words)
                        
                        # Test prediction
                        new_label = fasttext_classifier(candidate_text, return_label=True)
                        new_conf = fasttext_classifier(candidate_text, return_confidence=True)
                        new_name = "POSITIVE" if new_label == 1 else "NEGATIVE"
                        
                        print(f"   '{word}' → '{substitute}': {new_name} (conf: {new_conf:.3f})")
                        
                        if new_label != original_label:
                            successful_substitutions.append({
                                "original_word": word,
                                "substitute": substitute,
                                "new_text": candidate_text,
                                "old_label": original_name,
                                "new_label": new_name,
                                "conf_change": f"{original_conf:.3f} → {new_conf:.3f}"
                            })
                            print(f"   ✅ SUCCESS! Label flipped: {original_name} → {new_name}")
            
            if successful_substitutions:
                print(f"\n🎉 SUCCESSFUL WORD SUBSTITUTIONS:")
                for sub in successful_substitutions:
                    print(f"   Original: '{text}'")
                    print(f"   Modified: '{sub['new_text']}'")
                    print(f"   Change: '{sub['original_word']}' → '{sub['substitute']}'")
                    print(f"   Result: {sub['old_label']} → {sub['new_label']} ({sub['conf_change']})")
            else:
                print(f"   ❌ No successful word substitutions found")
            
            # Now test the full black-box attack
            print(f"\n🚀 Testing full black-box attack with Sub-W:")
            adversarial = attacker.textbugger_blackbox_attack(
                document_x=text,
                ground_truth_label_y=original_label,
                classifier_F=fasttext_classifier,
                threshold_epsilon=0.7  # Allow some similarity loss
            )
            
            if adversarial and adversarial != text:
                adv_label = fasttext_classifier(adversarial, return_label=True) 
                adv_conf = fasttext_classifier(adversarial, return_confidence=True)
                adv_name = "POSITIVE" if adv_label == 1 else "NEGATIVE"
                
                if adv_label != original_label:
                    print(f"   ✅ FULL ATTACK SUCCESS!")
                    print(f"   Result: '{adversarial}'")
                    print(f"   Flipped: {original_name} → {adv_name} ({original_conf:.3f} → {adv_conf:.3f})")
                    
                    # Show what changed
                    original_words = text.split()
                    adversarial_words = adversarial.split()
                    changes = []
                    for orig, adv in zip(original_words, adversarial_words):
                        if orig != adv:
                            changes.append(f"'{orig}' → '{adv}'")
                    print(f"   Changes: {', '.join(changes)}")
                else:
                    print(f"   ⚠️  Text modified but no label flip: '{adversarial}'")
            else:
                print(f"   ❌ Full attack failed")
            
            print("-" * 50)
        
        print(f"\n📊 SUMMARY")
        print("Sub-W attacks test semantic word substitutions that might bypass")
        print("FastText's character-level robustness by changing meaning rather than spelling.")
        
    except Exception as e:
        print(f"❌ Error in Sub-W testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_subw_against_fasttext()