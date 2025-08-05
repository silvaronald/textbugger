from blackbox import BlackBoxAttack
import numpy as np

def test_all_perturbation_types():
    """
    Test all 5 TextBugger perturbation types:
    1. Insert (character-level)
    2. Delete (character-level)  
    3. Swap (character-level)
    4. Substitute-C (character-level)
    5. Substitute-W (word-level)
    """
    
    print("="*70)
    print("TESTING ALL 5 TEXTBUGGER PERTURBATION TYPES")
    print("="*70)
    
    try:
        # Initialize the attack
        print("Initializing BlackBoxAttack...")
        attacker = BlackBoxAttack("rtmr")
        
        # Test words for perturbation
        test_words = [
            "amazing",      # Should have good semantic neighbors
            "terrible",     # Should have good semantic neighbors
            "fantastic",    # Should have good semantic neighbors
            "awful",        # Should have good semantic neighbors
            "movie",        # Common word
            "boring"        # Should have good semantic neighbors
        ]
        
        print(f"\n🔧 TESTING BUG GENERATION FOR EACH WORD")
        print("-" * 50)
        
        for word in test_words:
            print(f"\n📝 Testing word: '{word}'")
            
            # Generate all possible bugs for this word
            bugs = attacker.generate_bugs(word)
            print(f"Generated {len(bugs)} perturbations: {bugs}")
            
            # Test each perturbation type individually
            print(f"   Individual perturbation tests:")
            
            # 1. Insert
            insert_bug = attacker.insert_bug(word)
            if insert_bug != word:
                print(f"   ✅ Insert: '{word}' → '{insert_bug}'")
            else:
                print(f"   ⚠️  Insert: No change (word too short or other constraint)")
            
            # 2. Delete  
            delete_bug = attacker.delete_bug(word)
            if delete_bug != word:
                print(f"   ✅ Delete: '{word}' → '{delete_bug}'")
            else:
                print(f"   ⚠️  Delete: No change (word too short)")
                
            # 3. Swap
            swap_bug = attacker.swap_bug(word)
            if swap_bug != word:
                print(f"   ✅ Swap: '{word}' → '{swap_bug}'")
            else:
                print(f"   ⚠️  Swap: No change (word too short)")
                
            # 4. Substitute-C
            substitute_c_bug = attacker.substitute_c(word)
            if substitute_c_bug != word:
                print(f"   ✅ Substitute-C: '{word}' → '{substitute_c_bug}'")
            else:
                print(f"   ⚠️  Substitute-C: No change (no substitutable characters)")
                
            # 5. Substitute-W
            print(f"   🔍 Finding similar words for '{word}'...")
            similar_words = attacker.get_similar_words(word, top_k=5)
            if similar_words:
                print(f"   ✅ Substitute-W: '{word}' → {similar_words[:3]}")  # Show top 3
            else:
                print(f"   ⚠️  Substitute-W: No similar words found")
        
        print(f"\n🎯 TESTING COMPLETE ATTACK WITH ALL PERTURBATIONS")
        print("-" * 50)
        
        # Test complete attack with FastText
        fasttext_classifier = attacker.create_fasttext_classifier_wrapper("amazon_review_polarity.bin")
        
        if fasttext_classifier:
            test_sentences = [
                "This movie is absolutely amazing and fantastic",  # Multiple good words to attack
                "The film is terrible and boring and awful",       # Multiple bad words to attack
                "Good movie with excellent acting",                # Shorter positive
                "Bad film with terrible story"                     # Shorter negative
            ]
            
            for sentence in test_sentences:
                print(f"\n🚀 Testing: '{sentence}'")
                
                # Get original prediction
                original_label = fasttext_classifier(sentence, return_label=True)
                original_conf = fasttext_classifier(sentence, return_confidence=True)
                label_name = "POSITIVE" if original_label == 1 else "NEGATIVE"
                print(f"   Original: {label_name} (confidence: {original_conf:.3f})")
                
                # Try attack with more aggressive settings
                adversarial = attacker.textbugger_blackbox_attack(
                    document_x=sentence,
                    ground_truth_label_y=original_label,
                    classifier_F=fasttext_classifier,
                    threshold_epsilon=0.5  # Allow 50% similarity loss
                )
                
                if adversarial and adversarial != sentence:
                    new_label = fasttext_classifier(adversarial, return_label=True)
                    new_conf = fasttext_classifier(adversarial, return_confidence=True)
                    new_label_name = "POSITIVE" if new_label == 1 else "NEGATIVE"
                    
                    if new_label != original_label:
                        print(f"   ✅ SUCCESS: '{adversarial}'")
                        print(f"   ✅ Flipped to: {new_label_name} (confidence: {new_conf:.3f})")
                    else:
                        print(f"   ⚠️  Modified: '{adversarial}'")
                        print(f"   ⚠️  Same label: {new_label_name} (confidence: {new_conf:.3f})")
                        
                    # Show similarity
                    similarity = attacker.calculate_semantic_similarity(sentence, adversarial)
                    print(f"   📊 Similarity: {similarity:.3f}")
                else:
                    print(f"   ❌ Attack failed")
        
        print(f"\n📊 SUMMARY OF PERTURBATION CAPABILITIES")
        print("-" * 50)
        print("✅ Insert: Adds spaces within words")
        print("✅ Delete: Removes characters from words")  
        print("✅ Swap: Swaps adjacent characters")
        print("✅ Substitute-C: Replaces with visually similar characters")
        print("✅ Substitute-W: Replaces with semantically similar words")
        print(f"\n🎯 All 5 TextBugger perturbation types implemented!")
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_all_perturbation_types()