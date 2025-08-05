from blackbox import BlackBoxAttack
import numpy as np
import pandas as pd

def test_fasttext_blackbox_attack():
    """
    Test the black-box TextBugger attack against FastText model
    """
    try:
        # Initialize the black-box attacker
        print("Initializing BlackBoxAttack...")
        attacker = BlackBoxAttack("rtmr")  # Using RT Movie Reviews dataset
        
        # Create a FastText classifier wrapper
        print("Creating FastText classifier wrapper...")
        fasttext_classifier = attacker.create_fasttext_classifier_wrapper("amazon_review_polarity.bin")
        
        if fasttext_classifier is None:
            print("❌ Failed to load FastText model")
            return
        
        # Test texts with different sentiment strengths
        test_texts = [
            # Strong negative (should be easy to attack)
            "This movie is absolutely terrible and boring and awful",
            # Strong positive (should be easy to attack) 
            "I love this product, it's amazing and wonderful and fantastic",
            # Mild sentiment (might be harder to attack)
            "The film was okay, nothing special but not bad either",
            # Simple negative
            "Bad movie",
            # Simple positive
            "Good film"
        ]
        
        print("\n" + "="*70)
        print("TESTING BLACK-BOX TEXTBUGGER ATTACK AGAINST FASTTEXT")
        print("="*70)
        
        successful_attacks = 0
        total_tests = len(test_texts)
        
        for i, text in enumerate(test_texts):
            print(f"\n--- Test Case {i+1}/{total_tests} ---")
            print(f"Original text: '{text}'")
            
            # Get original prediction
            try:
                original_label = fasttext_classifier(text, return_label=True)
                original_confidence = fasttext_classifier(text, return_confidence=True)
                label_name = "POSITIVE" if original_label == 1 else "NEGATIVE"
                print(f"Original prediction: {label_name} (Label {original_label}, confidence: {original_confidence:.3f})")
                
                # Launch black-box attack with more relaxed similarity threshold
                print("🚀 Launching black-box attack...")
                adversarial_text = attacker.textbugger_blackbox_attack(
                    document_x=text,
                    ground_truth_label_y=original_label,
                    classifier_F=fasttext_classifier,
                    threshold_epsilon=0.6  # Allow 40% similarity loss for better attack success
                )
                
                if adversarial_text and adversarial_text != text:
                    # Test adversarial example
                    adv_label = fasttext_classifier(adversarial_text, return_label=True)
                    adv_confidence = fasttext_classifier(adversarial_text, return_confidence=True)
                    adv_label_name = "POSITIVE" if adv_label == 1 else "NEGATIVE"
                    
                    if adv_label != original_label:
                        print(f"✅ ATTACK SUCCESSFUL!")
                        print(f"Adversarial text: '{adversarial_text}'")
                        print(f"New prediction: {adv_label_name} (Label {adv_label}, confidence: {adv_confidence:.3f})")
                        
                        # Calculate similarity
                        similarity = attacker.calculate_semantic_similarity(text, adversarial_text)
                        print(f"Semantic similarity: {similarity:.3f}")
                        
                        # Show the differences
                        original_words = text.split()
                        adversarial_words = adversarial_text.split()
                        print(f"Original words: {original_words}")
                        print(f"Modified words: {adversarial_words}")
                        
                        successful_attacks += 1
                    else:
                        print(f"⚠️  Text modified but label didn't flip:")
                        print(f"Adversarial text: '{adversarial_text}'")
                        print(f"Same prediction: {adv_label_name} (confidence: {adv_confidence:.3f})")
                else:
                    print("❌ Attack failed - no adversarial example found")
                    
            except Exception as e:
                print(f"❌ Error during attack: {e}")
                import traceback
                traceback.print_exc()
            
            print("-" * 50)
        
        # Summary
        print(f"\n📊 ATTACK SUMMARY:")
        print(f"Successful attacks: {successful_attacks}/{total_tests}")
        print(f"Success rate: {(successful_attacks/total_tests)*100:.1f}%")
        
        # Test individual components
        print(f"\n🔧 COMPONENT TESTS:")
        
        # Test FastText predictions
        print("\n--- FastText Predictions ---")
        simple_tests = [
            "This movie is terrible",
            "This movie is amazing", 
            "okay film"
        ]
        
        for text in simple_tests:
            label = fasttext_classifier(text, return_label=True)
            confidence = fasttext_classifier(text, return_confidence=True)
            probs = fasttext_classifier(text)
            label_name = "POSITIVE" if label == 1 else "NEGATIVE"
            print(f"'{text}' → {label_name} (conf: {confidence:.3f}, probs: {probs})")
        
        # Test bug generation
        print(f"\n--- Bug Generation ---")
        test_words = ["terrible", "amazing", "good", "bad"]
        for word in test_words:
            bugs = attacker.generate_bugs(word)
            print(f"'{word}' → {bugs}")
            
        print(f"\n✅ FastText black-box attack testing completed!")
        
    except Exception as e:
        print(f"❌ Error initializing test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fasttext_blackbox_attack()