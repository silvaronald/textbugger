from blackbox import BlackBoxAttack
import numpy as np
import pandas as pd

def test_blackbox_attack():
    """
    Test the black-box TextBugger attack implementation
    """
    try:
        # Initialize the black-box attacker
        print("Initializing BlackBoxAttack...")
        attacker = BlackBoxAttack("rtmr")  # Using RT Movie Reviews dataset
        
        # Create a classifier wrapper for testing
        print("Creating classifier wrapper...")
        classifier = attacker.create_classifier_wrapper("lstm")  # Test with LSTM model
        
        # Test texts
        test_texts = [
            "This movie is absolutely terrible and boring",
            "I love this product, it's amazing and wonderful",
            "The film was okay, nothing special but not bad either"
        ]
        
        print("\n" + "="*60)
        print("TESTING BLACK-BOX TEXTBUGGER ATTACK")
        print("="*60)
        
        for i, text in enumerate(test_texts):
            print(f"\n--- Test Case {i+1} ---")
            print(f"Original text: {text}")
            
            # Get original prediction
            try:
                original_label = classifier(text, return_label=True)
                original_confidence = classifier(text, return_confidence=True)
                print(f"Original prediction: Label {original_label} (confidence: {original_confidence:.3f})")
                
                # Launch black-box attack
                print("Launching black-box attack...")
                adversarial_text = attacker.textbugger_blackbox_attack(
                    document_x=text,
                    ground_truth_label_y=original_label,
                    classifier_F=classifier,
                    threshold_epsilon=0.7  # Allow 30% similarity loss
                )
                
                if adversarial_text:
                    # Test adversarial example
                    adv_label = classifier(adversarial_text, return_label=True)
                    adv_confidence = classifier(adversarial_text, return_confidence=True)
                    
                    print(f"✅ ATTACK SUCCESSFUL!")
                    print(f"Adversarial text: {adversarial_text}")
                    print(f"New prediction: Label {adv_label} (confidence: {adv_confidence:.3f})")
                    
                    # Calculate similarity
                    similarity = attacker.calculate_semantic_similarity(text, adversarial_text)
                    print(f"Semantic similarity: {similarity:.3f}")
                    
                else:
                    print("❌ Attack failed - no adversarial example found")
                    
            except Exception as e:
                print(f"❌ Error during attack: {e}")
            
            print("-" * 50)
        
        # Test bug generation
        print("\n--- Testing Bug Generation ---")
        test_words = ["amazing", "terrible", "movie", "excellent"]
        for word in test_words:
            bugs = attacker.generate_bugs(word)
            print(f"'{word}' → {bugs}")
            
        print("\n--- Testing Sentence Segmentation ---")
        long_text = "This is the first sentence. And this is the second sentence! Finally, here's the third sentence?"
        sentences = attacker.segment_into_sentences(long_text)
        print(f"Original: {long_text}")
        print(f"Sentences: {sentences}")
        
        print("\n✅ Black-box attack testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error initializing BlackBoxAttack: {e}")
        print("Make sure you have the required datasets and models trained.")

if __name__ == "__main__":
    test_blackbox_attack()