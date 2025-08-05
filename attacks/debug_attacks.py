from whitebox import AdversarialAttack
from blackbox import BlackBoxAttack
import numpy as np

def compare_attacks():
    """
    Compare white-box vs black-box attack performance to debug issues
    """
    
    print("="*70)
    print("DEBUGGING: WHITE-BOX vs BLACK-BOX ATTACK COMPARISON")
    print("="*70)
    
    # Test sentences - start with simple, clear sentiment
    test_sentences = [
        "This movie is amazing",           # Clear positive
        "This movie is terrible",         # Clear negative  
        "I love this film",               # Simple positive
        "I hate this film",               # Simple negative
        "Great movie",                    # Very simple positive
        "Bad movie"                       # Very simple negative
    ]
    
    try:
        print("\n🔧 INITIALIZING ATTACKS...")
        whitebox_attacker = AdversarialAttack("rtmr")
        blackbox_attacker = BlackBoxAttack("rtmr") 
        
        # Create black-box classifier wrapper 
        fasttext_classifier = blackbox_attacker.create_fasttext_classifier_wrapper("amazon_review_polarity.bin")
        lstm_classifier = blackbox_attacker.create_classifier_wrapper("lstm")
        
        if not fasttext_classifier:
            print("❌ FastText classifier failed to load")
            return
            
        print("✅ Both attackers initialized successfully")
        
        for i, sentence in enumerate(test_sentences):
            print(f"\n{'='*50}")
            print(f"TEST {i+1}: '{sentence}'")
            print(f"{'='*50}")
            
            # 1. Test FastText predictions (our black-box target)
            try:
                fasttext_label = fasttext_classifier(sentence, return_label=True)
                fasttext_conf = fasttext_classifier(sentence, return_confidence=True)
                fasttext_name = "POSITIVE" if fasttext_label == 1 else "NEGATIVE"
                print(f"📊 FastText: {fasttext_name} (Label {fasttext_label}, conf: {fasttext_conf:.3f})")
            except Exception as e:
                print(f"❌ FastText prediction error: {e}")
                continue
                
            # 2. Test LSTM predictions (our white-box target)
            try:
                lstm_label = lstm_classifier(sentence, return_label=True)
                lstm_conf = lstm_classifier(sentence, return_confidence=True)
                lstm_name = "POSITIVE" if lstm_label == 1 else "NEGATIVE"
                print(f"📊 LSTM: {lstm_name} (Label {lstm_label}, conf: {lstm_conf:.3f})")
            except Exception as e:
                print(f"❌ LSTM prediction error: {e}")
                continue
            
            print(f"\n🎯 TESTING WHITE-BOX ATTACK (against LSTM)...")
            try:
                # Test white-box attack
                whitebox_result = whitebox_attacker.generate_adversarial(
                    text=sentence,
                    model_type="lstm",
                    embedding_dict=None,  # Test without word substitution first
                    epsilon=0.7
                )
                
                if whitebox_result and whitebox_result != sentence:
                    # Test if attack succeeded
                    new_lstm_label = lstm_classifier(whitebox_result, return_label=True)
                    new_lstm_conf = lstm_classifier(whitebox_result, return_confidence=True)
                    new_lstm_name = "POSITIVE" if new_lstm_label == 1 else "NEGATIVE"
                    
                    if new_lstm_label != lstm_label:
                        print(f"✅ WHITE-BOX SUCCESS!")
                        print(f"   Original: '{sentence}' → {lstm_name}")
                        print(f"   Adversarial: '{whitebox_result}' → {new_lstm_name}")
                        print(f"   Confidence: {lstm_conf:.3f} → {new_lstm_conf:.3f}")
                    else:
                        print(f"⚠️  WHITE-BOX: Text changed but label same")
                        print(f"   Adversarial: '{whitebox_result}' → {new_lstm_name}")
                else:
                    print(f"❌ WHITE-BOX: No adversarial example generated")
                    
            except Exception as e:
                print(f"❌ White-box attack error: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n🎯 TESTING BLACK-BOX ATTACK (against FastText)...")
            try:
                # Test black-box attack with more permissive settings
                blackbox_result = blackbox_attacker.textbugger_blackbox_attack(
                    document_x=sentence,
                    ground_truth_label_y=fasttext_label,
                    classifier_F=fasttext_classifier,
                    threshold_epsilon=0.5  # Very permissive similarity
                )
                
                if blackbox_result and blackbox_result != sentence:
                    # Test if attack succeeded
                    new_fasttext_label = fasttext_classifier(blackbox_result, return_label=True)
                    new_fasttext_conf = fasttext_classifier(blackbox_result, return_confidence=True)
                    new_fasttext_name = "POSITIVE" if new_fasttext_label == 1 else "NEGATIVE"
                    
                    if new_fasttext_label != fasttext_label:
                        print(f"✅ BLACK-BOX SUCCESS!")
                        print(f"   Original: '{sentence}' → {fasttext_name}")
                        print(f"   Adversarial: '{blackbox_result}' → {new_fasttext_name}")
                        print(f"   Confidence: {fasttext_conf:.3f} → {new_fasttext_conf:.3f}")
                    else:
                        print(f"⚠️  BLACK-BOX: Text changed but label same")
                        print(f"   Adversarial: '{blackbox_result}' → {new_fasttext_name}")
                else:
                    print(f"❌ BLACK-BOX: No adversarial example generated")
                    
            except Exception as e:
                print(f"❌ Black-box attack error: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n🔍 POTENTIAL ISSUES ANALYSIS:")
        print("-" * 50)
        print("1. Model Robustness: FastText might be more robust than LSTM")
        print("2. Attack Strategy: Black-box uses word removal, white-box uses gradients") 
        print("3. Perturbation Quality: Different bug generation effectiveness")
        print("4. Similarity Constraints: Too strict thresholds")
        print("5. Word Importance: Different importance calculation methods")
        
        # Test individual bug generation
        print(f"\n🔧 TESTING BUG GENERATION QUALITY:")
        print("-" * 30)
        
        test_words = ["amazing", "terrible", "great", "bad"]
        for word in test_words:
            wb_bugs = []
            # Test white-box bugs
            wb_bugs.append(whitebox_attacker.insert_bug(word))
            wb_bugs.append(whitebox_attacker.delete_bug(word))
            wb_bugs.append(whitebox_attacker.swap_bug(word))
            wb_bugs.append(whitebox_attacker.substitute_c(word))
            wb_bugs = [b for b in wb_bugs if b != word]
            
            # Test black-box bugs  
            bb_bugs = blackbox_attacker.generate_bugs(word)
            
            print(f"'{word}':")
            print(f"  White-box bugs: {wb_bugs}")
            print(f"  Black-box bugs: {bb_bugs}")
            
    except Exception as e:
        print(f"❌ Error in comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compare_attacks()