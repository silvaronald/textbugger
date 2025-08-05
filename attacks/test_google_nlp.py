from blackbox import BlackBoxAttack
import time

def test_google_nlp_api():
    """
    Test TextBugger attacks against Google Cloud Natural Language API
    This replicates the real-world commercial API attacks from the TextBugger paper
    """
    
    print("="*70)
    print("TESTING TEXTBUGGER ATTACKS AGAINST GOOGLE CLOUD NLP API")
    print("="*70)
    
    try:
        # Initialize attacker
        print("Initializing BlackBoxAttack...")
        attacker = BlackBoxAttack("rtmr")
        
        # Create Google NLP classifier wrapper
        print("Creating Google Cloud NLP API wrapper...")
        google_classifier = attacker.create_google_nlp_classifier_wrapper()
        
        if not google_classifier:
            print("❌ Failed to initialize Google NLP API")
            print("This requires Google Cloud credentials and billing setup")
            return
        
        # Test sentences similar to TextBugger paper
        test_cases = [
            {
                "text": "This movie is amazing and fantastic",
                "expected": "positive",
                "description": "Strong positive sentiment"
            },
            {
                "text": "This film is terrible and awful", 
                "expected": "negative",
                "description": "Strong negative sentiment"
            },
            {
                "text": "I love this wonderful movie",
                "expected": "positive",
                "description": "Positive with emotional words"
            },
            {
                "text": "I hate this boring film",
                "expected": "negative", 
                "description": "Negative with emotional words"
            },
            {
                "text": "The acting was brilliant and the story was compelling",
                "expected": "positive",
                "description": "Complex positive sentence"
            },
            {
                "text": "The plot was confusing and the dialogue was poor",
                "expected": "negative",
                "description": "Complex negative sentence"
            }
        ]
        
        print(f"\n🔍 GOOGLE NLP BASELINE PREDICTIONS")
        print("-" * 50)
        
        # First test Google NLP baseline predictions
        for i, case in enumerate(test_cases):
            print(f"\nTest {i+1}: {case['description']}")
            print(f"Text: '{case['text']}'")
            
            try:
                label = google_classifier(case['text'], return_label=True)
                confidence = google_classifier(case['text'], return_confidence=True)
                probs = google_classifier(case['text'])
                
                sentiment_name = "POSITIVE" if label == 1 else "NEGATIVE"
                print(f"Google NLP: {sentiment_name} (label: {label}, conf: {confidence:.3f})")
                print(f"Probabilities: {probs}")
                
                # Small delay to respect API rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error getting baseline prediction: {e}")
                continue
        
        print(f"\n🎯 TEXTBUGGER ATTACK TESTING")
        print("-" * 50)
        
        successful_attacks = 0
        total_attacks = 0
        
        for i, case in enumerate(test_cases):
            print(f"\n{'='*40}")
            print(f"ATTACKING: '{case['text']}'")
            print(f"{'='*40}")
            
            try:
                # Get original prediction
                original_label = google_classifier(case['text'], return_label=True)
                original_conf = google_classifier(case['text'], return_confidence=True)
                original_name = "POSITIVE" if original_label == 1 else "NEGATIVE"
                
                print(f"Original: {original_name} (conf: {original_conf:.3f})")
                
                # Launch TextBugger attack
                print(f"🚀 Launching TextBugger attack...")
                start_time = time.time()
                
                adversarial_text = attacker.textbugger_blackbox_attack(
                    document_x=case['text'],
                    ground_truth_label_y=original_label,
                    classifier_F=google_classifier,
                    threshold_epsilon=0.6  # Allow 40% similarity loss
                )
                
                attack_time = time.time() - start_time
                total_attacks += 1
                
                if adversarial_text and adversarial_text != case['text']:
                    # Test adversarial prediction
                    time.sleep(1)  # Rate limit
                    adv_label = google_classifier(adversarial_text, return_label=True)
                    adv_conf = google_classifier(adversarial_text, return_confidence=True)
                    adv_name = "POSITIVE" if adv_label == 1 else "NEGATIVE"
                    
                    if adv_label != original_label:
                        print(f"✅ ATTACK SUCCESSFUL! ({attack_time:.2f}s)")
                        print(f"   Original: '{case['text']}'")
                        print(f"   Adversarial: '{adversarial_text}'")
                        print(f"   Prediction: {original_name} → {adv_name}")
                        print(f"   Confidence: {original_conf:.3f} → {adv_conf:.3f}")
                        
                        # Show word changes
                        original_words = case['text'].split()
                        adversarial_words = adversarial_text.split()
                        changes = []
                        for orig, adv in zip(original_words, adversarial_words):
                            if orig != adv:
                                changes.append(f"'{orig}' → '{adv}'")
                        
                        if changes:
                            print(f"   Changes: {', '.join(changes)}")
                        
                        # Calculate similarity
                        similarity = attacker.calculate_semantic_similarity(case['text'], adversarial_text)
                        print(f"   Semantic similarity: {similarity:.3f}")
                        
                        successful_attacks += 1
                        
                    else:
                        print(f"⚠️  Text modified but no label flip:")
                        print(f"   Adversarial: '{adversarial_text}'")
                        print(f"   Still: {adv_name} (conf: {adv_conf:.3f})")
                else:
                    print(f"❌ Attack failed ({attack_time:.2f}s)")
                
            except Exception as e:
                print(f"❌ Error during attack: {e}")
                total_attacks += 1
                
            # Rate limiting for Google API
            time.sleep(2)
            
        print(f"\n📊 ATTACK SUMMARY")
        print("-" * 30)
        print(f"Successful attacks: {successful_attacks}/{total_attacks}")
        if total_attacks > 0:
            success_rate = (successful_attacks / total_attacks) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        print(f"\n🔬 COMPARISON WITH TEXTBUGGER PAPER")
        print("-" * 40)
        print("TextBugger paper reported:")
        print("- 100% success rate against commercial APIs")
        print("- Average attack time: ~4.61 seconds")
        print("- Semantic similarity preservation: 97%+")
        print(f"\nOur results:")
        print(f"- Success rate: {success_rate:.1f}% against Google NLP API")
        print("- Testing against modern 2024 API (more robust than 2018)")
        
    except Exception as e:
        print(f"❌ Error in Google NLP testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_google_nlp_api()