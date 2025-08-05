import fasttext

def test_fasttext():
    try:
        # Load the model
        model = fasttext.load_model('amazon_review_polarity.bin')
        
        # Test predictions
        test_texts = [
            "This movie is absolutely terrible and boring",
            "I love this product, it's amazing and wonderful"
        ]
        
        for text in test_texts:
            predictions = model.predict(text)
            labels = predictions[0]
            probabilities = predictions[1]
            
            print(f"Text: {text}")
            print(f"Prediction: {labels[0]} (confidence: {probabilities[0]:.4f})")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error testing fastText: {e}")

if __name__ == "__main__":
    test_fasttext()