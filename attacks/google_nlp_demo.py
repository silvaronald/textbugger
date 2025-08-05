from google.cloud import language_v2

def sample_analyze_sentiment(text_content: str = "I am so happy and joyful.") -> dict:
    """
    Analyzes Sentiment in a string.

    Args:
      text_content: The text content to analyze.
      
    Returns:
      Dictionary with sentiment analysis results
    """

    client = language_v2.LanguageServiceClient()

    # Available types: PLAIN_TEXT, HTML
    document_type_in_plain_text = language_v2.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language_code = "en"
    document = {
        "content": text_content,
        "type_": document_type_in_plain_text,
        "language_code": language_code,
    }

    # Available values: NONE, UTF8, UTF16, UTF32
    # See https://cloud.google.com/natural-language/docs/reference/rest/v2/EncodingType.
    encoding_type = language_v2.EncodingType.UTF8

    response = client.analyze_sentiment(
        request={"document": document, "encoding_type": encoding_type}
    )
    
    # Get overall sentiment of the input document
    print(f"Document sentiment score: {response.document_sentiment.score}")
    print(f"Document sentiment magnitude: {response.document_sentiment.magnitude}")
    
    # Get sentiment for all sentences in the document
    for sentence in response.sentences:
        print(f"Sentence text: {sentence.text.content}")
        print(f"Sentence sentiment score: {sentence.sentiment.score}")
        print(f"Sentence sentiment magnitude: {sentence.sentiment.magnitude}")

    # Get the language of the text, which will be the same as
    # the language specified in the request or, if not specified,
    # the automatically-detected language.
    print(f"Language of the text: {response.language_code}")
    
    return {
        "score": response.document_sentiment.score,
        "magnitude": response.document_sentiment.magnitude,
        "language": response.language_code
    }

def test_google_nlp_baseline():
    """
    Test Google NLP API with various sentiment examples
    """
    
    print("="*60)
    print("GOOGLE CLOUD NLP API - BASELINE TESTING")
    print("="*60)
    
    # Test texts from your Colab
    test_texts = [
        "I am so happy and joyful.",
        "This movie is amazing and fantastic",
        "This film is terrible and awful", 
        "I love this wonderful movie",
        "I hate this boring film",
        "The acting was brilliant",
        "The plot was confusing and poor"
    ]

    try:
        for i, text in enumerate(test_texts):
            print(f"\n--- Test {i+1} ---")
            print(f"Text: '{text}'")
            result = sample_analyze_sentiment(text)
            
            # Convert to binary classification like TextBugger
            sentiment_label = "POSITIVE" if result["score"] >= 0 else "NEGATIVE"
            print(f"Binary Classification: {sentiment_label}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Error testing Google NLP: {e}")
        print("Make sure you have:")
        print("1. Google Cloud project with Natural Language API enabled")
        print("2. Authentication: gcloud auth application-default login")
        print("3. GOOGLE_APPLICATION_CREDENTIALS environment variable set")

if __name__ == "__main__":
    test_google_nlp_baseline()