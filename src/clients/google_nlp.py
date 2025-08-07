import os
from dotenv import load_dotenv
from google.cloud import language_v2

# Load environment variables
load_dotenv()

class GoogleNLPClassifier:
    def __init__(self, credentials_path=None):
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        self.client = language_v2.LanguageServiceClient()

    def classify(self, text: str):
        document = {
            "content": text,
            "type_": language_v2.Document.Type.PLAIN_TEXT,
            "language_code": "en"
        }
        
        response = self.client.analyze_sentiment(
            request={"document": document, "encoding_type": language_v2.EncodingType.UTF8}
        )
        
        sentiment = response.document_sentiment
        score = sentiment.score
        magnitude = sentiment.magnitude
        
        # Determine label based on score
        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"
            
        # Create score dictionary
        scores = {
            "positive": max(score, 0),
            "negative": abs(min(score, 0)),
            "neutral": 1 - abs(score) if abs(score) < 1 else 0
        }
        
        return label, scores