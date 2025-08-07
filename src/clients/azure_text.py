import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

# Load environment variables
load_dotenv()

class AzureTextAnalyticsClassifier:
    def __init__(self, endpoint=None, api_key=None):
        self.endpoint = endpoint or os.getenv('AZURE_TEXT_ANALYTICS_ENDPOINT')
        self.api_key = api_key or os.getenv('AZURE_TEXT_ANALYTICS_KEY')
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure endpoint and API key must be provided either as parameters or environment variables")
            
        self.client = TextAnalyticsClient(
            endpoint=self.endpoint, 
            credential=AzureKeyCredential(self.api_key)
        )

    def classify(self, text: str):
        response = self.client.analyze_sentiment(documents=[text])[0]
        sentiment = response.sentiment.lower()
        scores = {
            "positive": response.confidence_scores.positive,
            "neutral": response.confidence_scores.neutral,
            "negative": response.confidence_scores.negative,
        }
        return sentiment, scores