import os
from dotenv import load_dotenv
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions

# Load environment variables
load_dotenv()

class IBMWatsonClassifier:
    def __init__(self, api_key=None, service_url=None):
        self.api_key = api_key or os.getenv('IBM_API_KEY')
        self.service_url = service_url or os.getenv('IBM_SERVICE_URL')
        
        if not self.api_key or not self.service_url:
            raise ValueError("IBM API key and service URL must be provided either as parameters or environment variables")
            
        authenticator = IAMAuthenticator(self.api_key)
        self.client = NaturalLanguageUnderstandingV1(
            version="2022-04-07",
            authenticator=authenticator
        )
        self.client.set_service_url(self.service_url)

    def classify(self, text: str):
        response = self.client.analyze(
            text=text,
            features=Features(sentiment=SentimentOptions())
        ).get_result()

        sentiment = response["sentiment"]["document"]["label"]
        score = response["sentiment"]["document"]["score"]
        scores = {
            "positive": score if sentiment == "positive" else 0,
            "negative": -score if sentiment == "negative" else 0,
            "neutral": 1 - abs(score) if sentiment == "neutral" else 0
        }
        return sentiment, scores