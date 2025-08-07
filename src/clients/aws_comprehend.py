import os
from dotenv import load_dotenv
import boto3

# Load environment variables
load_dotenv()

class AWSComprehendClassifier:
    def __init__(self, aws_access_key=None, aws_secret_key=None, region_name="us-east-1"):
        if aws_access_key and aws_secret_key:
            self.client = boto3.client(
                "comprehend",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region_name,
            )
        else:
            # Use environment variables or default AWS credentials
            self.client = boto3.client("comprehend", region_name=region_name)

    def classify(self, text: str):
        response = self.client.detect_sentiment(Text=text, LanguageCode="en")
        sentiment = response["Sentiment"].lower()
        scores = {k.lower(): v for k, v in response["SentimentScore"].items()}
        return sentiment, scores