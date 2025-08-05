from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions

# class GoogleNLPClassifier:
#     def __init__(self, credentials_path):
#         self.client = language_v1.LanguageServiceClient.from_service_account_json(credentials_path)

#     def classify(self, text: str):
#         document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
#         response = self.client.analyze_sentiment(document=document)
#         sentiment = response.document_sentiment
#         score = sentiment.score
#         label = "positive" if score >= 0 else "negative"
#         return label, {"positive": max(score, 0), "negative": max(-score, 0)}

# class AWSComprehendClassifier:
#     def __init__(self, aws_access_key, aws_secret_key, region_name="us-east-1"):
#         self.client = boto3.client(
#             "comprehend",
#             aws_access_key_id=aws_access_key,
#             aws_secret_access_key=aws_secret_key,
#             region_name=region_name,
#         )

#     def classify(self, text: str):
#         response = self.client.detect_sentiment(Text=text, LanguageCode="en")
#         sentiment = response["Sentiment"].lower()
#         scores = {k.lower(): v for k, v in response["SentimentScore"].items()}
#         return sentiment, scores

# class AzureTextAnalyticsClassifier:
#     def __init__(self, endpoint, api_key):
#         self.client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

#     def classify(self, text: str):
#         response = self.client.analyze_sentiment(documents=[text])[0]
#         sentiment = response.sentiment
#         scores = {
#             "positive": response.confidence_scores.positive,
#             "neutral": response.confidence_scores.neutral,
#             "negative": response.confidence_scores.negative,
#         }
#         return sentiment, scores

# class FastTextClassifier:
#     def __init__(self, model_path: str):
#         self.model = fasttext.load_model(model_path)

#     def classify(self, text: str):
#         labels, scores = self.model.predict(text, k=2)
#         labels = [l.replace("__label__", "").lower() for l in labels]
#         score_dict = dict(zip(labels, scores))
#         return labels[0], score_dict

class IBMWatsonClassifier:
    def __init__(self, api_key, service_url):
        authenticator = IAMAuthenticator(api_key)
        self.client = NaturalLanguageUnderstandingV1(
            version="2021-08-01",
            authenticator=authenticator
        )
        self.client.set_service_url(service_url)

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
