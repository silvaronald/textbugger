"""
External API Clients
"""

from .ibm_watson import IBMWatsonClassifier
from .azure_text import AzureTextAnalyticsClassifier  
from .google_nlp import GoogleNLPClassifier
from .aws_comprehend import AWSComprehendClassifier

__all__ = ["IBMWatsonClassifier", "AzureTextAnalyticsClassifier", "GoogleNLPClassifier", "AWSComprehendClassifier"]