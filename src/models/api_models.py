"""
API Model Wrappers and Utilities

This module provides wrappers to make API clients compatible with BlackBoxTextBugger.
"""

class APIModelWrapper:
    """
    Wrapper to make API clients compatible with BlackBoxTextBugger
    
    API clients return: (label, scores_dict)
    BlackBoxTextBugger expects: (label, single_score)
    """
    
    def __init__(self, api_client):
        self.api_client = api_client
    
    def __call__(self, text):
        """
        Convert API client output to format expected by BlackBoxTextBugger
        """
        label, scores_dict = self.api_client.classify(text)
        
        # Extract the confidence score for the predicted label
        if label in scores_dict:
            confidence_score = scores_dict[label]
        else:
            # Fallback: use the highest score
            confidence_score = max(scores_dict.values())
            
        return label, confidence_score
    
    def classify(self, text):
        """Alternative method name for consistency"""
        return self.__call__(text)

# Legacy alias for backward compatibility
APIClassifierWrapper = APIModelWrapper