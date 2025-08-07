import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import fasttext
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

class LocalModelWrapper:
    """Wrapper for local models including FastText and HuggingFace models"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name == "fasttext":
            self.predict = self.fasttext_predict
        else:
            self.classifier = self.load_hf_model(f"./models/{model_name}")
            self.predict = self.hugging_face_predict

    def fasttext_predict(self, text):
        """Predict using FastText model"""
        model = fasttext.load_model('models/external/fasttext/amazon_review_polarity.bin')
        label, prob = model.predict(text, k=1)
        return label[0], prob[0]
    
    def hugging_face_predict(self, text):
        """Predict using HuggingFace model"""
        pred = self.classifier(text)[0]
        return pred["label"], pred["score"]

    def load_hf_model(self, path):
        """Load HuggingFace model from path"""
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    def classify(self, text):
        """Main classification method"""
        return self.predict(text)

# Legacy alias for backward compatibility
ClassificationWrapper = LocalModelWrapper