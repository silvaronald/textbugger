import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import fasttext
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

class ClassificationWrapper():
    def __init__(self, model):
        if model == "fasttext":
            self.predict = self.fasttext
        else:
            self.classifier = self.load_hf_model(f"./models/{model}")
            self.predict = self.hugging_face

    def fasttext(self, text):
        model = fasttext.load_model('models/amazon_review_polarity.bin')
        label, prob = model.predict(text, k=1)
        return label[0], prob[0]
    
    def hugging_face(self, text):
        pred = self.classifier(text)[0]
        return pred["label"], pred["score"]

    def load_hf_model(self, path):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)