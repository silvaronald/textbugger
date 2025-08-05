from blackbox import BlackBoxTextBugger
from model_wrapper import ClassificationWrapper
#import fasttext
#from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

text = "I don't know what to say about this product, but i also kind of love it, it's amazing, wonderful, terrific!"

cw = ClassificationWrapper("fasttext")
print(cw.predict(text))

cw = ClassificationWrapper("sst2")
print(cw.predict(text))

cw = ClassificationWrapper("toxic-bert")
print(cw.predict(text))

cw = ClassificationWrapper("twitter-roberta")
print(cw.predict(text))