import spacy
import re

nlp = spacy.load("en_core_web_sm")

def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def split_words(sentence):
    return [token.text for token in nlp(sentence)]

def fix_spacing(text):
    return re.sub(r'\s+([?.!,])', r'\1', text)
