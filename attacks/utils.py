import spacy

nlp = spacy.load("en_core_web_sm")

def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def split_words(sentence):
    return [token.text for token in nlp(sentence)]
