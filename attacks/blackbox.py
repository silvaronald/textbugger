import random
import string
import copy
import numpy as np
import spacy
import numpy as np
import tensorflow_hub as hub
from nltk.corpus import wordnet
from utils import split_sentences, split_words

class BugGenerator:
    def __init__(self):
        self.visual_chars = {'o': '0', 'l': '1', 'a': '@', 'e': '3', 's': '$', 'i': '1'}
        self.keyboard_adjacent = {'m': 'n', 'n': 'm'}

    def insert_space(self, word: str) -> str:
        if len(word) > 5:
            return word
        pos = random.randint(1, len(word)-1)
        return word[:pos] + ' ' + word[pos:]

    def delete_char(self, word: str) -> str:
        if len(word) <= 3:
            return word
        pos = random.randint(1, len(word)-2)
        return word[:pos] + word[pos+1:]

    def swap_adjacent(self, word: str) -> str:
        if len(word) <= 4:
            return word
        pos = random.randint(1, len(word)-2)
        return word[:pos] + word[pos+1] + word[pos] + word[pos+2:]

    def substitute_char(self, word: str) -> str:
        chars = list(word)
        idx = random.randint(0, len(chars)-1)
        chars[idx] = self.visual_chars.get(chars[idx].lower(), self.keyboard_adjacent.get(chars[idx], chars[idx]))
        return ''.join(chars)

    def substitute_word(self, word: str) -> str:
        synonyms = wordnet.synsets(word)
        if synonyms:
            lemmas = [l.name().replace("_", " ") for s in synonyms for l in s.lemmas()]
            lemmas = [w for w in set(lemmas) if w.lower() != word.lower()]
            if lemmas:
                return random.choice(lemmas)
        return word

    def generate_bugs(self, word: str):
        if len(word) <= 1:  # pula palavras muito curtas
            return [word]
        return [
            self.insert_space(word),
            self.delete_char(word),
            self.swap_adjacent(word),
            self.substitute_char(word),
            self.substitute_word(word),
        ]

class SemanticSimilarity:
    def __init__(self, threshold=0.8):
        self.encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.threshold = threshold

    def cosine_similarity(self, sent1, sent2):
        emb = self.encoder([sent1, sent2])
        v1, v2 = emb[0].numpy(), emb[1].numpy()
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def is_similar(self, original, perturbed):
        similarity = self.cosine_similarity(original, perturbed)
        return similarity >= self.threshold


class BlackBoxTextBugger:
    def __init__(self, classifier_fn, similarity_threshold=0.8):
        """
        classifier_fn: função de classificação (string) -> (label, score)
        """
        self.classifier = classifier_fn
        self.bug_gen = BugGenerator()
        self.similarity = SemanticSimilarity(threshold=similarity_threshold)

    def score_sentence(self, sentence, original_label):
        _, score = self.classifier(sentence)
        return score[original_label]

    def score_word_importance(self, sentence, word_idx, original_label):
        words = split_words(sentence)
        modified = " ".join(w for i, w in enumerate(words) if i != word_idx)
        _, score = self.classifier(modified)
        return score[original_label]

    # Faltar limitar o número de frases e palavras que serão perturbadas
    def attack(self, text):
        original_label, _ = self.classifier(text)
        adv_text = copy.deepcopy(text)

        sentences = split_sentences(text)
        sentence_scores = [(s, self.score_sentence(s, original_label)) for s in sentences]
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        for sentence, _ in sentence_scores:
            words = split_words(sentence)
            word_scores = [(i, self.score_word_importance(sentence, i, original_label)) for i in range(len(words))]
            word_scores.sort(key=lambda x: x[1])

            for idx, _ in word_scores:
                bugs = self.bug_gen.generate_bugs(words[idx])
                best_bug = None
                best_drop = 0

                for bug in bugs:
                    temp_words = words.copy()
                    temp_words[idx] = bug
                    perturbed_sentence = " ".join(temp_words)
                    perturbed_text = adv_text.replace(sentence, perturbed_sentence)

                    new_label, new_score = self.classifier(perturbed_text)
                    drop = new_score[original_label]

                    if drop < best_drop and self.similarity.is_similar(text, perturbed_text):
                        print(f"Perturbing '{sentence}' by replacing '{words[idx]}' with '{bug}'")
                        best_bug = bug
                        best_drop = drop

                        if new_label != original_label:
                            return perturbed_text

                if best_bug:
                    words[idx] = best_bug
                    adv_text = adv_text.replace(sentence, " ".join(words))

        return adv_text
