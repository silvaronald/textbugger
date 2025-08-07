import random
import copy
from nltk.corpus import wordnet
from utils import split_sentences, split_words, fix_spacing
from difflib import SequenceMatcher


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
        self.threshold = threshold

    def similarity(self, x, x_adv):
        return SequenceMatcher(None, x, x_adv).ratio() >= self.threshold


class BlackBoxTextBugger:
    def __init__(self, classifier_fn, similarity_threshold=0.8):
        self.classifier = classifier_fn
        self.bug_gen = BugGenerator()
        self.similarity = SemanticSimilarity(threshold=similarity_threshold)

    def attack(self, text):
        num_reqs = 0
        num_bugs = 0

        original_label, original_score = self.classifier(text)
        num_reqs += 1
        adv_text = copy.deepcopy(text)

        # Step 1: Sentence importance filtering and sorting
        sentences = split_sentences(text)
        important_sentences = []
        for s in sentences:
            pred_label, score = self.classifier(s)
            num_reqs += 1
            if pred_label == original_label:
                important_sentences.append((s, score))

        important_sentences.sort(key=lambda x: -x[1])  # descending confidence

        # Step 2: Word importance within selected sentences
        for sentence, _ in important_sentences:
            words = split_words(sentence)
            original_sentence_label, original_sentence_score = self.classifier(sentence)
            num_reqs += 1

            word_scores = []
            for i in range(len(words)):
                modified = " ".join(w for j, w in enumerate(words) if j != i)
                _, score = self.classifier(modified)
                num_reqs += 1

                drop = original_sentence_score - score
                word_scores.append((i, drop))

            word_scores.sort(key=lambda x: -x[1])

            for idx, _ in word_scores:
                num_bugs += 1
                bugs = self.bug_gen.generate_bugs(words[idx])
                best_bug = None
                best_drop = 0

                for bug in bugs:
                    temp_words = words.copy()
                    temp_words[idx] = bug
                    perturbed_sentence = fix_spacing(" ".join(temp_words))
                    perturbed_text = adv_text.replace(sentence, perturbed_sentence)

                    new_label, new_score = self.classifier(perturbed_text)
                    num_reqs += 1

                    drop = original_score - new_score

                    if drop > best_drop and self.similarity.similarity(text, perturbed_text):
                        best_bug = bug
                        best_drop = drop
                        if new_label != original_label:
                            return perturbed_text, original_label, num_reqs, num_bugs, True

                if best_bug:
                    words[idx] = best_bug
                    new_sentence = fix_spacing(" ".join(words))
                    adv_text = adv_text.replace(sentence, new_sentence)
                    sentence = new_sentence


        return adv_text, original_label, num_reqs, num_bugs, False
