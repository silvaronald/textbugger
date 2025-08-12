import random
import copy
import nltk
from collections import defaultdict
from nltk.corpus import wordnet
from nltk import pos_tag
from difflib import SequenceMatcher
import numpy as np

try:
    # Try relative import first (when used as package)
    from ..utils.text_processing import split_sentences, split_words, fix_spacing
except ImportError:
    # Fall back to absolute import (when src/ is in path)
    from utils.text_processing import split_sentences, split_words, fix_spacing

# Download required NLTK data if not present
def download_nltk_dependencies():
    """Download necessary NLTK data with proper error handling"""
    dependencies = [
        ('tokenizers/punkt', 'punkt'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
    ]
    
    for path, name in dependencies:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {name}")
            try:
                nltk.download(name, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download {name}: {e}")

# Download dependencies
download_nltk_dependencies()


class POSWeightCalculator:
    def __init__(self):
        self.pos_weights = {}
        self.is_trained = False
        self.num_reqs = 0

    def calculate_pos_weights(self, training_data, classifier_fn):
        pos_impact_count = defaultdict(int)
        total_samples = 0
        
        print("Calculando pesos dos POS tags a partir do dataset...")
        
        for text in training_data[:200]: 
            words = split_words(text)
            if len(words) < 2:
                continue
                
            original_label, original_score = classifier_fn(text)
            self.num_reqs += 1
            max_impact = 0
            best_pos = None
            
            # Testa remoção de cada palavra
            for i, word in enumerate(words):
                modified_words = words[:i] + words[i+1:]
                modified_text = " ".join(modified_words)
                
                try:
                    _, new_score = classifier_fn(modified_text)
                    self.num_reqs += 1
                    impact = abs(original_score - new_score)
                    
                    if impact > max_impact:
                        max_impact = impact
                        # Obtém POS tag da palavra com maior impacto
                        try:
                            pos_tags = nltk.pos_tag([word])
                            best_pos = pos_tags[0][1] if pos_tags else None
                        except Exception:
                            print(f"Warning: NLTK POS tagging failed for '{word}', fallback to simple heuristic")
                            continue
                except Exception:
                    continue
            
            if best_pos:
                pos_impact_count[best_pos] += 1
            total_samples += 1
            
            if total_samples % 20 == 0:
                print(f"Processados {total_samples} exemplos...")
        
        if len(pos_impact_count) == 0:
            raise ValueError("Não foi possível calcular pesos POS - nenhum impacto detectado no dataset")
        
        counts = np.array(list(pos_impact_count.values()))
        softmax_weights = np.exp(counts) / np.sum(np.exp(counts))
        self.pos_weights = dict(zip(pos_impact_count.keys(), softmax_weights))
        
        self.is_trained = True
        print(f"Pesos POS calculados: {dict(sorted(self.pos_weights.items(), key=lambda x: x[1], reverse=True))}")
    
    def get_word_priority(self, word):
        try:
            pos_tags = None
            
            try:
                pos_tags = nltk.pos_tag([word])
            except Exception:
                print(f"Warning: NLTK POS tagging falhou, tentando PerceptronTagger")
                try:
                    from nltk.tag import PerceptronTagger
                    tagger = PerceptronTagger()
                    pos_tags = tagger.tag([word])
                except Exception:
                    print(f"Warning: PerceptronTagger falhou, tentando Heuristic")
                    pos = self._simple_pos_heuristic(word)
                    return self.pos_weights.get(pos, 0.0)
            
            if pos_tags:
                pos = pos_tags[0][1] if pos_tags else 'NN'
                if pos in self.pos_weights:
                    return self.pos_weights[pos]
                elif pos[:2] in self.pos_weights:  
                    return self.pos_weights[pos[:2]]
                else:
                    return 0.0
            else:
                return 0.0
                
        except Exception as e:
            print(f"Warning: POS tagging failed for '{word}': {e}")
            return 0.0
    
    def _simple_pos_heuristic(self, word):
        word_lower = word.lower()
        
        # Adjetivos - sufixos comuns
        if word_lower.endswith(('ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ic', 'ed')):
            return 'JJ'
        
        # Advérbios - principalmente -ly
        if word_lower.endswith('ly'):
            return 'RB'
        
        # Verbos - sufixos comuns  
        if word_lower.endswith(('ing', 'ed', 'er', 'est', 'ize', 'ise', 'fy')):
            return 'VB'
        
        # Substantivos - sufixos comuns
        if word_lower.endswith(('tion', 'sion', 'ness', 'ment', 'ship', 'hood', 'ity', 'ty')):
            return 'NN'
        
        # Palavras funcionais específicas
        determiners = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        if word_lower in determiners:
            return 'DT'
        
        prepositions = {'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about', 'into', 'through'}
        if word_lower in prepositions:
            return 'IN'
        
        conjunctions = {'and', 'or', 'but', 'yet', 'so', 'because', 'although', 'while', 'if', 'when', 'where'}
        if word_lower in conjunctions:
            return 'CC'
        
        pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        if word_lower in pronouns:
            return 'PRP'
        
        if word_lower.isdigit() or word_lower in ['one', 'two', 'three', 'first', 'second', 'third']:
            return 'CD'
        
        interjections = {'wow', 'oh', 'ah', 'hey', 'hi', 'hello', 'yes', 'no', 'okay', 'ok'}
        if word_lower in interjections:
            return 'UH'
        
        return 'NN'


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


class FastWordBugger:
    def __init__(self, classifier_fn, similarity_threshold=0.8, pos_threshold=0.0):
        self.classifier = classifier_fn
        self.bug_gen = BugGenerator()
        self.similarity = SemanticSimilarity(threshold=similarity_threshold)
        self.pos_calculator = POSWeightCalculator()
        self.pos_threshold = pos_threshold
        self.training_reqs = 0
        
    def train_pos_weights(self, training_texts):
        if not training_texts:
            raise ValueError("Dados de treinamento são obrigatórios para calcular pesos POS")
        
        self.pos_calculator.calculate_pos_weights(training_texts, self.classifier)
        self.training_reqs = self.pos_calculator.num_reqs

    def filter_words_by_pos(self, words):
        if not words:
            return []
     
        try:
            pos_tags = None
            
            try:
                pos_tags = nltk.pos_tag(words)
            except Exception as e:
                print(f"Warning: NLTK POS tagging failed: {e}")
                pos_tags = [(word, self.pos_calculator._simple_pos_heuristic(word)) for word in words]
            
            word_priorities = []
            for i, (word, pos) in enumerate(pos_tags):
                priority = self.pos_calculator.get_word_priority(word)
                word_priorities.append((i, word, priority))
            
            important_words = [(i, word, priority) for i, word, priority in word_priorities if priority > 0.0]
            
            if len(important_words) == 0:
                print("Warning: Nenhuma palavra passou no filtro POS. Usando palavras com maior prioridade relativa.")
                word_priorities.sort(key=lambda x: x[2], reverse=True)
                important_words = word_priorities[:min(10, len(word_priorities))]
            
            elif len(important_words) < max(2, len(words) * 0.15):
                print(f"Info: Apenas {len(important_words)} palavras passaram no filtro POS. Usando threshold adaptativo.")
                word_priorities.sort(key=lambda x: x[2], reverse=True)
                target_words = max(3, int(len(words) * 0.2)) 
                important_words = word_priorities[:min(target_words, len(word_priorities))]
            

            important_words.sort(key=lambda x: x[2], reverse=True)
            
            return important_words
            
        except Exception as e:
            print(f"Erro no filtro POS: {e}")
            # Fallback: retorna palavras ordenadas por comprimento
            word_priorities = [(i, word, len(word)) for i, word in enumerate(words)]
            word_priorities.sort(key=lambda x: x[2], reverse=True)
            return word_priorities[:min(5, len(words))]

    def attack(self, text, max_modifications=10):
        if not self.pos_calculator.is_trained:
            raise ValueError("Pesos POS não foram treinados! Use train_pos_weights() primeiro")
        
        num_reqs = 0
        num_bugs = 0
        
        original_label, original_score = self.classifier(text)
        num_reqs += 1
        
        adv_text = text
        modifications_made = 0
        
        print(f"Texto original classificado como: {original_label} (score: {original_score:.3f})")
        
        sentences = split_sentences(text)
        important_sentences = []
        
        for sentence in sentences:
            pred_label, score = self.classifier(sentence)
            num_reqs += 1
            
            if pred_label == original_label:
                important_sentences.append((sentence, score))
        
        important_sentences.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Encontradas {len(important_sentences)} sentenças importantes")
        
        for sentence_idx, (sentence, sentence_score) in enumerate(important_sentences):

            words = split_words(sentence)
            
            print(f"Processando sentença {sentence_idx + 1}: '{sentence[:50]}...'")
            
            important_words = self.filter_words_by_pos(words)
            
            print(f"  Palavras filtradas por POS empírico: {len(important_words)} de {len(words)} palavras")
            
            for word_pos, word, priority in important_words[:min(5, len(important_words))]:
                    
                print(f"    Testando palavra: '{word}' (POS priority empírica: {priority:.3f})")
                
                bugs = self.bug_gen.generate_bugs(word)
                best_bug = None
                best_score_drop = 0
                
                for bug in bugs:
                    if bug == word: 
                        continue
                        
                    modified_words = words.copy()
                    modified_words[word_pos] = bug
                    modified_sentence = fix_spacing(" ".join(modified_words))
                    modified_text = adv_text.replace(sentence, modified_sentence)
                    
                    if not self.similarity.similarity(text, modified_text):
                        continue
                    
                    new_label, new_score = self.classifier(modified_text)
                    num_reqs += 1
                    num_bugs += 1
                    
                    score_drop = original_score - new_score
                    
                    if new_label != original_label:
                        print(f"✓ SUCESSO! Label mudou de {original_label} para {new_label}")
                        print(f"  Modificação: '{word}' -> '{bug}'")
                        print(f"  Score: {original_score:.3f} -> {new_score:.3f}")
                        modifications_made += 1
                        return modified_text, original_label, num_reqs, modifications_made, True
                    
                    # Guarda o melhor bug (maior redução de score)
                    if score_drop > best_score_drop:
                        best_score_drop = score_drop
                        best_bug = bug
                
                # Aplica o melhor bug se houver melhoria
                if best_bug and best_score_drop > 0.01:
                    modified_words = words.copy()
                    modified_words[word_pos] = best_bug
                    modified_sentence = fix_spacing(" ".join(modified_words))
                    adv_text = adv_text.replace(sentence, modified_sentence)
                    sentence = modified_sentence  # Atualiza para próximas modificações
                    words = modified_words
                    modifications_made += 1
                    
                    print(f"    Aplicada modificação: '{word}' -> '{best_bug}' (drop: {best_score_drop:.3f})")
        
        # Testa o texto final
        final_label, final_score = self.classifier(adv_text)
        num_reqs += 1
        
        success = final_label != original_label
        
        print(f"\n=== RESULTADO FINAL ===")
        print(f"Texto original: {original_label} ({original_score:.3f})")
        print(f"Texto adversarial: {final_label} ({final_score:.3f})")
        print(f"Modificações feitas: {modifications_made}")
        print(f"Requisições ao modelo: {num_reqs}")
        print(f"Bugs testados: {num_bugs}")
        print(f"Sucesso: {success}")
        
        return adv_text, original_label, num_reqs, modifications_made, success