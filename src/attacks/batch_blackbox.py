import random
import copy
from nltk.corpus import wordnet
try:
    # Try relative import first (when used as package)
    from ..utils.text_processing import split_sentences, split_words, fix_spacing
    from .blackbox import BugGenerator, BlackBoxTextBugger
except ImportError:
    # Fall back to absolute import (when src/ is in path)
    from utils.text_processing import split_sentences, split_words, fix_spacing
    from attacks.blackbox import BugGenerator, BlackBoxTextBugger
from difflib import SequenceMatcher
from typing import List, Tuple, Dict, Optional
from enum import Enum


class BatchLevel(Enum):
    SINGLE_WORD = "single_word"      # True 1-by-1 attacks (for comparison)
    MULTI_WORD = "multi_word"        # Perturbations to multiple words
    ADAPTIVE_DYNAMIC = "adaptive_dynamic"        # Adaptive batch sizing with dynamic adjustments (±2)
    ADAPTIVE_GRADUAL = "adaptive_gradual"        # Adaptive batch sizing with gradual adjustments (±1)


class Perturbation:
    """Represents a single perturbation operation"""
    def __init__(self, word_index: int, sentence_index: int, bug_type: str, 
                 original_word: str, perturbed_word: str, importance_score: float = 0.0):
        self.word_index = word_index
        self.sentence_index = sentence_index
        self.bug_type = bug_type
        self.original_word = original_word
        self.perturbed_word = perturbed_word
        self.importance_score = importance_score
    
    def __repr__(self):
        return f"Perturbation({self.original_word}->{self.perturbed_word}, {self.bug_type}, score={self.importance_score:.3f})"
    

class BatchPerturbationManager:
    """Manages batch perturbation operations"""
    
    def __init__(self, batch_level: BatchLevel = BatchLevel.MULTI_WORD, batch_size: int = 3):
        self.batch_level = batch_level
        self.batch_size = batch_size
        self.bug_generator = BugGenerator()
    
    
    def create_multi_word_batch_with_best_selection(self, important_words: List[Tuple[int, int, str, float]], 
                                                   batch_size: int = None, classifier_fn=None, 
                                                   current_text: str = None, original_score: float = None, 
                                                   similarity_fn=None, original_text: str = None) -> List[Perturbation]:
        """Create batch of perturbations using BEST confidence drop selection (SAME as 1-by-1)"""
        if batch_size is None:
            batch_size = self.batch_size
        
        perturbations = []
        
        # Take top batch_size words
        for word_idx, sentence_idx, word, importance_score in important_words[:batch_size]:
            bugs = self.bug_generator.generate_bugs(word)
            valid_bugs = [bug for bug in bugs if bug != word]
            
            # Skip this word if no valid perturbations available
            if not valid_bugs:
                print(f"    🔍 No valid perturbations for word: '{word}'")
                continue
            
            # Find BEST perturbation based on confidence drop (SAME as 1-by-1)
            best_bug = None
            best_drop = 0
            
            if classifier_fn and current_text and original_score is not None:
                for bug in valid_bugs:
                    # Apply single perturbation to current text to test
                    temp_perturbation = Perturbation(word_idx, sentence_idx, "test", word, bug, importance_score)
                    test_text = self.apply_perturbations_to_text(current_text, [temp_perturbation])
                    
                    # Test this perturbation
                    _, new_score = classifier_fn(test_text)
                    drop = original_score - new_score
                    
                    # Keep if better drop AND maintains similarity (SAME as 1-by-1)
                    if drop > best_drop and (similarity_fn is None or similarity_fn(original_text, test_text)):
                        best_bug = bug
                        best_drop = drop
            
            # Use best bug or random if no classifier provided
            perturbed_word = best_bug if best_bug else random.choice(valid_bugs)
            selection_type = "best_drop" if best_bug else "random"
            
            print(f"    🔍 Perturbing: '{word}' → '{perturbed_word}' ({selection_type}, drop: {best_drop:.3f})")
            
            perturbations.append(Perturbation(
                word_index=word_idx,
                sentence_index=sentence_idx,
                bug_type=selection_type,
                original_word=word,
                perturbed_word=perturbed_word,
                importance_score=importance_score
            ))
        
        return perturbations
    
    def create_multi_word_batch(self, important_words: List[Tuple[int, int, str, float]], 
                               batch_size: int = None) -> List[Perturbation]:
        """Create batch of perturbations across multiple words (backward compatibility)"""
        return self.create_multi_word_batch_with_best_selection(important_words, batch_size)
    
    def create_adaptive_batch(self, important_words: List[Tuple[int, int, str, float]], 
                             similarity_fn, original_text: str, 
                             max_batch_size: int = None) -> List[Perturbation]:
        """Create adaptive batch that respects similarity constraints"""
        if max_batch_size is None:
            # More conservative: start with base batch_size instead of 2x
            max_batch_size = min(self.batch_size, len(important_words))
        
        best_batch = []
        current_size = max_batch_size
        
        while current_size >= 1:
            batch = self.create_multi_word_batch(important_words, current_size)
            
            if batch:
                # Test similarity without querying model
                test_text = self.apply_perturbations_to_text(original_text, batch)
                if similarity_fn(original_text, test_text):
                    return batch
            
            current_size = max(1, current_size // 2)
        
        # Fallback: return single best perturbation
        if important_words:
            return self.create_multi_word_batch(important_words, 1)
        
        return []
    
    def create_confidence_driven_batch(self, important_words: List[Tuple[int, int, str, float]], 
                                     confidence_score: float,
                                     confidence_thresholds: Dict[str, float] = None) -> List[Perturbation]:
        """Create batch size based on model confidence"""
        if confidence_thresholds is None:
            confidence_thresholds = {
                "very_high": 0.9,    # Confidence > 0.9 → large batch (need more perturbations)
                "high": 0.75,        # Confidence > 0.75 → medium batch  
                "medium": 0.6,       # Confidence > 0.6 → small batch
                "low": 0.0           # Confidence ≤ 0.6 → minimal batch
            }
        
        # Determine batch size based on confidence (more conservative for better success rates)
        if confidence_score >= confidence_thresholds["very_high"]:
            # Very confident model → moderate attack (reduced from 6 to 3)
            target_batch_size = min(3, len(important_words))
            print(f"🔥 Very high confidence ({confidence_score:.3f}) → moderate batch size: {target_batch_size}")
        elif confidence_score >= confidence_thresholds["high"]:
            # High confidence → small attack (reduced from 4 to 2)  
            target_batch_size = min(2, len(important_words))
            print(f"🔶 High confidence ({confidence_score:.3f}) → small batch size: {target_batch_size}")
        elif confidence_score >= confidence_thresholds["medium"]:
            # Medium confidence → minimal attack (reduced from 2 to 1)
            target_batch_size = min(1, len(important_words))
            print(f"🔸 Medium confidence ({confidence_score:.3f}) → minimal batch size: {target_batch_size}")
        else:
            # Low confidence → single word attack
            target_batch_size = min(1, len(important_words))
            print(f"🔹 Low confidence ({confidence_score:.3f}) → single word attack: {target_batch_size}")
        
        # Create batch with determined size
        return self.create_multi_word_batch(important_words, target_batch_size)
    
    def apply_perturbations_to_text(self, original_text: str, 
                                   perturbations: List[Perturbation]) -> str:
        """Apply a batch of perturbations to text"""
        if not perturbations:
            return original_text
        
        # Group perturbations by sentence
        sentence_perturbations = {}
        sentences = split_sentences(original_text)
        
        for pert in perturbations:
            if pert.sentence_index not in sentence_perturbations:
                sentence_perturbations[pert.sentence_index] = []
            sentence_perturbations[pert.sentence_index].append(pert)
        
        # Apply perturbations sentence by sentence
        modified_sentences = sentences.copy()
        
        for sentence_idx, perturbs in sentence_perturbations.items():
            if sentence_idx < len(sentences):
                words = split_words(sentences[sentence_idx])
                
                # Sort by word index in descending order to avoid index shifting
                perturbs.sort(key=lambda x: x.word_index, reverse=True)
                
                for pert in perturbs:
                    if pert.word_index < len(words):
                        words[pert.word_index] = pert.perturbed_word
                
                modified_sentences[sentence_idx] = fix_spacing(" ".join(words))
        
        return " ".join(modified_sentences)


class BatchBlackBoxTextBugger(BlackBoxTextBugger):
    """
    Query-efficient TextBugger using batch perturbations
    
    Inherits from BlackBoxTextBugger and overrides:
    - attack(): Implements iterative batch-based attacks instead of single-word attacks
    - similarity_check(): Uses simple string similarity for efficiency instead of semantic similarity
    """
    
    def __init__(self, classifier_fn, similarity_threshold=0.8, 
                 batch_level: BatchLevel = BatchLevel.MULTI_WORD, 
                 batch_size: int = 3):
        # Initialize parent class (gives us self.classifier, self.bug_gen, self.similarity)
        super().__init__(classifier_fn, similarity_threshold)
        
        # Add batch-specific attributes
        self.batch_level = batch_level
        self.batch_size = batch_size
        self.perturbation_manager = BatchPerturbationManager(batch_level, batch_size)
        
    def similarity_check(self, original: str, modified: str) -> bool:
        """Simple string similarity for efficiency and consistency with 1-by-1 attacks"""
        ratio = SequenceMatcher(None, original, modified).ratio()
        return ratio >= self.similarity.threshold
    
    def get_word_importance_scores(self, text: str) -> List[Tuple[int, int, str, float]]:
        """Get importance scores for words across sentences, matching 1-by-1 attack logic"""
        sentences = split_sentences(text)
        original_label, original_score = self.classifier(text)
        
        # Step 1: Sentence importance filtering and sorting (SAME as 1-by-1)
        important_sentences = []
        for sentence_idx, sentence in enumerate(sentences):
            pred_label, score = self.classifier(sentence)
            if pred_label == original_label:  # Only keep sentences that maintain original label
                important_sentences.append((sentence_idx, sentence, score))
        
        important_sentences.sort(key=lambda x: -x[2])  # Sort by confidence (descending)
        print(f"🔍 Filtered sentences: {len(important_sentences)}/{len(sentences)} maintain original label")
        
        # Step 2: Word importance within selected sentences (SAME as 1-by-1)
        important_words = []  # (word_idx_in_sentence, sentence_idx, word, importance_score)
        all_word_scores = []  # For debugging
        
        for sentence_idx, sentence, sentence_score in important_sentences:
            words = split_words(sentence)
            
            word_scores = []
            for word_idx, word in enumerate(words):
                # Calculate importance by removal (SAME as 1-by-1)
                modified_words = [w for i, w in enumerate(words) if i != word_idx]
                modified_sentence = " ".join(modified_words)
                _, modified_score = self.classifier(modified_sentence)
                
                drop = sentence_score - modified_score  # SAME as 1-by-1
                word_scores.append((word_idx, drop))
                all_word_scores.append((word, drop))
                
                # More lenient filtering - allow more words for batch attacks
                if (len(word) > 1 and   # Allow 2+ letter words 
                    word.isalpha() and  # Skip punctuation only
                    word.lower() not in {'the', 'a', 'an'}):  # Minimal stopword list
                    important_words.append((word_idx, sentence_idx, word, drop))
            
            word_scores.sort(key=lambda x: -x[1])  # Sort by importance drop (SAME as 1-by-1)
        
        # DEBUG: Show all word scores
        print(f"🔍 All word importance scores: {[(w, f'{s:.3f}') for w, s in all_word_scores]}")
        print(f"🔍 After filtering: {len(important_words)} words remain")
        
        # Sort by importance (descending) - include words with negative importance too
        important_words.sort(key=lambda x: x[3], reverse=True)
        return important_words
    
    def attack(self, text: str) -> Tuple[str, str, int, int, bool]:
        """
        Iterative batch-based attack implementation
        Returns: (adversarial_text, original_label, num_requests, num_perturbations, success)
        """
        num_reqs = 0
        total_perturbations = 0
        current_text = text
        
        # Get original prediction
        original_label, original_score = self.classifier(text)
        num_reqs += 1
        
        # Get word importance scores (shared across all strategies)
        important_words = self.get_word_importance_scores(text)
        num_reqs += len(split_sentences(text)) + len(important_words) + len(split_sentences(text))  # Approximate query count
        
        if not important_words:
            return text, original_label, num_reqs, total_perturbations, False
        
        # DEBUG: Show word importance scores
        print(f"🔍 Top important words: {[(word, f'{score:.3f}') for _, _, word, score in important_words[:10]]}")
        
        # Strategy-specific iterative attacks
        if self.batch_level == BatchLevel.SINGLE_WORD:
            return self._true_single_word_attack(text, original_label, original_score, num_reqs)
        elif self.batch_level == BatchLevel.MULTI_WORD:
            return self._iterative_multi_word_attack(text, important_words, original_label, original_score, num_reqs, self.batch_size)
        elif self.batch_level == BatchLevel.ADAPTIVE_GRADUAL:
            return self._iterative_adaptive_gradual_attack(text, important_words, original_label, original_score, num_reqs)
        else:  # ADAPTIVE_DYNAMIC
            return self._iterative_adaptive_dynamic_attack(text, important_words, original_label, original_score, num_reqs)
    
    def _true_single_word_attack(self, text: str, original_label: str, original_score: float, num_reqs: int) -> Tuple[str, str, int, int, bool]:
        """
        TRUE 1-by-1 attack - Uses SAME logic as batch attacks but with batch_size=1
        This guarantees identical behavior to batch methods for fair comparison
        """
        total_perturbations = 0
        used_word_indices = set()
        attempt = 0
        current_text = text  # Keep track of accumulated adversarial text
        
        # Get word importance scores using SAME method as batch attacks
        important_words = self.get_word_importance_scores(text)
        num_reqs += len(split_sentences(text)) + len(important_words) + len(split_sentences(text))  # Approximate query count
        
        if not important_words:
            return text, original_label, num_reqs, total_perturbations, False
        
        print(f"🔍 1-by-1 attack using {len(important_words)} important words")
        
        # Process words one by one (batch_size = 1)
        while len(used_word_indices) < len(important_words):
            attempt += 1
            print(f"🔸 1-by-1 attempt {attempt}: single word attack")
            
            # Find next unused word that still exists at its expected position
            current_sentences = split_sentences(current_text)
            available_words = []
            for i, (word_idx, sentence_idx, word, score) in enumerate(important_words):
                if i not in used_word_indices:
                    # Check if the word still exists at the expected position
                    if sentence_idx < len(current_sentences):
                        current_sentence_words = split_words(current_sentences[sentence_idx])
                        if word_idx < len(current_sentence_words) and current_sentence_words[word_idx] == word:
                            available_words.append((word_idx, sentence_idx, word, score))
            
            if not available_words:
                print(f"  🛑 No more candidate words available in current text")
                break
            
            # Create single-word batch using SAME method as batch attacks
            batch = self.perturbation_manager.create_multi_word_batch_with_best_selection(
                available_words[:1], 1,  # batch_size = 1
                classifier_fn=self.classifier, current_text=current_text, original_score=original_score,
                similarity_fn=lambda orig, test: self.similarity_check(orig, test), original_text=text
            )
            
            if not batch:
                print(f"  🛑 No valid perturbations generated")
                break
                
            # Track which word we used (same as batch logic)
            for pert in batch:
                for i, (word_idx, sentence_idx, word, score) in enumerate(important_words):
                    if pert.original_word == word and pert.word_index == word_idx:
                        used_word_indices.add(i)
                        break
            
            # Apply single perturbation to current accumulated text
            adversarial_text = self.perturbation_manager.apply_perturbations_to_text(current_text, batch)
            total_perturbations += len(batch)
            
            # DEBUG: Show what was actually changed
            print(f"  🔍 Current:     {current_text}")
            print(f"  🔍 Adversarial: {adversarial_text}")
            print(f"  🔍 Perturbation applied: {[f'{p.original_word}→{p.perturbed_word}' for p in batch]}")
            
            # Test with model
            new_label, new_score = self.classifier(adversarial_text)
            num_reqs += 1
            
            if new_label != original_label:
                print(f"  ✅ Success! Flipped {original_label} → {new_label}")
                return adversarial_text, original_label, num_reqs, total_perturbations, True
            
            print(f"  ⚪ No flip ({new_label}, conf: {new_score:.3f})")
            
            # SAME similarity check logic as batch attacks
            if self.similarity_check(text, adversarial_text):
                # Keep the adversarial text for next iteration (accumulate perturbations)
                current_text = adversarial_text
                print(f"  ✅ Similarity maintained, keeping perturbation")
            else:
                similarity_score = SequenceMatcher(None, text, adversarial_text).ratio()
                print(f"  ❌ Similarity violated ({similarity_score:.3f}), discarding perturbation")
        
        return current_text, original_label, num_reqs, total_perturbations, False
    
    def _iterative_multi_word_attack(self, text: str, important_words: List, original_label: str, original_score: float, num_reqs: int, batch_size: int) -> Tuple[str, str, int, int, bool]:
        """Multi-word attack: Try batch_size words at a time, exhausting all word combinations, accumulating successful perturbations"""
        total_perturbations = 0
        used_word_indices = set()
        attempt = 0
        current_text = text  # Keep track of accumulated adversarial text
        
        while True:
            attempt += 1
            print(f"🔸 Multi-word attempt {attempt}: Trying {batch_size} words...")
            
            # Find next batch_size unused words that still exist at their expected position
            current_sentences = split_sentences(current_text)
            available_words = []
            for i, (word_idx, sentence_idx, word, score) in enumerate(important_words):
                if i not in used_word_indices:
                    # Check if the word still exists at the expected position
                    if sentence_idx < len(current_sentences):
                        current_sentence_words = split_words(current_sentences[sentence_idx])
                        if word_idx < len(current_sentence_words) and current_sentence_words[word_idx] == word:
                            available_words.append((word_idx, sentence_idx, word, score))
            
            # Stop if no more candidate words available
            if len(available_words) < batch_size:
                print(f"  🛑 No more word combinations available (need {batch_size}, have {len(available_words)})")
                break
            
            batch = self.perturbation_manager.create_multi_word_batch_with_best_selection(
                available_words[:batch_size], batch_size, 
                classifier_fn=self.classifier, current_text=current_text, original_score=original_score,
                similarity_fn=lambda orig, test: self.similarity_check(orig, test), original_text=text
            )
            if not batch:
                print(f"  🛑 No valid perturbations generated")
                break
                
            # Track which words we used
            for pert in batch:
                for i, (word_idx, sentence_idx, word, score) in enumerate(important_words):
                    if pert.original_word == word and pert.word_index == word_idx:
                        used_word_indices.add(i)
                        break
            
            # Apply batch to current accumulated text
            adversarial_text = self.perturbation_manager.apply_perturbations_to_text(current_text, batch)
            total_perturbations += len(batch)
            
            # DEBUG: Show what was actually changed
            print(f"  🔍 Current:     {current_text}")
            print(f"  🔍 Adversarial: {adversarial_text}")
            print(f"  🔍 Perturbations applied: {[f'{p.original_word}→{p.perturbed_word}' for p in batch]}")
            
            # Note: We don't stop on similarity violation to match 1-by-1 attack behavior
            # The similarity check is used implicitly through perturbation selection
            
            # Test with model
            new_label, new_score = self.classifier(adversarial_text)
            num_reqs += 1
            
            if new_label != original_label:
                print(f"  ✅ Success! Flipped {original_label} → {new_label}")
                return adversarial_text, original_label, num_reqs, total_perturbations, True
            
            print(f"  ⚪ No flip ({new_label}, conf: {new_score:.3f})")
            
            # Only keep the perturbations if they maintain similarity (match 1-by-1 behavior)
            if self.similarity_check(text, adversarial_text):
                # Keep the adversarial text for next iteration (accumulate perturbations)
                current_text = adversarial_text
                print(f"  ✅ Similarity maintained, keeping perturbations")
            else:
                similarity_score = SequenceMatcher(None, text, adversarial_text).ratio()
                print(f"  ❌ Similarity violated ({similarity_score:.3f}), discarding batch")
        
        return current_text, original_label, num_reqs, total_perturbations, False
    
    def _iterative_adaptive_gradual_attack(self, text: str, important_words: List, original_label: str, original_score: float, num_reqs: int) -> Tuple[str, str, int, int, bool]:
        """Adaptive gradual attack: Adjust batch size gradually (±1) based on model confidence, accumulate perturbations"""
        total_perturbations = 0
        current_batch_size = 1  # Start conservative
        confidence_thresholds = {"high": 0.7, "medium": 0.35}
        used_word_indices = set()
        attempt = 0
        current_text = text  # Keep track of accumulated adversarial text
        
        while len(used_word_indices) < len(important_words):
            attempt += 1
            print(f"🔸 Adaptive gradual attempt {attempt}: batch size {current_batch_size}")
            
            # Get available words that still exist at their expected position
            current_sentences = split_sentences(current_text)
            available_words = []
            for i, (word_idx, sentence_idx, word, score) in enumerate(important_words):
                if i not in used_word_indices:
                    # Check if the word still exists at the expected position
                    if sentence_idx < len(current_sentences):
                        current_sentence_words = split_words(current_sentences[sentence_idx])
                        if word_idx < len(current_sentence_words) and current_sentence_words[word_idx] == word:
                            available_words.append((word_idx, sentence_idx, word, score))
            
            if len(available_words) < current_batch_size:
                if current_batch_size > 1:
                    current_batch_size = len(available_words)
                    print(f"  🔄 Reducing batch size to {current_batch_size} (remaining words)")
                else:
                    print(f"  🛑 No more candidate words available")
                    break
            
            batch = self.perturbation_manager.create_multi_word_batch_with_best_selection(
                available_words, current_batch_size,
                classifier_fn=self.classifier, current_text=current_text, original_score=original_score,
                similarity_fn=lambda orig, test: self.similarity_check(orig, test), original_text=text
            )
            if not batch:
                print(f"  🛑 No valid perturbations generated")
                break
                
            # Track used words
            for pert in batch:
                for i, (word_idx, sentence_idx, word, score) in enumerate(important_words):
                    if pert.original_word == word and pert.word_index == word_idx:
                        used_word_indices.add(i)
                        break
                        
            # Apply batch to current accumulated text
            adversarial_text = self.perturbation_manager.apply_perturbations_to_text(current_text, batch)
            total_perturbations += len(batch)
            
            # Note: We don't stop on similarity violation to match 1-by-1 attack behavior
            # The similarity check is used implicitly through perturbation selection
            
            new_label, new_score = self.classifier(adversarial_text)
            num_reqs += 1
            
            if new_label != original_label:
                print(f"  ✅ Success! Flipped {original_label} → {new_label}")
                return adversarial_text, original_label, num_reqs, total_perturbations, True
            
            # Only keep the perturbations if they maintain similarity (match 1-by-1 behavior)
            if self.similarity_check(text, adversarial_text):
                # Keep the adversarial text for next iteration (accumulate perturbations)
                current_text = adversarial_text
                print(f"  ✅ Similarity maintained, keeping perturbations")
            else:
                similarity_score = SequenceMatcher(None, text, adversarial_text).ratio()
                print(f"  ❌ Similarity violated ({similarity_score:.3f}), discarding batch")
            
            # Adjust batch size based on confidence (but limited by available words)
            max_possible_batch = min(4, len(important_words) - len(used_word_indices))
            
            if new_score >= confidence_thresholds["high"]:
                current_batch_size = min(max_possible_batch, current_batch_size + 1)
                print(f"  🔥 High confidence ({new_score:.3f}), increasing batch to {current_batch_size}")
            elif new_score >= confidence_thresholds["medium"]:
                print(f"  🔶 Medium confidence ({new_score:.3f}), keeping batch size {current_batch_size}")
            else:
                current_batch_size = max(1, current_batch_size - 1)
                print(f"  🔹 Lower confidence ({new_score:.3f}), reducing batch to {current_batch_size}")
        
        return current_text, original_label, num_reqs, total_perturbations, False
    
    def _iterative_adaptive_dynamic_attack(self, text: str, important_words: List, original_label: str, original_score: float, num_reqs: int) -> Tuple[str, str, int, int, bool]:
        """Adaptive dynamic attack: Adaptively adjust batch sizes with dynamic changes (±2) based on confidence, accumulate perturbations"""
        total_perturbations = 0
        # current_batch_size = min(self.batch_size, len(important_words))
        current_batch_size = 1
        used_word_indices = set()
        attempt = 0
        current_text = text  # Keep track of accumulated adversarial text
        
        while len(used_word_indices) < len(important_words):
            attempt += 1
            print(f"🔸 Adaptive dynamic attempt {attempt}: batch size {current_batch_size}")
            
            # Always respect importance order: take most important unused words that still exist at their expected position
            current_sentences = split_sentences(current_text)
            available_words = []
            for i, (word_idx, sentence_idx, word, score) in enumerate(important_words):
                if i not in used_word_indices:
                    # Check if the word still exists at the expected position
                    if sentence_idx < len(current_sentences):
                        current_sentence_words = split_words(current_sentences[sentence_idx])
                        if word_idx < len(current_sentence_words) and current_sentence_words[word_idx] == word:
                            available_words.append((word_idx, sentence_idx, word, score))
            
            # If we don't have enough unused words, adjust batch size
            if len(available_words) < current_batch_size:
                if len(available_words) == 0:
                    print(f"  🛑 No more candidate words available")
                    break
                current_batch_size = len(available_words)
                print(f"  🔄 Adjusting batch size to {current_batch_size} (remaining words)")
            
            batch = self.perturbation_manager.create_multi_word_batch_with_best_selection(
                available_words, current_batch_size,
                classifier_fn=self.classifier, current_text=current_text, original_score=original_score,
                similarity_fn=lambda orig, test: self.similarity_check(orig, test), original_text=text
            )
            if not batch:
                print(f"  🛑 No valid perturbations generated")
                break
                
            # Track which words we used
            for pert in batch:
                for i, (word_idx, sentence_idx, word, score) in enumerate(important_words):
                    if pert.original_word == word and pert.word_index == word_idx:
                        used_word_indices.add(i)
                        break
                
            # Apply batch to current accumulated text
            adversarial_text = self.perturbation_manager.apply_perturbations_to_text(current_text, batch)
            total_perturbations += len(batch)
            
            # Note: We don't stop on similarity violation to match 1-by-1 attack behavior
            # The similarity check is used implicitly through perturbation selection
            
            new_label, new_score = self.classifier(adversarial_text)
            num_reqs += 1
            
            if new_label != original_label:
                print(f"  ✅ Success! Flipped {original_label} → {new_label}")
                return adversarial_text, original_label, num_reqs, total_perturbations, True
            
            print(f"  ⚪ No flip ({new_label}, conf: {new_score:.3f})")
            
            # Only keep the perturbations if they maintain similarity (match 1-by-1 behavior)
            if self.similarity_check(text, adversarial_text):
                # Keep the adversarial text for next iteration (accumulate perturbations)
                current_text = adversarial_text
                print(f"  ✅ Similarity maintained, keeping perturbations")
            else:
                similarity_score = SequenceMatcher(None, text, adversarial_text).ratio()
                print(f"  ❌ Similarity violated ({similarity_score:.3f}), discarding batch")
            
            # Adaptive strategy: DYNAMIC adjustments based on confidence level
            remaining_words = len(important_words) - len(used_word_indices)
            max_possible_batch = min(self.batch_size, remaining_words)
            
            if new_score >= 0.8:  # Very high confidence - aggressive increase
                increase = min(2, max_possible_batch - current_batch_size)
                current_batch_size = min(max_possible_batch, current_batch_size + increase)
                print(f"  🚀 Very high confidence ({new_score:.3f}), aggressively increasing batch by {increase} to {current_batch_size}")
            elif new_score >= 0.5:  # High confidence - moderate increase
                current_batch_size = min(max_possible_batch, current_batch_size + 1)
                print(f"  🔼 High confidence ({new_score:.3f}), increasing batch size to {current_batch_size}")
            elif new_score < 0.3:  # Very low confidence - aggressive decrease
                decrease = min(2, current_batch_size - 1)
                current_batch_size = max(1, current_batch_size - decrease)
                print(f"  🎯 Very low confidence ({new_score:.3f}), aggressively reducing batch by {decrease} to {current_batch_size}")
            elif new_score < 0.3:  # Low confidence - moderate decrease
                current_batch_size = max(1, current_batch_size - 1)
                print(f"  🔽 Low confidence ({new_score:.3f}), reducing batch size to {current_batch_size}")
            else:  # Medium confidence (0.6-0.8)
                print(f"  ➡️ Medium confidence ({new_score:.3f}), keeping batch size {current_batch_size}")
        
        return current_text, original_label, num_reqs, total_perturbations, False