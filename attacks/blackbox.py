import random
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from whitebox import AdversarialAttack

class BlackBoxAttack(AdversarialAttack):
    """
    Black-box TextBugger attack implementation
    Extends the existing AdversarialAttack class for black-box scenarios
    """
    
    def __init__(self, folder):
        super().__init__(folder)
        # Load spaCy for sentence segmentation
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        # Character substitution map (inherited from parent class)
        self.sub_c_map = {'a': '@', 'o': '0', 'l': '1', 'e': '3', 'i': '1', 's': '$'}
        
    def textbugger_blackbox_attack(self, document_x, ground_truth_label_y, classifier_F, threshold_epsilon=0.8):
        """
        Black-box adversarial attack on text classification models
        
        Args:
            document_x: Input document/text (string)
            ground_truth_label_y: Original true label (int)
            classifier_F: Black-box classifier function
            threshold_epsilon: Semantic similarity threshold (float)
        
        Returns:
            adversarial_document (string) or None if attack fails
        """
        
        # Step 1: Initialize
        x_prime = document_x
        
        # Step 2: Find Important Sentences
        sentences = self.segment_into_sentences(document_x)
        if not sentences:
            sentences = [document_x]  # Fallback to single sentence
            
        sentence_importance = []
        
        for i, sentence_si in enumerate(sentences):
            if not sentence_si.strip():
                continue
                
            # Calculate sentence importance as confidence of predicted class
            try:
                confidence_score = self.get_classifier_confidence(sentence_si, ground_truth_label_y, classifier_F)
                sentence_importance.append((sentence_si, confidence_score, i))
            except Exception as e:
                print(f"Error processing sentence {i}: {e}")
                continue
        
        # Sort sentences by importance (descending)
        sentences_ordered = sorted(sentence_importance, key=lambda x: x[1], reverse=True)
        
        # Filter out sentences with different predicted labels
        sentences_filtered = []
        for sentence_si, score, idx in sentences_ordered:
            try:
                predicted_label = self.get_classifier_prediction(sentence_si, classifier_F)
                if predicted_label == ground_truth_label_y:
                    sentences_filtered.append((sentence_si, score, idx))
            except Exception as e:
                print(f"Error filtering sentence {idx}: {e}")
                continue
        
        # Step 3: Process Each Important Sentence
        for sentence_si, _, _ in sentences_filtered:
            
            # Step 3a: Find Important Words in Current Sentence
            words = sentence_si.split()
            if not words:
                continue
                
            word_importance = []
            
            for j, word_wj in enumerate(words):
                if len(word_wj.strip()) == 0:
                    continue
                    
                # Create sentence without word j
                words_without_j = words[:j] + words[j+1:]
                modified_sentence = " ".join(words_without_j)
                
                # Calculate word importance using removal impact
                try:
                    original_confidence = self.get_classifier_confidence(sentence_si, ground_truth_label_y, classifier_F)
                    modified_confidence = self.get_classifier_confidence(modified_sentence, ground_truth_label_y, classifier_F)
                    
                    importance_score = original_confidence - modified_confidence
                    word_importance.append((word_wj, importance_score, j))
                except Exception as e:
                    print(f"Error calculating importance for word '{word_wj}': {e}")
                    continue
            
            # Sort words by importance (descending)
            words_ordered = sorted(word_importance, key=lambda x: x[1], reverse=True)
            
            # Step 3b: Apply Perturbations to Important Words
            for word_wj, importance, position in words_ordered:
                if importance < 0.01:  # Skip words with very low importance
                    continue
                    
                # Generate bugs for current word
                best_bug = self.select_bug(word_wj, x_prime, ground_truth_label_y, classifier_F)
                
                if best_bug is not None and best_bug != word_wj:
                    # Replace word with best bug
                    x_prime_candidate = self.replace_word_in_text(x_prime, word_wj, best_bug)
                    
                    # Check semantic similarity constraint
                    similarity = self.calculate_semantic_similarity(document_x, x_prime_candidate)
                    if similarity <= threshold_epsilon:
                        continue  # Similarity constraint violated, try next word
                    
                    # Check if attack succeeded
                    try:
                        new_predicted_label = self.get_classifier_prediction(x_prime_candidate, classifier_F)
                        if new_predicted_label != ground_truth_label_y:
                            return x_prime_candidate  # Attack successful!
                        else:
                            # Update x_prime even if attack didn't succeed (for iterative improvement)
                            x_prime = x_prime_candidate
                    except Exception as e:
                        print(f"Error testing adversarial candidate: {e}")
                        continue
        
        return None  # Attack failed

    def select_bug(self, word, current_text, original_label, classifier_F):
        """
        Select the best bug from generated perturbations
        
        Args:
            word: Word to perturb
            current_text: Current text state
            original_label: Original predicted label
            classifier_F: Classifier function
            
        Returns:
            best_bug (string) or None
        """
        # Generate 5 types of bugs
        bugs = self.generate_bugs(word)
        
        if not bugs:
            return None
            
        best_bug = None
        best_score_change = 0
        
        try:
            original_confidence = self.get_classifier_confidence(current_text, original_label, classifier_F)
        except Exception as e:
            print(f"Error getting original confidence: {e}")
            return None
        
        for bug in bugs:
            try:
                # Create candidate text with this bug
                candidate_text = self.replace_word_in_text(current_text, word, bug)
                
                # Calculate score change
                modified_confidence = self.get_classifier_confidence(candidate_text, original_label, classifier_F)
                score_change = original_confidence - modified_confidence
                
                if score_change > best_score_change:
                    best_bug = bug
                    best_score_change = score_change
                    
            except Exception as e:
                print(f"Error testing bug '{bug}': {e}")
                continue
        
        return best_bug

    def generate_bugs(self, word):
        """
        Generate 5 types of character-level and word-level perturbations
        
        Args:
            word: Input word to perturb
            
        Returns:
            List of perturbed words
        """
        bugs = []
        
        # 1. Insert: Add space within word (for words < 6 characters)
        if len(word) < 6 and len(word) > 1:
            pos = random.randint(1, len(word)-1)
            bugs.append(word[:pos] + " " + word[pos:])
        
        # 2. Delete: Remove random character (not first/last)
        if len(word) > 2:
            pos = random.randint(1, len(word)-2)
            bugs.append(word[:pos] + word[pos+1:])
        
        # 3. Swap: Swap adjacent characters (for words > 4 letters)
        if len(word) > 4:
            pos = random.randint(1, len(word)-3)
            chars = list(word)
            chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
            bugs.append(''.join(chars))
        
        # 4. Substitute-C: Replace with visually similar characters
        for i, char in enumerate(word.lower()):
            if char in self.sub_c_map:
                bug_chars = list(word)
                bug_chars[i] = self.sub_c_map[char]
                bugs.append(''.join(bug_chars))
                break  # Only one substitution per word
        
        # 5. Substitute-W: Replace with semantically similar word
        similar_words = self.get_similar_words(word, top_k=3)
        if similar_words:
            bugs.extend(similar_words)
        
        return [bug for bug in bugs if bug != word]  # Remove unchanged words

    # Helper Functions
    def segment_into_sentences(self, text):
        """Use spaCy to segment text into sentences"""
        if self.nlp is None:
            # Fallback to simple sentence splitting
            return [s.strip() for s in text.split('.') if s.strip()]
        
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def replace_word_in_text(self, text, old_word, new_word):
        """Replace first occurrence of old_word with new_word"""
        return text.replace(old_word, new_word, 1)

    def calculate_semantic_similarity(self, text1, text2):
        """
        Calculate semantic similarity using simple string similarity
        TODO: Implement Universal Sentence Encoder similarity
        """
        return SequenceMatcher(None, text1, text2).ratio()

    def get_classifier_confidence(self, text, label, classifier_F):
        """
        Get classifier confidence for a specific label
        
        Args:
            text: Input text
            label: Label to get confidence for
            classifier_F: Classifier function
            
        Returns:
            Confidence score (float)
        """
        # This method needs to be adapted based on your classifier interface
        # For now, assuming classifier_F returns probabilities
        try:
            if hasattr(classifier_F, 'predict_proba'):
                # sklearn-style classifier
                probs = classifier_F.predict_proba([text])[0]
                return probs[label] if label < len(probs) else 0.0
            else:
                # Custom classifier - assume it returns prediction confidence
                return classifier_F(text, return_confidence=True)
        except Exception as e:
            print(f"Error getting classifier confidence: {e}")
            return 0.0

    def get_classifier_prediction(self, text, classifier_F):
        """
        Get classifier prediction
        
        Args:
            text: Input text
            classifier_F: Classifier function
            
        Returns:
            Predicted label (int)
        """
        try:
            if hasattr(classifier_F, 'predict'):
                # sklearn-style classifier
                return classifier_F.predict([text])[0]
            else:
                # Custom classifier
                return classifier_F(text, return_label=True)
        except Exception as e:
            print(f"Error getting classifier prediction: {e}")
            return -1

    def create_classifier_wrapper(self, model_type="lstm"):
        """
        Create a wrapper function for existing models to work with black-box attack
        
        Args:
            model_type: Type of model ("lr", "cnn", "lstm")
            
        Returns:
            Classifier function compatible with black-box attack
        """
        def classifier_wrapper(text, return_confidence=False, return_label=False):
            try:
                # Tokenize and pad the input
                tokens = self.tokenizer.texts_to_sequences([text])
                padded = np.array(tokens)
                
                if len(padded) == 0 or len(padded[0]) == 0:
                    return 0 if return_label else 0.0
                    
                # Pad to max_len
                import tensorflow as tf
                padded = tf.keras.preprocessing.sequence.pad_sequences(
                    padded, maxlen=self.max_len, padding='post', truncating='post'
                )[0]
                
                if model_type == "lr":
                    X_flat = self.flatten_embeddings(padded, self.embedding_matrix)
                    pred_label = int(self.model_lr.predict(X_flat)[0])
                    pred_proba = self.model_lr.predict_proba(X_flat)[0]
                    
                    if return_label:
                        return pred_label
                    elif return_confidence:
                        return pred_proba[pred_label]
                    else:
                        return pred_proba
                        
                else:
                    model = self.model_cnn if model_type == "cnn" else self.model_lstm
                    pred = model.predict(np.array([padded]))
                    pred_label = int((pred > 0.5).astype("int32")[0][0])
                    confidence = float(pred[0][0]) if pred_label == 1 else 1 - float(pred[0][0])
                    
                    if return_label:
                        return pred_label
                    elif return_confidence:
                        return confidence
                    else:
                        return [1-confidence, confidence] if pred_label == 0 else [confidence, 1-confidence]
                        
            except Exception as e:
                print(f"Error in classifier wrapper: {e}")
                return 0 if return_label else 0.0
                
        return classifier_wrapper

    def create_fasttext_classifier_wrapper(self, model_path="amazon_review_polarity.bin"):
        """
        Create a wrapper function for FastText model to work with black-box attack
        
        Args:
            model_path: Path to FastText model file
            
        Returns:
            Classifier function compatible with black-box attack
        """
        import fasttext
        import os
        
        # Load FastText model
        try:
            # Try relative path first
            if not os.path.exists(model_path):
                model_path = os.path.join("..", "black-box", model_path)
            if not os.path.exists(model_path):
                model_path = os.path.join("..", model_path)
                
            self.fasttext_model = fasttext.load_model(model_path)
            print(f"✅ FastText model loaded from: {model_path}")
        except Exception as e:
            print(f"❌ Error loading FastText model: {e}")
            return None
        
        def fasttext_classifier_wrapper(text, return_confidence=False, return_label=False):
            try:
                # Clean text for FastText
                cleaned_text = text.strip().replace('\n', ' ')
                if not cleaned_text:
                    return 0 if return_label else 0.0
                
                # Get FastText prediction
                predictions = self.fasttext_model.predict(cleaned_text)
                labels = predictions[0]  # List of predicted labels
                probabilities = predictions[1]  # List of probabilities
                
                if not labels or not probabilities:
                    return 0 if return_label else 0.0
                
                # Parse label (assuming format like '__label__1', '__label__2')
                predicted_label_str = labels[0]
                if '__label__' in predicted_label_str:
                    predicted_label = int(predicted_label_str.replace('__label__', '')) - 1  # Convert to 0-indexed
                else:
                    predicted_label = 0
                    
                confidence = float(probabilities[0])
                
                if return_label:
                    return predicted_label
                elif return_confidence:
                    return confidence
                else:
                    # Return probability distribution [neg_prob, pos_prob]
                    if predicted_label == 0:  # Negative prediction
                        return [confidence, 1.0 - confidence]
                    else:  # Positive prediction  
                        return [1.0 - confidence, confidence]
                        
            except Exception as e:
                print(f"Error in FastText classifier wrapper: {e}")
                return 0 if return_label else 0.0
                
        return fasttext_classifier_wrapper
    
    def create_google_nlp_classifier_wrapper(self):
        """
        Create a wrapper function for Google Cloud Natural Language API
        
        Returns:
            Classifier function compatible with black-box attack
        """
        try:
            from google.cloud import language_v2
            self.google_client = language_v2.LanguageServiceClient()
            print("✅ Google Cloud NLP client initialized")
        except Exception as e:
            print(f"❌ Error initializing Google Cloud NLP: {e}")
            print("Make sure to:")
            print("1. Install: pip install google-cloud-language")
            print("2. Set up authentication: gcloud auth application-default login")
            print("3. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            return None
        
        def google_nlp_classifier_wrapper(text, return_confidence=False, return_label=False):
            try:
                # Clean text for API
                cleaned_text = text.strip()
                if not cleaned_text:
                    return 0 if return_label else 0.0
                
                # Prepare document for Google NLP API
                document = {
                    "content": cleaned_text,
                    "type_": language_v2.Document.Type.PLAIN_TEXT,
                    "language_code": "en",
                }
                
                # Call sentiment analysis
                response = self.google_client.analyze_sentiment(
                    request={"document": document, "encoding_type": language_v2.EncodingType.UTF8}
                )
                
                # Extract sentiment score (-1 to +1, where -1 is negative, +1 is positive)
                sentiment_score = response.document_sentiment.score
                magnitude = response.document_sentiment.magnitude
                
                # Convert to binary classification
                # Negative: score < 0, Positive: score >= 0
                predicted_label = 1 if sentiment_score >= 0 else 0
                
                # Convert score to confidence (0 to 1)
                # Use magnitude as confidence indicator
                confidence = min(abs(sentiment_score) + 0.1, 1.0)  # Ensure minimum confidence
                
                if return_label:
                    return predicted_label
                elif return_confidence:
                    return confidence
                else:
                    # Return probability distribution [neg_prob, pos_prob]
                    if predicted_label == 0:  # Negative prediction
                        return [confidence, 1.0 - confidence]
                    else:  # Positive prediction  
                        return [1.0 - confidence, confidence]
                        
            except Exception as e:
                print(f"Error in Google NLP classifier wrapper: {e}")
                return 0 if return_label else 0.0
                
        return google_nlp_classifier_wrapper
    
    def build_word_embedding_dict(self, max_words=10000):
        """
        Build a dictionary of word embeddings from the tokenizer and embedding matrix
        
        Args:
            max_words: Maximum number of words to include
            
        Returns:
            Dictionary mapping words to their embedding vectors
        """
        if not hasattr(self, '_word_embedding_dict'):
            print("Building word embedding dictionary...")
            self._word_embedding_dict = {}
            
            # Get word index from tokenizer
            word_index = getattr(self.tokenizer, 'word_index', {})
            
            # Limit to most common words for efficiency
            words_to_process = min(len(word_index), max_words)
            
            for word, idx in list(word_index.items())[:words_to_process]:
                if idx < self.embedding_matrix.shape[0]:
                    # Get embedding vector for this word
                    embedding_vector = self.embedding_matrix[idx]
                    # Only include if not all zeros (valid embedding)
                    if np.any(embedding_vector):
                        self._word_embedding_dict[word] = embedding_vector
            
            print(f"Built embedding dictionary with {len(self._word_embedding_dict)} words")
            
        return self._word_embedding_dict
    
    def get_similar_words(self, word, top_k=5):
        """
        Get semantically similar words using cosine similarity of embeddings
        
        Args:
            word: Target word to find similar words for
            top_k: Number of similar words to return
            
        Returns:
            List of similar words
        """
        try:
            # Build embedding dictionary if not exists
            embedding_dict = self.build_word_embedding_dict()
            
            # Check if word exists in our embeddings
            if word.lower() not in embedding_dict:
                return []
            
            # Get embedding for target word
            target_embedding = embedding_dict[word.lower()]
            target_embedding = target_embedding.reshape(1, -1)
            
            # Calculate similarities with all other words
            similarities = []
            for candidate_word, candidate_embedding in embedding_dict.items():
                if candidate_word != word.lower():
                    candidate_embedding = candidate_embedding.reshape(1, -1)
                    similarity = cosine_similarity(target_embedding, candidate_embedding)[0][0]
                    similarities.append((candidate_word, similarity))
            
            # Sort by similarity and return top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            similar_words = [word_sim[0] for word_sim in similarities[:top_k]]
            
            return similar_words
            
        except Exception as e:
            print(f"Error finding similar words for '{word}': {e}")
            return []