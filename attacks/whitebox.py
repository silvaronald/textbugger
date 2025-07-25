import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow as tf
from difflib import SequenceMatcher
import os
import pickle 
import joblib
from tensorflow.keras.models import load_model

class AdversarialAttack():
    def __init__(self, folder):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prefix_datasets = os.path.join(base_dir, "..", "datasets", folder)
        prefix_training = os.path.join(base_dir, "..", "training", folder)
                              
        with open(os.path.join(prefix_datasets, "tokenizer.pkl"), "rb") as f:
            self.tokenizer = pickle.load(f)
        self.embedding_matrix = np.load(os.path.join(prefix_datasets, "embedding_matrix.npy"))

        self.model_lr = joblib.load(os.path.join(prefix_training, "lr_model.joblib"))
        self.model_cnn = load_model(os.path.join(prefix_training, "cnn_model.h5"))
        self.model_lstm = load_model(os.path.join(prefix_training, "lstm_model.h5"))

        self.sub_c_map = {'a': '@', 'o': '0', 'l': '1', 'e': '3', 'i': '1', 's': '$'}
        self.bug_functions = [self.insert_bug, self.delete_bug, self.swap_bug, self.substitute_c]
        self.max_len = 100

    # ------------------- STEP 1: WORD IMPORTANCE -------------------
    def get_word_importance(self, X_padded, model_type):
        """
        Compute word importance scores for a single padded input.

        Parameters:
        - model: trained model (LR, CNN, or LSTM)
        - X_padded: 1D array of token IDs, shape (max_len,)
        - model_type: one of "lr", "cnn", "lstm"
        - embedding_matrix: shape (vocab_size, embedding_dim)

        Returns:
        - importance: 1D numpy array of shape (max_len,) with importance scores
        """
        print(X_padded)
        max_len = len(X_padded)
        embedding_dim = self.embedding_matrix.shape[1]

        # Step 1: Embed input tokens
        embedded = self.embedding_matrix[X_padded]  # shape: (max_len, embedding_dim)
        
        if model_type == "lr":
            # Flatten to (max_len * embedding_dim,)
            X_flat = embedded.flatten()  # shape: (max_len * embedding_dim,)
            coefs = self.model_lr.coef_.reshape(-1, max_len * embedding_dim)
            
            # Reshape both to (max_len, embedding_dim)
            coefs_2d = coefs.reshape(max_len, embedding_dim)
            input_2d = X_flat.reshape(max_len, embedding_dim)
            
            # Importance per word = norm of (coef * word_embedding)
            importance = np.linalg.norm(coefs_2d * input_2d, axis=1)

        elif model_type in ["cnn", "lstm"]:
            model = self.model_cnn if model_type == "cnn" else self.model_lstm
            X_tensor = tf.convert_to_tensor(X_padded.reshape(1, -1), dtype=tf.int32)

            embedding_layer = model.layers[0]
            with tf.GradientTape() as tape:
                embedded = embedding_layer(X_tensor)  # shape: (1, max_len, embedding_dim)
                tape.watch(embedded)

                x = embedded
                for layer in model.layers[1:]:
                    x = layer(x, training=False)  # Manually forward pass
                output = x  # Final output

            grads = tape.gradient(output, embedded)
            if grads is None:
                raise RuntimeError("Gradients could not be computed. Ensure the output depends on the embedding.")

            grads = grads.numpy()[0]  # shape: (max_len, embedding_dim)
            importance = np.linalg.norm(grads, axis=1)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        return importance


    # ------------------- BUG FUNCTIONS -------------------
    def insert_bug(self, word):
        if len(word) >= 3:
            pos = random.randint(1, len(word) - 1)
            return word[:pos] + ' ' + word[pos:]
        return word

    def delete_bug(self, word):
        if len(word) > 3:
            pos = random.randint(1, len(word) - 2)
            return word[:pos] + word[pos+1:]
        return word

    def swap_bug(self, word):
        if len(word) > 4:
            pos = random.randint(1, len(word) - 3)
            return word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
        return word

    def substitute_c(self, word):
        for i, ch in enumerate(word):
            if ch in self.sub_c_map:
                return word[:i] + self.sub_c_map[ch] + word[i+1:]
        return word

    def substitute_w(self, word, embedding_dict, top_k=5):
        if word not in embedding_dict:
            return word
        original_vec = embedding_dict[word].reshape(1, -1)
        candidates = [(w, cosine_similarity(original_vec, v.reshape(1, -1))[0][0])
                      for w, v in embedding_dict.items() if w != word]
        candidates = sorted(candidates, key=lambda x: -x[1])
        return candidates[0][0] if candidates else word

    # ------------------- SIMILARITY -------------------
    def similarity(self, x, x_adv):
        return SequenceMatcher(None, x, x_adv).ratio()

    # ------------------- MAIN ATTACK -------------------
    def generate_adversarial(self, text, model_type="lstm", embedding_dict=None, epsilon=0.8):
        tokens = self.tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen=self.max_len, padding='post', truncating='post')[0]
        
        # Step 1: Get original prediction and probability
        if model_type == "lr":
            X_flat = self.flatten_embeddings(padded, self.embedding_matrix)
            original_label = int(self.model_lr.predict(X_flat)[0])
            original_proba = self.model_lr.predict_proba(X_flat)[0][original_label]
        else:
            model = self.model_cnn if model_type == "cnn" else self.model_lstm
            pred = model.predict(np.array([padded]))
            original_label = int((pred > 0.5).astype("int32")[0][0])
            original_proba = float(pred[0][0]) if original_label == 1 else 1 - float(pred[0][0])

        # Step 2: Get importance scores
        importance_scores = self.get_word_importance(padded, model_type)
        words = text.split()
        sorted_indices = np.argsort(-importance_scores)  # Most important first

        x_adv_words = words.copy()

        print(importance_scores)
        print(sorted_indices, len(words))
        for idx in sorted_indices:
            if idx >= len(words) or importance_scores[idx] == 0:
                continue

            original_word = x_adv_words[idx]
            bugged_versions = [bug_fn(original_word) for bug_fn in self.bug_functions]
            if embedding_dict:
                bugged_versions.append(self.substitute_w(original_word, embedding_dict))

            best_bug = None
            best_conf_drop = 0

            for bug in bugged_versions:
                temp_words = x_adv_words.copy()
                temp_words[idx] = bug
                x_candidate = ' '.join(temp_words)

                # Check similarity
                sim = self.similarity(text, x_candidate)
                if sim <= epsilon:
                    continue

                # Predict new label and prob
                candidate_tokens = self.tokenizer.texts_to_sequences([x_candidate])
                candidate_padded = tf.keras.preprocessing.sequence.pad_sequences(candidate_tokens, maxlen=self.max_len, padding='post', truncating='post')[0]

                if model_type == "lr":
                    X_flat = self.flatten_embeddings(candidate_padded, self.embedding_matrix)
                    new_pred = int(self.model_lr.predict(X_flat)[0])
                    new_proba = self.model_lr.predict_proba(X_flat)[0][original_label]
                else:
                    model = self.model_cnn if model_type == "cnn" else self.model_lstm
                    pred = model.predict(np.array([candidate_padded]))
                    new_pred = int((pred > 0.5).astype("int32")[0][0])
                    new_proba = float(pred[0][0]) if original_label == 1 else 1 - float(pred[0][0])

                # If label flipped, stop immediately
                if new_pred != original_label:
                    temp_words[idx] = bug  # Apply it permanently
                    return ' '.join(temp_words)
                
                # Otherwise, keep best bug based on confidence drop
                conf_drop = original_proba - new_proba
                if conf_drop > best_conf_drop:
                    best_conf_drop = conf_drop
                    best_bug = bug

            # Apply the most effective bug (even if it didn't flip the label)
            if best_bug:
                x_adv_words[idx] = best_bug

            print(' '.join(x_adv_words))

        return None  # No successful adversarial attack found

    def flatten_embeddings(self, seq, embedding_matrix):
        embedded = np.array([embedding_matrix[idx] for idx in seq])
        return embedded.flatten().reshape(1, -1)
