import numpy as np
import optuna
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import os
import sys

class TextClassifierPipeline:
    def __init__(self, folder, max_len=100, embedding_dim=300):
         # Get the directory where this file is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        prefix = os.path.join(base_dir, "..", "datasets", folder)  # relative to this file
        
        self.X_train_pad = np.load(os.path.join(prefix, "X_train_pad.npy"))
        self.X_val_pad = np.load(os.path.join(prefix, "X_val_pad.npy"))
        self.X_test_pad = np.load(os.path.join(prefix, "X_test_pad.npy"))
        
        self.y_train = pd.read_csv(os.path.join(prefix, "y_train.csv"))["target"]
        self.y_val = pd.read_csv(os.path.join(prefix, "y_val.csv"))["target"]
        self.y_test = pd.read_csv(os.path.join(prefix, "y_test.csv"))["target"]

        with open(os.path.join(prefix, "tokenizer.pkl"), "rb") as f:
            self.tokenizer = pickle.load(f)
        self.embedding_matrix = np.load(os.path.join(prefix, "embedding_matrix.npy"))
        
        self.vocab_size = self.embedding_matrix.shape[0]
        self.embedding_dim = embedding_dim
        self.max_len = max_len

        self.models = {}

    def flatten_embeddings(self, sequences):
        # sequences: shape (n_samples, max_len)
        # output: shape (n_samples, max_len * embedding_dim)
        embedded = np.array([
            [self.embedding_matrix[idx] for idx in seq]
            for seq in sequences
        ])  # shape: (n_samples, max_len, embedding_dim)
        
        return embedded.reshape(len(sequences), -1)  # flatten to (n_samples, max_len * embedding_dim)

    def train(self, model_type="lr", epochs=5, batch_size=64):
        if model_type == "lr":
            print("Training Logistic Regression...")
            X_train_flatten = self.flatten_embeddings(self.X_train_pad)
            X_val_flatten = self.flatten_embeddings(self.X_val_pad)

            def objective(trial):
                C = trial.suggest_loguniform('C', 1e-4, 10.0)
                solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])

                clf = LogisticRegression(C=C, solver=solver, max_iter=500)
                clf.fit(X_train_flatten[:5], self.y_train[:5])
                y_pred = clf.predict(X_val_flatten)
                return accuracy_score(self.y_val, y_pred)

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=5)

            print("\nBest trial:")
            print(study.best_trial)

            best_params = study.best_params
            model = LogisticRegression(**best_params, max_iter=500)
            model.fit(X_train_flatten[:5], self.y_train[:5])

        elif model_type == "cnn":
            print("Training CNN...")
            model = models.Sequential([
                layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim,
                                 weights=[self.embedding_matrix], input_length=self.max_len,
                                 trainable=False),
                layers.Conv1D(128, 5, activation='relu'),
                layers.GlobalMaxPooling1D(),
                layers.Dense(10, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(self.X_train_pad[:5], self.y_train[:5], epochs=epochs,
                      validation_data=(self.X_val_pad, self.y_val), batch_size=batch_size)

        elif model_type == "lstm":
            print("Training LSTM...")
            model = models.Sequential([
                layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim,
                                 weights=[self.embedding_matrix], input_length=self.max_len,
                                 trainable=False),
                layers.LSTM(128),
                layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(self.X_train_pad[:5], self.y_train[:5], epochs=epochs,
                      validation_data=(self.X_val_pad, self.y_val), batch_size=batch_size)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.models[model_type] = model
        return model

    def save_model(self, model_type, path):
        model = self.models.get(model_type)
        if model is None:
            raise ValueError(f"No trained model found for type: {model_type}")

        if model_type == "lr":
            import joblib
            joblib.dump(model, path)
        else:
            model.save(path)

    def predict(self, model_type, X_input):
        model = self.models.get(model_type)
        if model is None:
            raise ValueError(f"No trained model found for type: {model_type}")

        if model_type == "lr":
            X_input_avg = self.average_embeddings(X_input)
            return model.predict(X_input_avg)
        else:
            return (model.predict(X_input) > 0.5).astype("int32")
