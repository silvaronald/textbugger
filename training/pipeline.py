import numpy as np
import optuna
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import pandas as pd
import os
import sys
import joblib
from tensorflow.keras.models import load_model

class TextClassifierPipeline:
    def __init__(self, folder, max_len=100, embedding_dim=300):
         # Get the directory where this file is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        prefix = os.path.join(base_dir, "..", "datasets", folder)
        
        self.X_train_pad = np.load(os.path.join(prefix, "X_train_pad.npy"))
        self.X_val_pad = np.load(os.path.join(prefix, "X_val_pad.npy"))
        self.X_test_pad = np.load(os.path.join(prefix, "X_test_pad.npy"))
        
        self.y_train = pd.read_csv(os.path.join(prefix, "y_train.csv"))["label"]
        self.y_val = pd.read_csv(os.path.join(prefix, "y_val.csv"))["label"]
        self.y_test = pd.read_csv(os.path.join(prefix, "y_test.csv"))["label"]

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

    def train(self, model_type="lr", trials=100):
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
            study.optimize(objective, n_trials=trials)

            print("\nBest trial:")
            print(study.best_trial)

            best_params = study.best_params
            model = LogisticRegression(**best_params, max_iter=500)
            model.fit(X_train_flatten, self.y_train)

        elif model_type == "cnn":
            print("Training CNN...")
            def build_model(hp):
                input_layer = layers.Input(shape=(self.X_train_pad.shape[1],))

                # Embedding Layer: static or trainable
                embedding_layer = layers.Embedding(
                    input_dim=self.vocab_size,
                    output_dim=self.embedding_dim,
                    weights=[self.embedding_matrix],
                    input_length=self.max_len,
                    trainable=False
                )(input_layer)

                # Filter sizes from Kim's paper
                filter_sizes = [3, 4, 5]
                pooled_outputs = []

                for fsz in filter_sizes:
                    conv = layers.Conv1D(
                        filters=hp.Int("num_filters", min_value=100, max_value=300, step=50),
                        kernel_size=fsz,
                        activation="relu"
                    )(embedding_layer)
                    pooled = layers.GlobalMaxPooling1D()(conv)
                    pooled_outputs.append(pooled)

                merged = layers.Concatenate()(pooled_outputs)

                dropout = layers.Dropout(rate=hp.Float("dropout_rate", 0.3, 0.6, step=0.1))(merged)
                output = layers.Dense(1, activation="sigmoid")(dropout)

                model = models.Model(inputs=input_layer, outputs=output)

                optimizer_name = hp.Choice("optimizer", ["adam", "rmsprop", "nadam"])
                lr = hp.Choice("learning_rate", [1e-4, 1e-3, 5e-3])
                optimizer = {"adam": optimizers.Adam(lr), "rmsprop": optimizers.RMSprop(lr), "nadam": optimizers.Nadam(lr)}[optimizer_name]

                model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
                return model

            tuner = kt.RandomSearch(
                build_model,
                objective="val_accuracy",
                max_trials=trials,
                executions_per_trial=1,
                overwrite=True,
                directory="cnn_tuning",
                project_name="kim2014_cnn"
            )

            early_stop = EarlyStopping(patience=8, restore_best_weights=True)

            tuner.search(
                self.X_train_pad, self.y_train,
                validation_data=(self.X_val_pad, self.y_val),
                epochs=15,
                batch_size=64,
                callbacks=[early_stop]
            )

            model = tuner.get_best_models(1)[0]

        elif model_type == "lstm":
            print("Training LSTM...")
            def build_model(hp):
                model = models.Sequential()

                model.add(layers.Embedding(input_dim=self.vocab_size,
                                    output_dim=self.embedding_dim,
                                    weights=[self.embedding_matrix],
                                    input_length=self.max_len,
                                    trainable=False))

                if hp.Boolean("bidirectional"):
                    model.add(layers.Bidirectional(layers.LSTM(
                        hp.Int("lstm_units", min_value=64, max_value=256, step=64),
                        dropout=hp.Float("dropout", 0.2, 0.5, step=0.1),
                        recurrent_dropout=hp.Float("recurrent_dropout", 0.2, 0.5, step=0.1)
                    )))
                else:
                    model.add(layers.LSTM(
                        hp.Int("lstm_units", min_value=64, max_value=256, step=64),
                        dropout=hp.Float("dropout", 0.2, 0.5, step=0.1),
                        recurrent_dropout=hp.Float("recurrent_dropout", 0.2, 0.5, step=0.1)
                    ))

                model.add(layers.Dense(1, activation='sigmoid'))

                optimizer_choice = hp.Choice("optimizer", ["adam", "rmsprop", "nadam"])
                lr = hp.Choice("learning_rate", [1e-4, 1e-3, 5e-3])

                optimizer = {"adam": optimizers.Adam(lr), "rmsprop": optimizers.RMSprop(lr), "nadam": optimizers.Nadam(lr)}[optimizer_choice]

                model.compile(optimizer=optimizer,
                            loss='binary_crossentropy',
                            metrics=['accuracy'])
                return model

            # Tuner setup
            tuner = kt.RandomSearch(
                build_model,
                objective="val_accuracy",
                max_trials=trials,
                executions_per_trial=1,
                overwrite=True,
                directory="lstm_tuning",
                project_name="sentiment"
            )

            # Early stopping
            stop = EarlyStopping(patience=8, restore_best_weights=True)

            # Run search
            tuner.search(self.X_train_pad, self.y_train,
                        epochs=15,
                        batch_size=64,
                        validation_data=(self.X_val_pad, self.y_val),
                        callbacks=[stop])
            
            model = tuner.get_best_models(1)[0]

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

    def evaluate(self, model_type):
        if model_type == "lr":
            model = joblib.load("lr_model.joblib")
            return accuracy_score(self.y_test, model.predict(self.flatten_embeddings(self.X_test_pad)))
        elif model_type == "cnn":
            model = load_model("cnn_model.h5")
            return model.evaluate(self.X_test_pad, self.y_test)
        elif model_type == "lstm":
            model = load_model("lstm_model.h5")
            return model.evaluate(self.X_test_pad, self.y_test)
