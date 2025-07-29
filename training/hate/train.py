import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(".."))

from pipeline import TextClassifierPipeline

FOLDER = "hate"
pp = TextClassifierPipeline(FOLDER)

pp.train("lr", 100)
pp.train("lstm", 100)
pp.train("cnn", 100)

pp.save_model("lr", "lr_model.joblib")
pp.save_model("cnn", "cnn_model.h5")
pp.save_model("lstm", "lstm_model.h5")