import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(".."))

from pipeline import TextClassifierPipeline

FOLDER = "rtmr"
pp = TextClassifierPipeline(FOLDER)

pp.train("lr")
pp.train("lstm")
pp.train("cnn")

pp.save_model("lr", "lr_model.joblib")
pp.save_model("cnn", "cnn_model.h5")
pp.save_model("lstm", "lstm_model.h5")