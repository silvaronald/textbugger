from pipeline import TextClassifierPipeline

FOLDER = "kaggle"
pp = TextClassifierPipeline(FOLDER)

pp.train("lr", 1)
pp.train("lstm", 100)
pp.train("cnn", 100)

pp.save_model("lr", f"{FOLDER}/lr_model.joblib")
pp.save_model("cnn", f"{FOLDER}/cnn_model.h5")
pp.save_model("lstm", f"{FOLDER}/lstm_model.h5")