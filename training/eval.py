from pipeline import TextClassifierPipeline

tcp = TextClassifierPipeline("kaggle")

print(tcp.evaluate("lr"))
# print(tcp.evaluate("lstm"))
# print(tcp.evaluate("cnn"))