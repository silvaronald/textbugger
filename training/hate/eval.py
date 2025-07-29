import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(".."))

from pipeline import TextClassifierPipeline

tcp = TextClassifierPipeline("hate")

print(tcp.evaluate("lr"))
print(tcp.evaluate("lstm"))
print(tcp.evaluate("cnn"))