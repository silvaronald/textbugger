import urllib.request
import os

model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/amazon_review_polarity.bin"
model_path = "../amazon_review_polarity.bin"

if os.path.exists(model_path):
    print(f"O modelo já está baixado em {model_path}.")
else:
    try:
        print("Baixando o modelo Amazon Review Polarity...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Modelo baixado com sucesso em {model_path}")
    except Exception as e:
        print(f"Falha ao baixar o modelo: {e}")
        print("Você pode precisar encontrar outra fonte ou treinar seu próprio modelo")