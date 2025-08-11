# Nome do ambiente virtual
VENV_NAME = .venv

# 1 Criar ambiente virtual e instalar dependências
setup:
	python3 -m venv $(VENV_NAME)
	. $(VENV_NAME)/bin/activate && pip install --upgrade pip
	. $(VENV_NAME)/bin/activate && pip install -r requirements.txt

# 2 Baixar modelos spaCy e NLTK
download-models:
	. $(VENV_NAME)/bin/activate && python -m spacy download en_core_web_sm
	. $(VENV_NAME)/bin/activate && python -c "import nltk; nltk.download('wordnet')"
	. $(VENV_NAME)/bin/activate && python -c "import nltk; nltk.download('averaged_perceptron_tagger')"

# 4 Configuração completa em um único comando
all: setup download-models

# 5 Ativar ambiente virtual manualmente
activate:
	@echo "Para ativar o ambiente virtual, execute:"
	@echo "source $(VENV_NAME)/bin/activate"
