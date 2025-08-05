from mlass_api_clients import IBMWatsonClassifier
from blackbox import BlackBoxTextBugger

# Credenciais do IBM Watson (API Key e URL da instância)
IBM_API_KEY = "API_KEY_HERE"
IBM_URL = "https://api.us-east.natural-language-understanding.watson.cloud.ibm.com/instances/instance"

# Inicializa o classificador IBM Watson
ibm_clf = IBMWatsonClassifier(api_key=IBM_API_KEY, service_url=IBM_URL)

# Inicializa o atacante TextBugger usando o classificador IBM Watson
attacker = BlackBoxTextBugger(classifier_fn=ibm_clf.classify)

# Textos para teste
test_texts = [
    "This film was a masterpiece. The acting was incredibly compelling, and the storyline was both original and emotionally resonant. I was captivated from start to finish.",
    "I was so disappointed by this movie. The plot was predictable and riddled with holes, and the dialogue felt completely unnatural. It was a total waste of time.",
    "You are an absolute idiot for even suggesting that. It's the stupidest idea I've ever heard, and you should just stop talking.",
    "The report indicates that the project is currently on schedule. Several key milestones have been met, although budget constraints remain a topic of discussion for the next quarter."
]

# Loop pelos textos e aplicação de ataque
for text in test_texts:
    print("\n==============================")
    print(f"Texto Original: {text}")
    
    # Classificação inicial
    label, scores = ibm_clf.classify(text)
    print(f"Classificação Original -> Label: {label}, Scores: {scores}")
    
    # Executa o ataque adversarial
    adversarial_text = attacker.attack(text)
    
    # Reclassifica após perturbação
    adv_label, adv_scores = ibm_clf.classify(adversarial_text)
    print(f"\nTexto Adversarial: {adversarial_text}")
    print(f"Classificação Adversarial -> Label: {adv_label}, Scores: {adv_scores}")
