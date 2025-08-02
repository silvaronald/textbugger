from blackbox import BlackBoxTextBugger

def dummy_classifier(text):
    # Classificador mock para teste
    if "awful" in text:
        return ("negative", {"negative": 0.9, "positive": 0.1})
    return ("positive", {"negative": 0.1, "positive": 0.9})

attacker = BlackBoxTextBugger(classifier_fn=dummy_classifier)
original = "This movie was awful and boring. I did not like it at all. The acting was terrible. I would not recommend it to anyone."
adversarial = attacker.attack(original)

print("Original:", original)
print("Adversarial:", adversarial)
