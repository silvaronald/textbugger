from whitebox import AdversarialAttack
import numpy as np
import pandas as pd 

attacker = AdversarialAttack("rtmr")

#X_test_pad = np.load("../datasets/rtmr/X_test_pad.npy")
#print(X_test_pad[0])

X_test = pd.read_csv("../datasets/rtmr/X_test.csv")["text"]
print(X_test[0])

print(attacker.generate_adversarial("era uma casa muito engraçada porra caralho nao quero nem saber vai tomar todo mundo no cu", "lr"))