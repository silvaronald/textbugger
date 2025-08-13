from whitebox import AdversarialAttack
import numpy as np
import pandas as pd 
import csv 
import os

datasets = ["rtmr", "hate", "kaggle"]

def log_result(csv_file, original_text, adversarial_text, original_label, num_bugs, attack_successful):
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([original_text, adversarial_text, original_label, num_bugs, attack_successful])

for d in datasets:
    attacker = AdversarialAttack(d)

    try:
        X_test = pd.read_csv(f"../../datasets/{d}/X_test.csv")["text"]
    except:
        X_test = pd.read_csv(f"../../datasets/{d}/test.csv")["text"]

    for model in ["lr", "lstm", "cnn"]:
        CSV_FILE = f"results/{d}/{model}_results.csv"

        os.makedirs(f"results/{d}", exist_ok=True)
        with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["original_text", "adversarial_text", "original_label", "num_perturbed", "attack_sucessful"])

        for x in X_test:
            adv_text, og_label, num_perturbed, success = attacker.generate_adversarial(x, model_type=model)

            log_result(CSV_FILE, x, adv_text, og_label, num_perturbed, success)