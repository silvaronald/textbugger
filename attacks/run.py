from whitebox import AdversarialAttack
import numpy as np
import pandas as pd 
import csv 
import os

FOLDER = "rtmr"

def log_result(csv_file, original_text, adversarial_text, original_label, attack_successful):
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([original_text, adversarial_text, original_label, attack_successful])

attacker = AdversarialAttack(FOLDER)

X_test = pd.read_csv(f"../datasets/{FOLDER}/X_test.csv")["text"]

for model in ["lr", "lstm", "cnn"]:
    CSV_FILE = f"{FOLDER}/{model}_results.csv"

    os.makedirs(FOLDER, exist_ok=True)
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["original_text", "adversarial_text", "original_label", "attack_sucessful"])

    for x in X_test:
        adv_text, og_label, success = attacker.generate_adversarial(x, model_type=model)

        log_result(CSV_FILE, x, adv_text, og_label, success)