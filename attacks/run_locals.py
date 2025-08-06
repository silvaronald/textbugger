from blackbox import BlackBoxTextBugger
from model_wrapper import ClassificationWrapper
import pandas as pd 
import csv 
import os
import re

models = ["fasttext", "sst2", "toxic-bert", "twitter-roberta"]
datasets = ["rtmr", "hate", "kaggle"]

def log_result(csv_file, original_text, adversarial_text, original_label, num_reqs, attack_successful):
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([original_text, adversarial_text, original_label, num_reqs, attack_successful])

for model in models:
    cw = ClassificationWrapper(model)
    bb = BlackBoxTextBugger(cw.predict)

    for dataset in datasets:
        try:
            X_test = pd.read_csv(f"../datasets/{dataset}/X_test.csv")["text"]
        except:
            X_test = pd.read_csv(f"../datasets/{dataset}/test.csv")["text"]

        file = f"results/{dataset}/{model}_results.csv"
        os.makedirs(f"results/{dataset}", exist_ok=True)

        with open(file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["original_text", "adversarial_text", "original_label", "requests_number", "attack_sucessful"])
        
        for x in X_test:
            adv_text, og_label, num_reqs, success = bb.attack(re.sub(r'\s+', ' ', x).strip()) # remover \n, \t e multiplos espaços

            log_result(file, x, adv_text, og_label, num_reqs, success)