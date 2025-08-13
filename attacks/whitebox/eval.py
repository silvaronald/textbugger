import pandas as pd

models = ["lr", "cnn", "lstm"]
datasets = ["rtmr", "hate", "kaggle"]

for m in models:
    print("-----------------")
    print(f"Results for {m}:")

    for d in datasets:
        print(f"Dataset {d}")
        df = pd.read_csv(f"results/{d}/{m}_results.csv")

        total_words = df["original_text"].astype(str).apply(lambda x: len(x.split())).sum()
        # Sum of num_perturbed
        total_perturbed = df["num_perturbed"].sum()

        # Rate of successful attacks
        # Ensure attack_sucessful column is boolean or castable
        success_rate = df["attack_sucessful"].astype(bool).mean()

        # Print results
        print(f"Perturbed words ratio: {(total_perturbed / total_words):.4f}")
        print(f"Attack success rate: {success_rate:.2%}")
        print()