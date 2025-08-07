import pandas as pd

models = ["bert-mini"]
datasets = ["rtmr", "hate", "kaggle"]

for m in models:
    print("-----------------")
    print(f"Results for {m}:")

    for d in datasets:
        df = pd.read_csv(f"results/{d}/{m}_results.csv")

        total_words = df["original_text"].astype(str).apply(lambda x: len(x.split())).sum()

        # Sum of requests_number
        total_requests = df["requests_number"].sum()

        # Sum of num_perturbed
        total_perturbed = df["num_perturbed"].sum()

        # Rate of successful attacks
        # Ensure attack_sucessful column is boolean or castable
        success_rate = df["attack_sucessful"].astype(bool).mean()

        # Print results
        print("Perturbed words ratio:", total_perturbed / total_words)
        print("Total requests_number:", total_requests)
        print(f"Attack success rate: {success_rate:.2%}")
        print()