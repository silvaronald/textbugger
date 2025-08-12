import pandas as pd

for model in ["lr", "cnn", "lstm"]:
    # Load the CSV file
    df = pd.read_csv(f"rtmr/{model}_results.csv")  # replace with your filename

    # Ensure the 'attack_sucessful' column is boolean
    df["attack_sucessful"] = df["attack_sucessful"].astype(bool)

    # Calculate the proportion of True values
    proportion = df["attack_sucessful"].mean()

    print(f"Proportion of successful attacks: {proportion:.2%}")
