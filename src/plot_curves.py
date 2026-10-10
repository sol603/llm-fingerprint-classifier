# Plot training curves (loss and accuracy per epoch) for the output-only CNN and LSTM.

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("results/figures", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for row, model in enumerate(["cnn", "lstm"]):
    h = pd.read_csv(f"results/{model}_output/history.csv")

    axes[row, 0].plot(h["epoch"], h["train_loss"], label="train")
    axes[row, 0].plot(h["epoch"], h["val_loss"], label="validation")
    axes[row, 0].set_title(f"{model.upper()} (output only) - loss")

    axes[row, 1].plot(h["epoch"], h["train_acc"], label="train")
    axes[row, 1].plot(h["epoch"], h["val_acc"], label="validation")
    axes[row, 1].set_title(f"{model.upper()} (output only) - accuracy")

    for ax in axes[row]:
        ax.set_xlabel("epoch")
        ax.set_xticks(range(0, 21, 5))
        ax.legend()

plt.tight_layout()
plt.savefig("results/figures/learning_curves_output.png", dpi=150)
print("saved results/figures/learning_curves_output.png")