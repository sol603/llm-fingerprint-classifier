# Train and evaluate CNN and LSTM on all three input settings (6 experiments).
# Results are saved under results/<model>_<mode>/ and summarized in results/summary.csv.

import copy
import json
import os
import random

import matplotlib
matplotlib.use("Agg")   # save plots to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from preprocess import get_tokens, build_vocab, encode
from models import CNNClassifier, LSTMClassifier

SEED = 42
MAX_LEN = 512
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
LABELS = ["gpt", "claude", "gemini"]
LABEL_ID = {name: i for i, name in enumerate(LABELS)}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_tensors(df, vocab, mode):
    ids, lengths = [], []
    for _, row in df.iterrows():
        x, length = encode(get_tokens(row, mode), vocab, MAX_LEN)
        ids.append(x)
        lengths.append(length)
    y = [LABEL_ID[name] for name in df["llm_name"]]
    return torch.tensor(ids), torch.tensor(lengths), torch.tensor(y)


def run_epoch(model, X, L, y, loss_fn, optimizer=None):
    # trains the model if an optimizer is given, otherwise only evaluates
    training = optimizer is not None
    if training:
        model.train()
        order = torch.randperm(len(y))   # shuffle training data every epoch
    else:
        model.eval()
        order = torch.arange(len(y))

    total_loss = 0.0
    preds = torch.zeros_like(y)
    for start in range(0, len(y), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        with torch.set_grad_enabled(training):
            logits = model(X[idx], L[idx])
            loss = loss_fn(logits, y[idx])
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(idx)
        preds[idx] = logits.argmax(dim=1)

    accuracy = (preds == y).float().mean().item()
    return total_loss / len(y), accuracy, preds


def run_experiment(model_name, mode, data):
    set_seed(SEED)
    train = data[data["split"] == "train"]
    val = data[data["split"] == "val"]
    test = data[data["split"] == "test"].copy()

    # vocabulary from training data only
    vocab = build_vocab([get_tokens(row, mode) for _, row in train.iterrows()])
    X_tr, L_tr, y_tr = make_tensors(train, vocab, mode)
    X_va, L_va, y_va = make_tensors(val, vocab, mode)
    X_te, L_te, y_te = make_tensors(test, vocab, mode)

    if model_name == "cnn":
        model = CNNClassifier(vocab_size=len(vocab))
    else:
        model = LSTMClassifier(vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # train, and keep the model from the epoch with the best validation accuracy
    history = []
    best_val_acc, best_epoch, best_state = -1, 0, None
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, _ = run_epoch(model, X_tr, L_tr, y_tr, loss_fn, optimizer)
        va_loss, va_acc, _ = run_epoch(model, X_va, L_va, y_va, loss_fn)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        print(f"epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}")
        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            best_state = copy.deepcopy(model.state_dict())

    # evaluate the best model on the test set (only once)
    model.load_state_dict(best_state)
    _, test_acc, test_preds = run_epoch(model, X_te, L_te, y_te, loss_fn)

    # save everything
    out_dir = f"results/{model_name}_{mode}"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(history).to_csv(f"{out_dir}/history.csv", index=False)

    report = classification_report(y_te, test_preds, target_names=LABELS,
                                   output_dict=True, zero_division=0)
    with open(f"{out_dir}/test_report.json", "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_te, test_preds)
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(cmap="Blues")
    plt.title(f"{model_name.upper()} - {mode}")
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    test["true"] = test["llm_name"]
    test["pred"] = [LABELS[i] for i in test_preds]
    test[["prompt_id", "category", "true", "pred"]].to_csv(f"{out_dir}/predictions.csv", index=False)

    return {"model": model_name, "mode": mode, "best_epoch": best_epoch,
            "val_acc": best_val_acc, "test_acc": test_acc,
            "test_precision": report["macro avg"]["precision"],
            "test_recall": report["macro avg"]["recall"],
            "test_f1": report["macro avg"]["f1-score"]}


if __name__ == "__main__":
    data = pd.read_csv("data/dataset.csv")
    summary = []
    for model_name in ["cnn", "lstm"]:
        for mode in ["input", "output", "both"]:
            print(f"\n===== {model_name} / {mode} =====")
            summary.append(run_experiment(model_name, mode, data))

    summary = pd.DataFrame(summary)
    summary.to_csv("results/summary.csv", index=False)
    print()
    print(summary.round(3).to_string(index=False))