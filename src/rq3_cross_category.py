# RQ3: does the LLM fingerprint carry over to a task type the classifier never saw?
# Leave one category out: train on the other 3 categories, test on the held-out one.

import copy
import pandas as pd
import torch
import torch.nn as nn

from preprocess import get_tokens, build_vocab
from models import CNNClassifier
from train import set_seed, make_tensors, run_epoch, SEED, EPOCHS, LR, LABELS

MODE = "output"   # response only, same as RQ1

data = pd.read_csv("data/dataset.csv")
summary, all_preds = [], []

for held_out in ["general", "writing", "math", "coding"]:
    set_seed(SEED)
    others = data[data["category"] != held_out]
    train = others[others["split"] == "train"]
    val = others[others["split"] == "val"]
    test = data[data["category"] == held_out]   # all 90 responses of the unseen category

    vocab = build_vocab([get_tokens(row, MODE) for _, row in train.iterrows()])
    X_tr, L_tr, y_tr = make_tensors(train, vocab, MODE)
    X_va, L_va, y_va = make_tensors(val, vocab, MODE)
    X_te, L_te, y_te = make_tensors(test, vocab, MODE)

    model = CNNClassifier(vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # same training setup as RQ1: keep the epoch with the best validation accuracy
    best_val_acc, best_epoch, best_state = -1, 0, None
    for epoch in range(1, EPOCHS + 1):
        run_epoch(model, X_tr, L_tr, y_tr, loss_fn, optimizer)
        _, va_acc, _ = run_epoch(model, X_va, L_va, y_va, loss_fn)
        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    _, test_acc, test_preds = run_epoch(model, X_te, L_te, y_te, loss_fn)
    print(f"held out {held_out:8s} | best epoch {best_epoch:2d} | "
          f"val acc {best_val_acc:.3f} | test acc on unseen category {test_acc:.3f}")

    summary.append({"held_out": held_out, "best_epoch": best_epoch,
                    "val_acc": best_val_acc, "test_acc": test_acc})
    preds = test[["prompt_id", "category"]].copy()
    preds["true"] = test["llm_name"]
    preds["pred"] = [LABELS[i] for i in test_preds]
    all_preds.append(preds)

pd.DataFrame(summary).to_csv("results/rq3_summary.csv", index=False)
all_preds = pd.concat(all_preds)
all_preds.to_csv("results/rq3_predictions.csv", index=False)

# accuracy for each LLM on each unseen category
print()
print(all_preds.assign(correct=all_preds["true"] == all_preds["pred"])
      .pivot_table(index="category", columns="true", values="correct").round(3))