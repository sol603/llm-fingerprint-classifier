# Merge the three response files and split the data into train / val / test.
# The split is done by prompt_id, so all 3 responses to the same prompt stay in the same split.

import pandas as pd

SEED = 42

# 1. merge the three response files into one dataset
parts = [pd.read_csv(f"data/raw/responses_{llm}.csv") for llm in ["gpt", "claude", "gemini"]]
data = pd.concat(parts, ignore_index=True)
data = data[["prompt_id", "category", "llm_name", "prompt", "response"]]

# 2. within each category: 20 prompts for train, 5 for val, 5 for test
prompts = pd.read_csv("data/prompts.csv")
split_of = {}
for category in ["general", "writing", "math", "coding"]:
    ids = prompts[prompts["category"] == category]["prompt_id"]
    ids = ids.sample(frac=1, random_state=SEED).tolist()   # shuffle with a fixed seed
    for pid in ids[:20]:
        split_of[pid] = "train"
    for pid in ids[20:25]:
        split_of[pid] = "val"
    for pid in ids[25:]:
        split_of[pid] = "test"

data["split"] = data["prompt_id"].map(split_of)
data.to_csv("data/dataset.csv", index=False)

# 3. quick checks
print(pd.crosstab(data["split"], data["llm_name"]))
print()
print(pd.crosstab(data["split"], data["category"]))
print()
print("prompts that ended up in more than one split:",
      (data.groupby("prompt_id")["split"].nunique() > 1).sum())