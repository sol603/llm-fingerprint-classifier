# Sample 120 prompts (30 per category) from public datasets.
# Only the prompts are used; responses are collected separately from GPT, Claude, and Gemini.

import pandas as pd
from datasets import load_dataset

SEED = 42
N_PER_CATEGORY = 30

# Prompts I removed after reading through the sampled list (matched by how the prompt starts)
EXCLUDE = [
    "What are the best places to eat",       # answer changes over time (local restaurant recs)
    "Suppose you started at 0 degrees N",    # labeled writing, but it's really a math puzzle
    "What is the most popular sport?",       # labeled writing, but it's really a general question
    "Imagine you are a java expert",         # labeled writing, but it's really a coding question
    "As an immigrant",                       # asks about the model itself, so it might say its own name
    "Given these two sets of instructions",  # labeled writing, but it's really a reasoning puzzle
]


def is_excluded(text):
    for start in EXCLUDE:
        if text.strip().startswith(start):
            return True
    return False


def pick_prompts(df, text_column, category, source):
    # shuffle with a fixed seed so anyone running this gets the same prompts
    shuffled = df.sample(frac=1, random_state=SEED)

    picked = []
    for row_id, text in zip(shuffled.index, shuffled[text_column]):
        if is_excluded(text):
            continue
        picked.append({
            "category": category,
            "source": source,
            "source_id": row_id,   # row number in the original dataset
            "prompt": text.strip(),
        })
        if len(picked) == N_PER_CATEGORY:
            break
    return picked


# Dolly: only keep questions that don't need an extra context passage
dolly = load_dataset("databricks/databricks-dolly-15k", split="train").to_pandas()
dolly = dolly[dolly["context"].str.strip() == ""]
open_qa = dolly[dolly["category"] == "open_qa"]
creative = dolly[dolly["category"] == "creative_writing"]

gsm8k = load_dataset("openai/gsm8k", "main", split="test").to_pandas()
mbpp = load_dataset("google-research-datasets/mbpp", "full", split="test").to_pandas()

rows = []
rows += pick_prompts(open_qa, "instruction", "general", "databricks-dolly-15k/open_qa")
rows += pick_prompts(creative, "instruction", "writing", "databricks-dolly-15k/creative_writing")
rows += pick_prompts(gsm8k, "question", "math", "gsm8k/main/test")
rows += pick_prompts(mbpp, "text", "coding", "mbpp/full/test")

prompts = pd.DataFrame(rows)
prompts.insert(0, "prompt_id", range(1, len(prompts) + 1))
prompts.to_csv("data/prompts.csv", index=False)

print(prompts["category"].value_counts())