# Quick sanity checks on the collected responses before training anything.

import pandas as pd

NAMES = ["openai", "chatgpt", "gpt", "anthropic", "claude", "gemini", "google"]

for llm in ["gpt", "claude", "gemini"]:
    df = pd.read_csv(f"data/raw/responses_{llm}.csv")
    responses = df["response"].fillna("")

    print(f"===== {llm} =====")
    print("rows:", len(df))
    print("missing prompt_ids:", sorted(set(range(1, 121)) - set(df["prompt_id"])))
    print("finish reasons:", df["finish_reason"].value_counts().to_dict())
    print("empty responses:", (responses.str.strip() == "").sum())
    print("average words:", round(responses.str.split().str.len().mean(), 1))

    # does the response mention any model or company name?
    for name in NAMES:
        hits = df[responses.str.lower().str.contains(name)]["prompt_id"].tolist()
        if hits:
            print(f"  mentions '{name}' in prompts:", hits)
    print()