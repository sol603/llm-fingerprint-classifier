# RQ4: what makes the three LLMs different?
# Part 1: simple style statistics for each LLM.
# Part 2: ablation - retrain the CNN on responses with all formatting removed.

import re
import pandas as pd
from train import run_experiment

data = pd.read_csv("data/dataset.csv")


def style_stats(text):
    text = str(text)
    words = text.split()
    first_100 = [w.lower() for w in words[:100]]
    return {
        "words": len(words),
        # unique words / total words, using only the first 100 words so long answers aren't penalized
        "lexical_diversity": len(set(first_100)) / max(len(first_100), 1),
        "lines": text.count("\n") + 1,
        "headers": len(re.findall(r"^#+ ", text, flags=re.MULTILINE)),
        "bold": text.count("**") // 2,
        "bullets": len(re.findall(r"^\s*[-*] ", text, flags=re.MULTILINE)),
        "code_blocks": text.count("```") // 2,
    }


# Part 1: average style stats per LLM
stats = pd.DataFrame([style_stats(t) for t in data["response"]])
stats["llm_name"] = data["llm_name"]
stats["category"] = data["category"]

per_llm = stats.groupby("llm_name").mean(numeric_only=True).round(2)
per_llm.to_csv("results/rq4_style_stats.csv")
print("===== average style stats per LLM =====")
print(per_llm.T)

print("\n===== average words per category =====")
print(stats.pivot_table(index="category", columns="llm_name", values="words").round(0))

# Part 2: same CNN setup as RQ1, but the responses have no markdown or line breaks
print("\n===== ablation: CNN on responses with formatting removed =====")
result = run_experiment("cnn", "plain", data)
pd.DataFrame([result]).to_csv("results/rq4_ablation.csv", index=False)
print(f"\nCNN output (with formatting):    test acc 0.883")
print(f"CNN plain (formatting removed):  test acc {result['test_acc']:.3f}")