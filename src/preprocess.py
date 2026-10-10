# Text preprocessing: raw text -> tokens -> vocabulary -> padded token IDs.

import re
from collections import Counter
import pandas as pd

PAD, UNK, SEP, NL = "<pad>", "<unk>", "<sep>", "<nl>"


def tokenize(text):
    # lowercase, then split into words, numbers, single symbols, and newlines.
    # symbols like * # - ` are kept on purpose because markdown formatting
    # is probably a big part of each LLM's style.
    tokens = re.findall(r"\n|\w+|[^\w\s]", str(text).lower())
    return [NL if t == "\n" else t for t in tokens]

def remove_formatting(text):
    # RQ4 ablation: strip markdown and line breaks, keep only the words and normal punctuation
    text = re.sub(r"^\s*[-*•]\s+", "", str(text), flags=re.MULTILINE)   # bullet markers at line start
    text = re.sub(r"[#*`|>]", " ", text)                                  # markdown symbols
    text = re.sub(r"\s+", " ", text)                                      # newlines -> single spaces
    return text


def get_tokens(row, mode):
    # mode decides what the classifier gets to see (RQ2)
    if mode == "input":
        return tokenize(row["prompt"])
    if mode == "output":
        return tokenize(row["response"])
    if mode == "both":
        return tokenize(row["prompt"]) + [SEP] + tokenize(row["response"])
    if mode == "plain":
        return tokenize(remove_formatting(row["response"]))
    raise ValueError(mode)

def build_vocab(token_lists, min_freq=2):
    # built from the training set only, so val/test words don't leak in.
    # words that appear only once are treated as unknown.
    counts = Counter(t for tokens in token_lists for t in tokens)
    vocab = {PAD: 0, UNK: 1, SEP: 2}
    for token, count in counts.most_common():
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(tokens, vocab, max_len):
    ids = [vocab.get(t, vocab[UNK]) for t in tokens][:max_len]   # cut if too long
    length = len(ids)
    ids = ids + [vocab[PAD]] * (max_len - length)                 # pad if too short
    return ids, length


if __name__ == "__main__":
    # print some stats so we can pick a good max_len
    data = pd.read_csv("data/dataset.csv")
    train = data[data["split"] == "train"]

    for mode in ["input", "output", "both"]:
        token_lists = [get_tokens(row, mode) for _, row in train.iterrows()]
        lengths = pd.Series([len(t) for t in token_lists])
        vocab = build_vocab(token_lists)
        print(f"{mode}: vocab size = {len(vocab)}, length mean = {lengths.mean():.0f}, "
              f"median = {lengths.median():.0f}, 90th pct = {lengths.quantile(0.9):.0f}, "
              f"max = {lengths.max()}")

    print()
    print(tokenize("## Title\n**Bold** text, with `code` and 3.14!"))