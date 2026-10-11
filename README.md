# LLM Fingerprint: Identifying LLMs from Their Responses

CMPSC 448 Midterm Project – Sol Lee (individual project)

This project tests whether a classifier can tell which LLM (GPT, Claude, or Gemini) wrote a response. I built a dataset of 360 responses (120 prompts × 3 LLMs) and trained a CNN and an LSTM on it.

Report: [CMPSC_448_Midterm_Project_Sol Lee.pdf](report/CMPSC_448_Midterm_Project_Sol%20Lee.pdf)

## Results (test accuracy)

| Model | Input only | Output only | Input + output |
|---|---|---|---|
| CNN | 0.333 | 0.883 | 0.817 |
| LSTM | 0.333 | 0.483 | 0.467 |

Random guessing = 0.333. See the report for RQ3 and RQ4 (extra credit).

## Dataset

- LLMs: `gpt-5.4-mini-2026-03-17`, `claude-haiku-5-5`, `gemini-3.6-flash`
- Prompts: 30 each from Dolly (general, writing), GSM8K (math), and MBPP (coding)
- Every prompt was sent to all three LLMs
- Split by prompt: 240 train / 60 validation / 60 test responses

## Files

```
data/
  prompts.csv              120 sampled prompts with source info
  raw/responses_*.csv      responses from each LLM
  dataset.csv              merged dataset with train/val/test split
src/
  sample_prompts.py        sample prompts from public datasets
  collect_responses.py     send prompts to an LLM API and save responses
  check_responses.py       check for missing, cut-off, or empty responses
  make_splits.py           merge responses and split by prompt
  preprocess.py            tokenizer, vocabulary, padding
  models.py                CNN and LSTM
  train.py                 RQ1 and RQ2 (6 experiments)
  rq3_cross_category.py    RQ3: leave-one-category-out
  rq4_analysis.py          RQ4: style statistics and formatting removal
  plot_curves.py           training curve figure
  test_call.py, test_gemini.py   quick API tests used during setup
results/                   metrics, confusion matrices, predictions, figures
report/                    final report (PDF)
```

## How to Run

1. Set up the environment:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create a `.env` file with API keys (only needed to collect new responses):

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

3. Run the scripts in order:

```
python src/sample_prompts.py
python src/collect_responses.py gpt
python src/collect_responses.py claude
python src/collect_responses.py gemini
python src/make_splits.py
python src/train.py
python src/rq3_cross_category.py
python src/rq4_analysis.py
python src/plot_curves.py
```

The collected responses are already in `data/`, so you can skip the API steps and start from `make_splits.py`.