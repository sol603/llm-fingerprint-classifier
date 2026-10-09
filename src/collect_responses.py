# Send every prompt in data/prompts.csv to one LLM and save the responses.
# Usage: python src/collect_responses.py gpt    (or claude / gemini)
# Each model saves to its own file, so the three can run at the same time.

import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types

load_dotenv()

MAX_TOKENS = 4096   # same output limit for all three models

MODELS = {
    "gpt": "gpt-5.4-mini-2026-03-17",
    "claude": "claude-haiku-5-5",
    "gemini": "gemini-3.6-flash",   # 3.8 and 3.7 kept giving 503 errors on Oct 9
}

gpt_client = OpenAI()
claude_client = Anthropic()
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# each function returns (response text, finish reason)
# no system prompt and default temperature for all three

def ask_gpt(prompt):
    r = gpt_client.chat.completions.create(
        model=MODELS["gpt"],
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=MAX_TOKENS,
    )
    return r.choices[0].message.content, r.choices[0].finish_reason


def ask_claude(prompt):
    r = claude_client.messages.create(
        model=MODELS["claude"],
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in r.content if block.type == "text")
    return text, r.stop_reason


def ask_gemini(prompt):
    r = gemini_client.models.generate_content(
        model=MODELS["gemini"],
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=MAX_TOKENS),
    )
    return r.text, str(r.candidates[0].finish_reason)


llm = sys.argv[1]   # "gpt", "claude", or "gemini"
ask = {"gpt": ask_gpt, "claude": ask_claude, "gemini": ask_gemini}[llm]
out_path = f"data/raw/responses_{llm}.csv"

prompts = pd.read_csv("data/prompts.csv")

# if some responses were already saved (e.g. the script stopped halfway), skip those
if os.path.exists(out_path):
    results = pd.read_csv(out_path).to_dict("records")
else:
    results = []
done_ids = {r["prompt_id"] for r in results}

for _, row in prompts.iterrows():
    if row["prompt_id"] in done_ids:
        continue

    success = False
    for attempt in range(1, 6):   # try up to 5 times
        try:
            text, finish = ask(row["prompt"])
            success = True
            break
        except Exception as e:
            print(f"  error on prompt {row['prompt_id']} (attempt {attempt}): {str(e)[:600]}")
            time.sleep(10 * attempt)   # wait a bit longer each time

    if not success:
        print(f"  skipping prompt {row['prompt_id']} for now, run the script again later")
        continue

    results.append({
        "prompt_id": row["prompt_id"],
        "category": row["category"],
        "llm_name": llm,
        "model": MODELS[llm],
        "prompt": row["prompt"],
        "response": text or "",
        "finish_reason": finish,
    })
    pd.DataFrame(results).to_csv(out_path, index=False)   # save after every response
    print(f"[{llm}] {len(results)}/{len(prompts)} saved (prompt {row['prompt_id']})")
    time.sleep(1)