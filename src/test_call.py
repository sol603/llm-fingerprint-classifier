import os
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types

load_dotenv()

PROMPT = "In two sentences, explain what photosynthesis is."
MAX_TOKENS = 1024

def test(name, fn):
    print(f"===== {name} =====")
    try:
        print(fn())
    except Exception as e:
        print(f"[FAIL] {e}")
    print()

def call_gpt():
    r = OpenAI().chat.completions.create(
        model="gpt-5.4-mini-2026-03-17",
        messages=[{"role": "user", "content": PROMPT}],
        max_completion_tokens=MAX_TOKENS,
    )
    return r.choices[0].message.content

def call_claude():
    r = Anthropic().messages.create(
        model="claude-haiku-5-5",
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": PROMPT}],
    )
    return r.content[0].text

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def call_gemini():
    r = gemini_client.models.generate_content(
        model="gemini-3.8-flash",
        contents=PROMPT,
        config=types.GenerateContentConfig(max_output_tokens=MAX_TOKENS),
    )
    return r.text

test("GPT", call_gpt)
test("Claude", call_claude)
test("Gemini", call_gemini)