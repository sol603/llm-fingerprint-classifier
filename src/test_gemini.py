import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

for model in ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"]:
    try:
        r = client.models.generate_content(model=model, contents="Say hi in one sentence.")
        print(f"[OK]   {model}: {r.text.strip()}")
    except Exception as e:
        print(f"[FAIL] {model}: {str(e)[:80]}")