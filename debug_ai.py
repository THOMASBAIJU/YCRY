import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

keys = []
if os.getenv("GOOGLE_API_KEY"): keys.append(os.getenv("GOOGLE_API_KEY"))
if os.getenv("GEMINI_API_KEY"): keys.append(os.getenv("GEMINI_API_KEY"))
if os.getenv("NUTRITION_API_KEY"): keys.append(os.getenv("NUTRITION_API_KEY"))

# Filter AIza
keys = [k for k in keys if k and k.startswith("AIza")]
keys = list(set(keys))

print(f"Found {len(keys)} potential Google Keys.")

models = ['gemini-2.0-flash-lite', 'gemini-1.5-flash', 'gemini-pro']

for key in keys:
    masked = key[:5] + "..." + key[-4:]
    print(f"\n--- Testing Key: {masked} ---")
    genai.configure(api_key=key)
    
    for m in models:
        print(f"Testing Model: {m}...", end=" ")
        try:
            model = genai.GenerativeModel(m)
            response = model.generate_content("Say hello", generation_config={"response_mime_type": "text/plain"})
            print(f"✅ SUCCESS! Response: {response.text.strip()}")
            exit(0) # Exit on first success
        except Exception as e:
            print(f"❌ FAILED. Error: {str(e)}")
