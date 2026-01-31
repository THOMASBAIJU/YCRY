import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ No API Key found in .env")
else:
    genai.configure(api_key=api_key)
    print(f"🔑 Key: {api_key[:5]}...{api_key[-5:]}")
    
    print("\n🔍 Listing Models available for generateContent:")
    try:
        with open("available_models.txt", "w") as f:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(f"- {m.name}")
                    f.write(f"{m.name}\n")
        print("✅ Models saved to available_models.txt")
    except Exception as e:
        print(f"❌ Error listing models: {e}")
