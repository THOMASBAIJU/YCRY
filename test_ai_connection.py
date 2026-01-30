import google.generativeai as genai
import sys
import traceback

API_KEY = "AIzaSyAS0ZyP5xXhalteK_XFdC9u5URTYmsvePU"

def log(msg):
    print(msg)
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def test_gemini():
    with open("debug_log.txt", "w", encoding="utf-8") as f:
        f.write("--- LOG START ---\n")
    
    try:
        genai.configure(api_key=API_KEY)
        
        log("📋 Fetching available models...")
        all_models = list(genai.list_models())
        
        # Filter for generateContent support
        candidates = []
        for m in all_models:
            if 'generateContent' in m.supported_generation_methods:
                candidates.append(m.name)
        
        log(f"   Found {len(candidates)} generation models.")
        log(f"   Candidates: {candidates[:5]}...")
        
        working_model = None
        
        # Try candidates in order
        for name in candidates:
            # Strip 'models/' prefix if needed, though sdk handles it
            short_name = name.replace("models/", "")
            log(f"👉 Trying: {name} (short: {short_name})")
            
            try:
                model = genai.GenerativeModel(short_name)
                response = model.generate_content("Hi")
                log(f"   ✅ SUCCESS! Response: {response.text}")
                working_model = short_name
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    log("   ⚠️ QUOTA ERROR (429)")
                else:
                    log(f"   ❌ FAILED: {err_str[:100]}...")
        
        if working_model:
            log(f"\n🎉 FOUND WORKING MODEL: {working_model}")
            
            # Save the working model name to a file so we can read it easily
            with open("working_model_name.txt", "w") as f:
                f.write(working_model)
                
        else:
            log("\n❌ NO WORKING MODELS FOUND.")
        
    except Exception as e:
        log("\n❌ CRITICAL FAILURE")
        log(traceback.format_exc())

if __name__ == "__main__":
    test_gemini()
