import os
import datetime
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import database as db
from werkzeug.utils import secure_filename
import google.generativeai as genai
import threading
app = Flask(__name__)
app.secret_key = "SUPER_SECRET_KEY_YCRY"

# CONFIG
PROFILE_FOLDER = 'static/profile_pics'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROFILE_FOLDER'] = PROFILE_FOLDER

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROFILE_FOLDER'], exist_ok=True)
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

db.init_db()

# --- LOAD AI ---
# --- CONFIG AI ---
GEMINI_API_KEY = "AIzaSyAS0ZyP5xXhalteK_XFdC9u5URTYmsvePU"
genai.configure(api_key=GEMINI_API_KEY)

# --- LOAD AI ---
MODEL_PATH = os.path.join(os.getcwd(), "model_brain.h5")
ai_model = None
plot_lock = threading.Lock()

print(f"🔍 Checking Model Path: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    print("✅ File Found. Attempting load...")
    try:
        ai_model = load_model(MODEL_PATH)
        print("✅ AI Model Loaded Successfully!")
        
        # --- WARMUP ---
        try:
            print("⏳ Warming up AI model...")
            dummy_input = np.zeros((1, 64, 64, 3))
            ai_model.predict(dummy_input, verbose=0)
            print("🔥 AI Model Warmed Up & Ready")
        except Exception as e:
            print(f"⚠️ Model Warmup Failed: {e}")
            
    except Exception as e:
        print(f"❌ Keras Load Model Failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ Model File NOT FOUND at path!")

# --- HELPERS ---
def calculate_age(dob_str):
    dob = datetime.datetime.strptime(dob_str, "%Y-%m-%d").date()
    return (datetime.date.today() - dob).days // 30

def get_milestones(age_months):
    """Returns a list of milestones based on the baby's age."""
    if age_months < 4:
        return {
            "title": "0-3 Months (Newborn)",
            "motor": ["Lifts head/chest on tummy", "Stretches & kicks on back"],
            "social": ["Social smile (smiles at people)", "Makes eye contact"],
            "comm": ["Coos & gurgles", "Different cries for needs"]
        }
    elif age_months < 7:
        return {
            "title": "4-6 Months (Active)",
            "motor": ["Rolls over (tummy to back)", "Sits with support", "Reaches/grabs toys"],
            "social": ["Knows familiar faces", "Likes looking in mirrors"],
            "comm": ["Babbles (ba-ba, ma-ma)", "Responds to name"]
        }
    elif age_months < 10:
        return {
            "title": "7-9 Months (Explorer)",
            "motor": ["Sits without support", "Crawls", "Pincer grasp (thumb & finger)"],
            "social": ["Separation anxiety (clingy)", "Has favorite toys"],
            "comm": ["Understands 'No'", "Points at things"]
        }
    else: # 10-12+ months
        return {
            "title": "10-12 Months (Toddler)",
            "motor": ["Pulls to stand", "Cruising (walks holding furniture)", "First steps"],
            "social": ["Plays peek-a-boo", "Waves bye-bye"],
            "comm": ["Says 'mama'/'dada'", "Tries to copy words"]
        }

def analyze_birth_health(weight, apgar, head, chest):
    analysis = []
    status = "Healthy"
    if weight < 2.5:
        analysis.append("⚠️ Low Birth Weight")
        status = "Attention Needed"
    elif weight > 4.5:
        analysis.append("⚠️ High Birth Weight")
    else:
        analysis.append("✅ Healthy Weight")

    if apgar >= 7:
        analysis.append("✅ Good Apgar")
    elif apgar >= 4:
        analysis.append("⚠️ Moderate Apgar")
        status = "Attention Needed"
    else:
        analysis.append("🚨 Low Apgar")
        status = "Critical"

    if head >= chest:
        analysis.append("✅ Normal Head Ratio")
    else:
        analysis.append("ℹ️ Chest > Head")

    return status, " | ".join(analysis)

def get_vaccine_schedule(dob_str, completed_list):
    dob = datetime.datetime.strptime(dob_str, "%Y-%m-%d").date()
    today = datetime.date.today()
    warning_active = False
    
    milestones = {
        0:   ["Birth", ["BCG", "OPV-0", "Hep-B1"]],
        42:  ["6 Weeks", ["DTwP-1", "IPV-1", "Hep-B2"]],
        70:  ["10 Weeks", ["DTwP-2", "IPV-2"]],
        98:  ["14 Weeks", ["DTwP-3", "IPV-3", "Hep-B3"]],
        270: ["9 Months", ["Measles-1", "Vit A"]],
        450: ["15 Months", ["MMR-1", "Varicella"]]
    }
    
    schedule = []
    for days, (label, shots) in milestones.items():
        due = dob + datetime.timedelta(days=days)
        is_date_arrived = today >= due
        is_overdue = is_date_arrived and not all(s in completed_list for s in shots)
        if is_overdue: warning_active = True
        
        row_status = "Upcoming"
        row_bg = "white"
        if is_overdue:
            row_status = "Overdue"
            row_bg = "red-50"
        elif all(s in completed_list for s in shots):
            row_status = "Completed"
            row_bg = "green-50"

        shots_data = [{"name": s, "done": s in completed_list} for s in shots]
        
        schedule.append({
            "milestone": label, 
            "date": due.strftime("%d %b %Y"), 
            "shots": shots_data, 
            "status": row_status,
            "bg_color": row_bg,
            "can_take": is_date_arrived
        })
    return schedule, warning_active

@app.context_processor
def inject_user_profile():
    if 'user' in session:
        profile = db.get_profile(session['user'])
        if profile and profile.get('profile_pic'):
            return {'navbar_pic': profile['profile_pic']}
    return {'navbar_pic': None}

# --- ROUTES ---

@app.route('/')
def home():
    # 🚨 SECURITY FIX: If not logged in, go to Login Page immediately
    if 'user' not in session:
        return redirect(url_for('login'))

    # If logged in, load dashboard
    profile = db.get_profile(session['user'])
    if profile:
        baby_name = profile['baby_name']
        dob_str = profile['dob']
        health_summary = profile['health_summary']
        pic = profile['profile_pic'] if profile.get('profile_pic') else None
        
        age = calculate_age(dob_str)
        growth = db.get_latest_growth(session['user'])
        weight = f"{growth['weight']} kg" if growth else f"{profile['weight_birth']} kg (Birth)"

        # Vaccination Alert Logic
        completed_vaccines = db.get_completed_vaccines(session['user'])
        schedule, _ = get_vaccine_schedule(dob_str, completed_vaccines)
        
        vaccine_alert = {
             "type": "success", # success (green) or danger (red)
             "title": "Vaccination Status",
             "text": "All Up to Date",
             "icon": "🛡️"
        }

        for milestone in schedule:
            if milestone['status'] != "Completed":
                # Find first incomplete shot
                missing_shots = [s['name'] for s in milestone['shots'] if not s['done']]
                next_shot = missing_shots[0] if missing_shots else "Vaccine"
                
                if milestone['status'] == "Overdue":
                    vaccine_alert = {
                        "type": "danger",
                        "title": "Missing Vaccine!",
                        "text": f"Overdue: {next_shot}",
                        "icon": "⚠️"
                    }
                else:
                    vaccine_alert = {
                        "type": "success",
                        "title": "Vaccination Alert",
                        "text": f"Upcoming: {next_shot}",
                        "icon": "🛡️"
                    }
                break

        return render_template('home.html', logged_in=True, 
                               name=baby_name, age=age, weight=weight, 
                               health_summary=health_summary, profile_pic=pic,
                               vaccine_alert=vaccine_alert)
    
    # Fallback if profile missing (should rarely happen)
    return redirect(url_for('profile'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = db.login_user(username, password)
        if name == "DB_ERROR":
            flash("⚠️ Database connection failed. Please check your internet or whitelist your IP.")
        elif name:
            session['user'] = username
            session['real_name'] = name
            return redirect(url_for('home'))
        else:
            flash("Invalid Username or Password")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            user = request.form['username']
            pw = request.form['password']
            caregiver_name = request.form['caregiver_name']
            
            # Validation: Text Fields
            if len(user) < 4:
                flash("Error: Username must be at least 4 characters.")
                return render_template('register.html')
            if len(pw) < 6:
                flash("Error: Password must be at least 6 characters.")
                return render_template('register.html')

            b_name = request.form['baby_name']
            dob = request.form['dob']
            gender = request.form['gender']
            blood = request.form['blood_group']

            # Validation: Numeric Fields
            weight = float(request.form['weight'])
            height = float(request.form['height'])
            head = float(request.form['head_circ']) if request.form['head_circ'] else 0.0
            chest = float(request.form['chest_circ']) if request.form['chest_circ'] else 0.0
            apgar = int(request.form['apgar']) if request.form['apgar'] else 0

            if weight <= 0 or height <= 0:
                flash("Error: Weight and Height must be positive numbers.")
                return render_template('register.html')
            
            if not (0 <= apgar <= 10):
                flash("Error: Apgar score must be between 0 and 10.")
                return render_template('register.html')
            
            pic_filename = ""
            if 'baby_pic' in request.files:
                file = request.files['baby_pic']
                if file.filename != '':
                    ext = file.filename.split('.')[-1]
                    pic_filename = f"{user}_profile.{ext}"
                    file.save(os.path.join(app.config['PROFILE_FOLDER'], pic_filename))

            status, report = analyze_birth_health(weight, apgar, head, chest)
            
            if db.create_user(user, pw, caregiver_name):
                db.save_full_profile(user, (b_name, dob, gender, blood, weight, height, head, chest, apgar, report, pic_filename))
                session['user'] = user
                session['real_name'] = caregiver_name
                flash(f"Welcome! Health Status: {status}")
                return redirect(url_for('home'))
            else:
                flash("Error: Username already exists.")
                
        except ValueError:
            flash("Error: Invalid number format entered.")
        except Exception as e:
            print(f"Registration Error: {e}")
            flash("An unexpected error occurred.")
            
    return render_template('register.html')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user' not in session: return redirect(url_for('login'))
    
    if request.method == 'POST':
        b_name = request.form['baby_name']
        dob = request.form['dob']
        gender = request.form['gender']
        blood = request.form['blood_group']
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        head = float(request.form['head_circ'])
        chest = float(request.form['chest_circ'])
        apgar = int(request.form['apgar'])
        
        old_profile = db.get_profile(session['user'])
        pic_filename = old_profile['profile_pic'] if old_profile else "" 

        if 'baby_pic' in request.files:
            file = request.files['baby_pic']
            if file.filename != '':
                ext = file.filename.split('.')[-1]
                pic_filename = f"{session['user']}_profile.{ext}"
                file.save(os.path.join(app.config['PROFILE_FOLDER'], pic_filename))

        status, report = analyze_birth_health(weight, apgar, head, chest)
        
        db.save_full_profile(session['user'], 
                            (b_name, dob, gender, blood, weight, height, head, chest, apgar, report, pic_filename))
        
        flash("✅ Profile & Picture Updated!")
        return redirect(url_for('profile'))

    data = db.get_profile(session['user'])
    return render_template('profile.html', profile=data)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/cry', methods=['GET', 'POST'])
def cry():
    if 'user' not in session: return redirect(url_for('login'))
    
    # Initialize variables for template rendering
    pred, conf, advice = None, 0, ""

    print(f"🔍 CRY ROUTE ACCESSED. AI_MODEL STATUS: {ai_model}")

    if request.method == 'POST':
        import traceback
        try:
            # Check for file
            if 'audio' not in request.files:
                return jsonify({"error": "No file part"}), 400
            
            f = request.files['audio']
            if f.filename == '':
                return jsonify({"error": "No selected file"}), 400

            if f:
                # Save file
                import uuid
                unique_id = str(uuid.uuid4())
                audio_filename = f"temp_{unique_id}.wav"
                img_filename = f"temp_spec_{unique_id}.png"
                
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
                try:
                    f.save(audio_path)
                except Exception as e:
                    print(f"File Save Error: {str(e)}")
                    return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

                # Lock plotting to prevent thread issues
                # plt.close('all') moved inside lock downstream
                
                # Load audio
                try:
                    y, sr = librosa.load(audio_path, sr=22050, duration=5)
                except Exception as librosa_error:
                    print(f"Librosa Error: {str(librosa_error)}\n{traceback.format_exc()}")
                    return jsonify({"error": "Error reading audio file. Please ensure ffmpeg is installed and the file is a valid audio format."}), 400

                # Generate Spectrogram
                try:
                    with plot_lock:
                        plt.close('all')
                        fig = plt.figure(figsize=(4, 4))
                        librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max), sr=sr)
                        plt.axis('off')
                        
                        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
                        plt.close(fig)
                        plt.close('all')
                except Exception as plot_error:
                      print(f"Plot Error: {str(plot_error)}\n{traceback.format_exc()}")
                      return jsonify({"error": f"Error generating spectrogram: {str(plot_error)}"}), 500

                # AI Prediction
                if ai_model:
                    try:
                        img = image.img_to_array(image.load_img(img_path, target_size=(64, 64))) / 255.0
                        probs = ai_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
                        classes = ["Burping", "Discomfort", "Hunger", "Pain", "Tired"]
                        
                        pred = str(classes[np.argmax(probs)])
                        conf_val = float(np.max(probs) * 100)
                        conf = f"{conf_val:.1f}"
                        advice = {"Hunger": "Feed baby", "Pain": "Check injury", "Burping": "Burp baby", "Discomfort": "Check diaper", "Tired": "Sleep time"}.get(pred, "Check baby")
                        
                        return jsonify({
                            "prediction": pred,
                            "confidence": conf,
                            "advice": advice,
                            "success": True
                        })
                    except Exception as ai_error:
                        print(f"AI Prediction Error: {str(ai_error)}\n{traceback.format_exc()}")
                        return jsonify({"error": f"AI model prediction failed: {str(ai_error)}"}), 500
                else:
                    print("AI Model not loaded")
                    return jsonify({"error": "AI Model not loaded."}), 500

        except Exception as e:
            print(f"General Error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500
        finally:
             # plt.close('all') # Unsafe in threaded env without lock
             # Cleanup temp files
             try:
                 if 'audio_path' in locals() and os.path.exists(audio_path):
                     os.remove(audio_path)
                 if 'img_path' in locals() and os.path.exists(img_path):
                     os.remove(img_path)
             except Exception as cleanup_k:
                 print(f"Cleanup Error: {cleanup_k}")
    
    return render_template('cry.html', prediction=None, confidence=0, advice="")

@app.route('/vaccine', methods=['GET', 'POST'])
def vaccine():
    if 'user' not in session: return redirect(url_for('login'))
    prof = db.get_profile(session['user'])
    if not prof: return redirect(url_for('profile'))
    if request.method == 'POST':
        vaccine_name = request.form.get('vaccine_name')
        if vaccine_name:
            db.mark_vaccine_done(session['user'], vaccine_name)
            flash(f"✅ Marked {vaccine_name} as completed!")
            return redirect(url_for('vaccine'))
    completed = db.get_completed_vaccines(session['user'])
    schedule, show_warning = get_vaccine_schedule(prof['dob'], completed)
    return render_template('vaccine.html', schedule=schedule, baby_name=prof['baby_name'], warning=show_warning)

@app.route('/growth', methods=['GET', 'POST'])
def growth():
    if 'user' not in session: return redirect(url_for('login'))
    
    prof = db.get_profile(session['user'])
    if not prof: return redirect(url_for('profile'))
    
    # 1. Handle Physical Growth (Weight/Height)
    if request.method == 'POST':
        db.add_growth_record(session['user'], float(request.form['weight']), float(request.form['height']))
        flash("✅ Physical growth recorded!")
        
    latest = db.get_latest_growth(session['user'])
    age = calculate_age(prof['dob'])
    
    # 2. Get Mental/Motor Milestones
    milestones = get_milestones(age)
    
    # 3. Get Growth Chart Data
    history = db.get_growth_history(session['user'])
    # history is list of dicts {date, weight, height}
    chart_dates = [r['date'] for r in history] if history else []
    chart_weights = [r['weight'] for r in history] if history else []
    chart_heights = [r['height'] for r in history] if history else []

    # If no history, add birth data if available
    if not history and prof:
        chart_dates = [prof['dob']]
        chart_weights = [prof['weight_birth']]
        chart_heights = [prof['height_birth']]
    
    return render_template('growth.html', 
                           latest=latest, 
                           baby_name=prof['baby_name'], 
                           age=age, 
                           status="Healthy",
                           milestones=milestones,
                           dates=chart_dates,
                           weights=chart_weights,
                           heights=chart_heights)
@app.route('/nutrition')
def nutrition():
    if 'user' not in session: return redirect(url_for('login'))
    prof = db.get_profile(session['user'])
    if not prof: return redirect(url_for('profile'))
    
    age_months = calculate_age(prof['dob'])
    guide = get_nutrition_guide(age_months)
    return render_template('nutrition.html', guide=guide, age=age_months, baby_name=prof['baby_name'])

@app.route('/exercises')
def exercises():
    if 'user' not in session: return redirect(url_for('login'))
    prof = db.get_profile(session['user'])
    if not prof: return redirect(url_for('profile'))
    
    age_months = calculate_age(prof['dob'])
    exercises_list = get_exercises(age_months)
    return render_template('exercises.html', exercises=exercises_list, age=age_months, baby_name=prof['baby_name'])

@app.route('/health')
def health():
    if 'user' not in session: return redirect(url_for('login'))
    prof = db.get_profile(session['user'])
    if not prof: return redirect(url_for('profile'))
    
    warnings = get_warning_signs()
    warnings = get_warning_signs()
    return render_template('health.html', warnings=warnings, baby_name=prof['baby_name'])

@app.route('/assistant')
def assistant():
    if 'user' not in session: return redirect(url_for('login'))
    return render_template('ai_assistant.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    if 'user' not in session:
        return jsonify({"error": "Please login first."}), 401

    data = request.json
    user_msg = data.get('message', '')

    # 1. Get Profile Context
    profile = db.get_profile(session['user'])
    if not profile:
        return jsonify({"error": "Profile not found"}), 400

    baby_name = profile['baby_name']
    dob_str = profile['dob']
    age_months = calculate_age(dob_str)
    
    # 2. Get Growth History
    growth_history = db.get_growth_history(session['user'])
    growth_txt = "No records yet."
    if growth_history:
        # Format: "2023-01-01: 4.5kg, 55cm"
        growth_entries = [f"- {g['date']}: {g['weight']}kg, {g['height']}cm" for g in growth_history[-10:]] # Last 10
        growth_txt = "\n".join(growth_entries)

    # 3. Get Vaccine Status
    completed_vax = db.get_completed_vaccines(session['user'])
    schedule, _ = get_vaccine_schedule(dob_str, completed_vax)
    
    overdue = []
    upcoming = []
    for milestone in schedule:
        if milestone['status'] == 'Overdue':
            missing = [s['name'] for s in milestone['shots'] if not s['done']]
            if missing: overdue.extend(missing)
        elif milestone['status'] == 'Upcoming':
            upcoming.extend([s['name'] for s in milestone['shots']])
            
    vax_txt = f"Completed: {', '.join(completed_vax) if completed_vax else 'None'}\n"
    vax_txt += f"OVERDUE: {', '.join(overdue) if overdue else 'None'}\n"
    vax_txt += f"Next Due: {upcoming[0] if upcoming else 'All likely completed'}"

    # 4. Manage Session Memory
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    recent_history = session['chat_history'][-6:] 
    history_text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in recent_history])

    # 5. Build Medical System Prompt
    system_prompt = f"""
    You are Dr. Ycry, an expert Pediatric Nurse Assistant.
    You are caring for a baby named {baby_name}, who is {age_months} months old.
    
    === MEDICAL FILE ===
    [GROWTH HISTORY]
    {growth_txt}

    [VACCINATION STATUS]
    {vax_txt}

    [CONTEXT]
    - Location: India 🇮🇳 (Emergency: 112/102).
    - Culture: Respect Indian norms.

    RULES:
    1. ANALYZE the medical file. If asked about growth, look at the specific numbers in [GROWTH HISTORY].
    2. MONITOR VACCINES. If {baby_name} has [OVERDUE] vaccines, gently remind the parent to schedule them.
    3. ONLY answer pediatric questions.
    4. Be warm, nurturing, and concise.
    
    CHAT HISTORY:
    {history_text}
    
    User Query: {user_msg}
    """

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(system_prompt)
        ai_text = response.text.replace("*", "") 

        session['chat_history'].append({"role": "User", "text": user_msg})
        session['chat_history'].append({"role": "Dr. Ycry", "text": ai_text})
        session.modified = True
        
        return jsonify({"response": ai_text})

    except Exception as e:
        print(f"Gemini Error: {e}")
        return jsonify({"error": "AI Brain is tired."}), 500

# --- NEW HELPERS ---
def get_nutrition_guide(age):
    if age < 6:
        return {
            "title": "0-6 Months: Milk Only",
            "allowed": ["Breast Milk", "Infant Formula"],
            "avoid": ["Water", "Honey", "Cow's Milk", "Solid Food"],
            "schedule": "On demand (every 2-3 hours)"
        }
    elif age < 8:
        return {
            "title": "6-8 Months: First Tastes",
            "allowed": ["Pureed Vegetables (Carrot, Sweet Potato)", "Iron-fortified Cereal", "Mashed Banana"],
            "avoid": ["Honey", "Salt", "Sugar", "Whole Nuts"],
            "schedule": "Milk + 1-2 small meals"
        }
    elif age < 10:
        return {
            "title": "8-10 Months: Textured Food",
            "allowed": ["Mashed Fruits", "Soft Cooked Pasta", "Yogurt", "Scrambled Egg Yolk"],
            "avoid": ["Honey", "Raw Apple slices (choking hazard)"],
            "schedule": "Milk + 2-3 meals"
        }
    else:
        return {
            "title": "10-12+ Months: Table Food",
            "allowed": ["Finger Foods", "Small pieces of Chicken", "Cheese", "Most Family Foods"],
            "avoid": ["Honey (until 1yr)", "Large chunks"],
            "schedule": "3 meals + 2 snacks"
        }

def get_exercises(age):
    if age < 3:
        exercises = [{"name": "Tummy Time", "desc": "Place baby on stomach while awake.", "benefit": "Strengthens neck & shoulders", "video_id": "bq0S_nulAyk"}]
        
        if age == 1:
            exercises.append({"name": "Leg Bicycle", "desc": "Gently cycle baby's legs towards tummy.", "benefit": "Relieves gas & constipations", "video_id": "nzix4pZtdXs"})
        else:
            exercises.append({"name": "Visual Tracking", "desc": "Move a toy slowly side-to-side.", "benefit": "Improves eye coordination", "video_id": "k3Y0f24aI74"})
            
        return exercises
    elif age < 6:
        return [
            {"name": "Supported Sit", "desc": "Prop baby up with pillows.", "benefit": "Core strength", "video_id": "_WwlTvU1DOs"},
            {"name": "Reach & Grab", "desc": "Hold toy just out of reach.", "benefit": "Hand-eye coordination", "video_id": "p4vW9K2E138"}
        ]
    elif age < 9:
        return [
            {"name": "Peek-a-Boo", "desc": "Hide face behind hands/cloth.", "benefit": "Object permanence", "video_id": "8PIGK9l8K1c"}, 
            {"name": "Obstacle Course", "desc": "Pillows on floor to crawl over.", "benefit": "Motor skills", "video_id": "T050VqC69Qk"}
        ]
    else: # 9-12+
        return [
            {"name": "Cruising", "desc": "Place toys on sofa to encourage standing.", "benefit": "Leg strength for walking", "video_id": "T050VqC69Qk"},
            {"name": "Stacking Blocks", "desc": "Build simple towers.", "benefit": "Fine motor skills", "video_id": "8PIGK9l8K1c"}
        ]

def get_warning_signs():
    return [
        {"symptom": "High Fever", "desc": "Rectal temp > 100.4°F (38°C) if < 3 months.", "action": "Call Doctor Immediately"},
        {"symptom": "Dehydration", "desc": "No wet diaper for 6+ hours, dry lips.", "action": "Hydrate & Seek Help"},
        {"symptom": "Breathing Trouble", "desc": "Fast breathing, ribs sucking in.", "action": "Emergency Room"},
        {"symptom": "Persistent Vomiting", "desc": "Vomiting for more than 12 hours.", "action": "Consult Pediatrician"},
        {"symptom": "Unusual Rash", "desc": "Rash that doesn't fade when pressed.", "action": "Urgent Care"}
    ]

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, threaded=True)