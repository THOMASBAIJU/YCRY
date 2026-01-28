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
MODEL_PATH = "model_brain.h5"
ai_model = None
try:
    if os.path.exists(MODEL_PATH):
        ai_model = load_model(MODEL_PATH)
        print("✅ AI Model Loaded")
except Exception as e:
    print(f"❌ Model Error: {e}")

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
        if profile and profile[11]:
            return {'navbar_pic': profile[11]}
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
        baby_name = profile[1]
        dob_str = profile[2]
        health_summary = profile[10]
        pic = profile[11] if profile[11] else None
        
        age = calculate_age(dob_str)
        growth = db.get_latest_growth(session['user'])
        weight = f"{growth[0]} kg" if growth else f"{profile[5]} kg (Birth)"
        
        return render_template('home.html', logged_in=True, 
                               name=baby_name, age=age, weight=weight, 
                               health_summary=health_summary, profile_pic=pic)
    
    # Fallback if profile missing (should rarely happen)
    return redirect(url_for('profile'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = db.login_user(username, password)
        if name:
            session['user'] = username
            session['real_name'] = name
            return redirect(url_for('home'))
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
        pic_filename = old_profile[11] 

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
    pred, conf, advice = None, 0, ""
    if request.method == 'POST' and 'audio' in request.files:
        f = request.files['audio']
        if f.filename:
            path = os.path.join(app.config['UPLOAD_FOLDER'], "temp.wav")
            f.save(path)
            try:
                # Clear any existing plots
                plt.close('all')
                
                # Load audio
                try:
                    y, sr = librosa.load(path, sr=22050, duration=5)
                except Exception as librosa_error:
                    print(f"❌ Librosa Load Error: {librosa_error}")
                    return {"error": "Error reading audio file. Please try a different format (WAV/MP3)."}, 400

                fig = plt.figure(figsize=(4, 4))
                librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max), sr=sr)
                plt.axis('off')
                
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_spec.png")
                plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                plt.close('all')

                img = image.img_to_array(image.load_img(img_path, target_size=(64, 64))) / 255.0
                
                if ai_model:
                    probs = ai_model.predict(np.expand_dims(img, axis=0))[0]
                    classes = ["Burping", "Discomfort", "Hunger", "Pain", "Tired"]
                    # Explicit type casting for JSON serialization
                    pred = str(classes[np.argmax(probs)])
                    conf_val = float(np.max(probs) * 100)
                    conf = f"{conf_val:.1f}"
                    advice = {"Hunger": "Feed baby", "Pain": "Check injury", "Burping": "Burp baby", "Discomfort": "Check diaper", "Tired": "Sleep time"}.get(pred, "Check baby")
                    
                    # RETURN JSON
                    return jsonify({
                        "prediction": pred,
                        "confidence": conf,
                        "advice": advice,
                        "success": True
                    })
                else:
                    return jsonify({"error": "AI Model not loaded."}), 500

            except Exception as e:
                print(f"❌ Cry Analysis Error: {e}")
                return jsonify({"error": str(e)}), 500
            finally:
                plt.close('all')
    
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
    schedule, show_warning = get_vaccine_schedule(prof[2], completed)
    return render_template('vaccine.html', schedule=schedule, baby_name=prof[1], warning=show_warning)

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
    age = calculate_age(prof[2])
    
    # 2. Get Mental/Motor Milestones
    milestones = get_milestones(age)
    
    # 3. Get Growth Chart Data
    history = db.get_growth_history(session['user'])
    # history is list of (date, weight, height)
    chart_dates = [r[0] for r in history] if history else []
    chart_weights = [r[1] for r in history] if history else []
    chart_heights = [r[2] for r in history] if history else []

    # If no history, add birth data if available
    if not history and prof:
        chart_dates = [prof[2]]
        chart_weights = [prof[5]]
        chart_heights = [prof[6]]
    
    return render_template('growth.html', 
                           latest=latest, 
                           baby_name=prof[1], 
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
    
    age_months = calculate_age(prof[2])
    guide = get_nutrition_guide(age_months)
    return render_template('nutrition.html', guide=guide, age=age_months, baby_name=prof[1])

@app.route('/exercises')
def exercises():
    if 'user' not in session: return redirect(url_for('login'))
    prof = db.get_profile(session['user'])
    if not prof: return redirect(url_for('profile'))
    
    age_months = calculate_age(prof[2])
    exercises_list = get_exercises(age_months)
    return render_template('exercises.html', exercises=exercises_list, age=age_months, baby_name=prof[1])

@app.route('/health')
def health():
    if 'user' not in session: return redirect(url_for('login'))
    prof = db.get_profile(session['user'])
    if not prof: return redirect(url_for('profile'))
    
    warnings = get_warning_signs()
    return render_template('health.html', warnings=warnings, baby_name=prof[1])

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
        return [
            {"name": "Tummy Time", "desc": "Place baby on stomach while awake.", "benefit": "Strengthens neck & shoulders", "video_id": "bq0S_nulAyk"},
            {"name": "Visual Tracking", "desc": "Move a toy slowly side-to-side.", "benefit": "Improves eye coordination", "video_id": "k3Y0f24aI74"}
        ]
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
    app.run(debug=True, threaded=False)