import os
from dotenv import load_dotenv
load_dotenv() # Load env vars from .env file
import datetime
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_mail import Mail, Message
import random
import uuid
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import database as db
from werkzeug.utils import secure_filename
import google.generativeai as genai
import threading
import json
app = Flask(__name__)
app.secret_key = os.urandom(24) # Random key = Logout on server restart
# Reload Trigger: 2

# --- STATIC RECIPES ---
STATIC_RECIPES = {
    "Pureed Vegetables (Carrot, Sweet Potato)": {
        "title": "Simple Veggie Puree",
        "ingredients": ["1 cup washed & peeled vegetables (carrot/sweet potato)", "Water for steaming", "Breast milk or formula (optional)"],
        "steps": ["Chop vegetables into small chunks.", "Steam or boil until very soft (15-20 mins).", "Mash or blend with a little water/milk until smooth.", "Cool before serving."]
    },
    "Iron-fortified Cereal": {
        "title": "Baby's First Cereal",
        "ingredients": ["1 tbsp Iron-fortified baby cereal", "4-5 tbsp Breast milk, formula, or water"],
        "steps": ["Mix cereal and liquid in a bowl.", "Stir until smooth.", "Adjust liquid to get a runny texture for first-timers.", "Serve immediately."]
    },
    "Mashed Banana": {
        "title": "Banana Mash",
        "ingredients": ["1/2 ripe Banana"],
        "steps": ["Peel the banana.", "Mash thoroughly with a fork until no lumps remain.", "Add a little milk if needed to thin it out.", "Serve fresh."]
    },
    "Mashed Fruits": {
        "title": "Stewed Fruit Mash",
        "ingredients": ["1 Apple or Pear (peeled & cored)", "Water"],
        "steps": ["Chop fruit into small pieces.", "Steam for 10-15 mins until soft.", "Mash or puree until smooth.", "Let it cool completely."]
    },
    "Soft Cooked Pasta": {
        "title": "First Pasta",
        "ingredients": ["Small handful of pasta shapes", "Tomato sauce (optional, low salt)"],
        "steps": ["Boil pasta until very soft (overcooked is better).", "Drain and let cool.", "Chop into tiny pieces if large.", "Mix with a little sauce or olive oil."]
    },
    "Yogurt": {
        "title": "Plain Yogurt Bowl",
        "ingredients": ["Plain whole-milk yogurt (unsweetened)"],
        "steps": ["Spoon yogurt into a bowl.", "Optional: Mix in a little fruit puree.", "Serve chilled."]
    },
    "Scrambled Egg Yolk": {
        "title": "Soft Egg Yolk",
        "ingredients": ["1 Egg (large)"],
        "steps": ["Hard boil the egg (10-12 mins).", "Peel and remove the yolk.", "Mash the yolk with a little milk or water.", "Serve warm."]
    },
    "Finger Foods": {
        "title": "Basic Finger Foods",
        "ingredients": ["Banana chunks, Avocados slices, or Steamed Carrot sticks"],
        "steps": ["Cut food into stick shapes (size of an adult pinky finger).", "Ensure it is soft enough to squish between fingers.", "Place on high chair tray for baby to grab."]
    },
    "Small pieces of Chicken": {
        "title": "Poached Chicken",
        "ingredients": ["Small chicken breast fillet"],
        "steps": ["Poach chicken in simmering water for 15-20 mins.", "Ensure it is fully cooked (white all through).", "Shred into very small, swallowable pieces.", "Moisten with broth."]
    },
    "Cheese": {
        "title": "Cheese Strips",
        "ingredients": ["Mild Cheddar or Mozzarella"],
        "steps": ["Cut cheese into thin strips.", "Serve as a finger food."]
    },
    "Most Family Foods": {
        "title": "Family Meal Adaptation",
        "ingredients": ["Your regular family meal (low salt/sugar)"],
        "steps": ["Take a portion of the family meal.", "Chop or mash to appropriate texture.", "Ensure no choking hazards (nuts, grapes).", "Serve warm."]
    }
}

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
from dotenv import load_dotenv
load_dotenv() # Load environment variables

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not found in .env file!")
    
genai.configure(api_key=GEMINI_API_KEY)

# --- MAIL CONFIG ---
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS') == 'True'
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')
mail = Mail(app)

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
            "raw_date": due.strftime("%Y-%m-%d"), # For Calendar Link
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
        
        name, is_admin = db.login_user(username, password)
        
        if name == "DB_ERROR":
            flash("⚠️ Database connection failed. Please check your internet or whitelist your IP.")
        elif name:
            session['user'] = username
            session['real_name'] = name
            session['is_admin'] = is_admin
            
            if is_admin:
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('home'))
        else:
            flash("Invalid Username or Password")
    return render_template('login.html')

@app.route('/admin')
def admin_dashboard():
    if 'user' not in session: return redirect(url_for('login'))
    if not session.get('is_admin'):
        flash("⛔ Access Denied: Admin privileges required.")
        return redirect(url_for('home'))
        
    users = db.get_all_users()
    return render_template('admin.html', users=users)

@app.route('/admin/delete/<username>', methods=['POST'])
def admin_delete_user(username):
    if 'user' not in session or not session.get('is_admin'):
         return jsonify({"success": False, "message": "Unauthorized"}), 401 # Changed from 403 to avoid redirect loop issues in some clients, but 403 is correct. 401 is easier to handle.
         
    if username == session['user']:
        return jsonify({"success": False, "message": "You cannot delete yourself!"}), 400
        
    success, pic = db.delete_user(username)
    if success:
        if pic:
            try:
                os.remove(os.path.join(app.config['PROFILE_FOLDER'], pic))
            except: pass
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Delete failed"}), 500

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            user = request.form['username']
            pw = request.form['password']
            caregiver_name = request.form['caregiver_name']
            email = request.form['email']
            
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
            
            if db.create_user(user, pw, caregiver_name, email):
                # 1. Save Profile
                db.save_full_profile(user, (b_name, dob, gender, blood, weight, height, head, chest, apgar, report, pic_filename))
                
                # 2. Save Initial Growth Record (Backdated to DOB)
                db.add_growth_record(user, weight, height, date_str=dob)

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
        try:
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
            
            return jsonify({"success": True, "message": "✅ Profile & Picture Updated!"})
        except Exception as e:
            return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

    data = db.get_profile(session['user'])
    return render_template('profile.html', profile=data)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/cry', methods=['GET', 'POST'])
def cry():
    # Handle Session Expiry for AJAX
    if 'user' not in session:
        if request.method == 'POST':
             return jsonify({"error": "Session expired. Please login again.", "redirect": url_for('login')}), 401
        return redirect(url_for('login'))
    
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
    # Handle Session Expiry for AJAX
    if 'user' not in session:
        if request.method == 'POST':
             return jsonify({"error": "Session expired.", "redirect": url_for('login')}), 401
        return redirect(url_for('login'))
    prof = db.get_profile(session['user'])
    if not prof: return redirect(url_for('profile'))
    
    if request.method == 'POST':
        # AJAX Request Handler
        vaccine_name = request.form.get('vaccine_name')
        if vaccine_name:
            db.mark_vaccine_done(session['user'], vaccine_name)
            return jsonify({"success": True, "vaccine": vaccine_name})
        return jsonify({"success": False, "error": "No vaccine name provided"}), 400

    completed = db.get_completed_vaccines(session['user'])
    schedule, show_warning = get_vaccine_schedule(prof['dob'], completed)
    return render_template('vaccine.html', schedule=schedule, baby_name=prof['baby_name'], warning=show_warning)

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = db.get_user_by_email(email)
        
        if user:
            # Generate 6-digit OTP
            otp = str(random.randint(100000, 999999))
            
            # Use 'username' field if it exists, else fallback to '_id' (legacy support)
            target_username = user.get('username', user['_id'])
            
            db.save_reset_otp(target_username, otp)
            
            # Send Email
            msg = Message("Password Reset Code", recipients=[email])
            msg.body = f"Your Verification Code is: {otp}\n\nThis code expires in 15 minutes."
            
            try:
                mail.send(msg)
                # Store username in session to know who is resetting
                session['reset_user_id'] = target_username
                flash("Code sent to your email!")
                return redirect(url_for('verify_otp'))
            except Exception as e:
                print(f"Mail Error: {e}")
                flash("Error sending email. Please try again.")
        else:
            flash("Email not found.")
            
    return render_template('forgot_password.html')

@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    if 'reset_user_id' not in session:
        return redirect(url_for('forgot_password'))
        
    if request.method == 'POST':
        otp = request.form.get('otp')
        username = session['reset_user_id']
        
        if db.verify_reset_otp(username, otp):
            session['verified_for_reset'] = True
            return redirect(url_for('reset_password'))
        else:
            flash("Invalid or expired code.")
            
    return render_template('verify_otp.html')

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    # Security check: Must have passed OTP verification
    if not session.get('verified_for_reset'):
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        new_password = request.form.get('password')
        username = session.get('reset_user_id')
        
        if len(new_password) < 6:
            flash("Password must be at least 6 characters.")
        else:
            db.update_password(username, new_password)
            # Clear reset session vars
            session.pop('reset_user_id', None)
            session.pop('verified_for_reset', None)
            
            flash("Password updated! Please login.")
            return redirect(url_for('login'))
            
    return render_template('reset_password.html')

@app.route('/growth', methods=['GET', 'POST'])
def growth():
    # Handle Session Expiry for AJAX
    if 'user' not in session:
        if request.method == 'POST':
             return jsonify({"error": "Session expired.", "redirect": url_for('login')}), 401
        return redirect(url_for('login'))
    
    prof = db.get_profile(session['user'])
    if not prof: return redirect(url_for('profile'))
    
    # 1. Handle Physical Growth (Weight/Height)
    if request.method == 'POST':
        try:
            # Handle JSON or Form Data
            w = 0
            h = 0
            
            if request.is_json:
                data = request.get_json(silent=True) or {}
                w = float(data.get('weight', 0))
                h = float(data.get('height', 0))
            
            # Fallback to form data if JSON didn't provide values
            if w == 0 and h == 0:
                w = float(request.form.get('weight', 0))
                h = float(request.form.get('height', 0))

            if w > 0 and h > 0:
                print(f"Adding Growth: {w}kg, {h}cm")
                db.add_growth_record(session['user'], w, h)
                msg = "✅ Physical growth recorded!"
                success = True
            else:
                msg = "❌ Invalid values."
                success = False
                
            # If AJAX request, return JSON
            if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                # Get updated history for chart
                history = db.get_growth_history(session['user'])
                dates = [r['date'] for r in history]
                weights = [r['weight'] for r in history]
                heights = [r['height'] for r in history]
                
                return jsonify({
                    "success": success,
                    "message": msg,
                    "latest": [w, h, str(datetime.date.today())] if success else None,
                    "dates": dates,
                    "weights": weights,
                    "heights": heights
                })
                
            flash(msg)

        except Exception as e:
            print(f"Growth Error: {e}")
            if request.is_json: return jsonify({"success": False, "message": str(e)}), 500
            flash("❌ Error saving record.")
        
        return redirect(url_for('growth'))
        
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
    food_log = db.get_food_log(session['user'])
    
    return render_template('nutrition.html', 
                          guide=guide, 
                          age=age_months, 
                          baby_name=prof['baby_name'],
                          food_log=food_log,
                          recipes=STATIC_RECIPES)

@app.route('/add_food_log', methods=['POST'])
def add_food_log():
    if 'user' not in session: return jsonify({'success': False, 'error': 'Login required'}), 401
    
    food = request.form.get('food')
    reaction = request.form.get('reaction')
    
    if food and reaction:
        db.add_food_log(session['user'], food, reaction)
        return jsonify({
            'success': True,
            'food': food,
            'reaction': reaction,
            'date': str(datetime.date.today())
        })
    else:
        return jsonify({'success': False, 'error': 'Missing information.'}), 400

@app.route('/remove_food_log', methods=['POST'])
def remove_food_log():
    if 'user' not in session: return jsonify({'success': False, 'error': 'Login required'}), 401
    
    food = request.form.get('food')
    if food:
        db.remove_food_log(session['user'], food)
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Missing food name'}), 400

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
    medical_id = db.get_medical_id(session['user'])
    age_months = calculate_age(prof['dob'])
    
    # Get latest logged weight, fallback to birth weight
    latest_growth = db.get_latest_growth(session['user'])
    current_weight = latest_growth['weight'] if latest_growth else prof['weight_birth']
    
    return render_template('health.html', warnings=warnings, 
                          baby_name=prof['baby_name'], 
                          profile=prof,
                          mid=medical_id,
                          age=age_months,
                          weight=current_weight)

@app.route('/update_medical_id', methods=['POST'])
def update_medical_id():
    if 'user' not in session: return jsonify({'success': False, 'error': 'Login required'}), 401
    
    data = {
        "doctor_name": request.form.get('doctor_name'),
        "doctor_phone": request.form.get('doctor_phone'),
        "insurance_provider": request.form.get('insurance_provider'),
        "policy_number": request.form.get('policy_number'),
        "allergies": request.form.get('allergies')
    }
    
    db.save_medical_id(session['user'], data)
    return jsonify({'success': True})

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
    You are Dr. Ycry, a warm, caring, and highly experienced pediatrician who loves helping parents.
    You are caring for a baby named {baby_name}, who is {age_months} months old.
    
    === MEDICAL FILE ===
    [GROWTH HISTORY]
    {growth_txt}

    [VACCINATION STATUS]
    {vax_txt}

    [CONTEXT]
    - Location: India 🇮🇳 (Emergency: 112/102).
    - Culture: Respect Indian norms. Be supportive of the family structure.

    RULES:
    1. BE HUMAN & WARM: Speak like a kind doctor, not a robot. Use phrases like "I understand," "That sounds tough," or "You're doing a great job."
    2. USE NAMES: Refer to the baby as {baby_name} naturally in conversation.
    3. ANALYZE GROWTH: If asked about weight/height, look at [GROWTH HISTORY] and give specific feedback.
    4. VACCINE CHECK: If {baby_name} has [OVERDUE] vaccines, gently and kindly remind the parent to schedule them soon for safety.
    5. SCOPE: ONLY answer pediatric/parenting questions.
    6. TONE: Supportive, reassuring, and concise (keep answers under 3-4 sentences unless detailed advice is needed).
    
    CHAT HISTORY:
    {history_text}
    
    User Query: {user_msg}
    """

    # List of models to try in order of priority (Prioritizing LITE for quota)
    print("🚀 Using Updated Model List (Lite)")
    models_to_try = [
        'gemini-2.0-flash-lite',          # Try Lite first (usually higher limits)
        'gemini-flash-latest',            # Generic alias
        'gemini-2.0-flash',               # Fancy one
        'gemini-2.5-flash-lite',          # Newer Lite
        'gemini-2.5-flash'                # Newer Fancy (likely capped)
    ]

    last_error = None

    for model_name in models_to_try:
        try:
            print(f"🤖 Trying AI Model: {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(system_prompt)
            ai_text = response.text.replace("*", "") 

            session['chat_history'].append({"role": "User", "text": user_msg})
            session['chat_history'].append({"role": "Dr. Ycry", "text": ai_text})
            session.modified = True
            
            print(f"✅ Success with {model_name}!")
            return jsonify({"response": ai_text})

        except Exception as e:
            print(f"⚠️ Failed with {model_name}: {e}")
            last_error = e
            continue # Try next model

    # If all models fail
    print(f"❌ All models failed. Last error: {last_error}")
    return jsonify({"error": f"AI Error after trying multiple models. Last error: {str(last_error)}"}), 500

@app.route('/api/analyze_growth', methods=['POST'])
def analyze_growth():
    if 'user' not in session: return jsonify({"error": "Login required"}), 401
    
    prof = db.get_profile(session['user'])
    if not prof: return jsonify({"error": "Profile not found"}), 400
    
    history = db.get_growth_history(session['user'])
    if not history: return jsonify({"analysis": "No growth records found yet. Add some measurements above!"})
    
    # Format data
    data_points = "\n".join([f"- {h['date']}: {h['weight']}kg, {h['height']}cm" for h in history])
    age_months = calculate_age(prof['dob'])
    
    prompt = f"""
    Act as a friendly pediatrician. 
    Child: {prof['baby_name']}, {age_months} months old.
    Gender: {prof['gender']}.
    Birth: {prof['weight_birth']}kg, {prof['height_birth']}cm.
    
    Growth History:
    {data_points}
    
    Analyze this growth trend. 
    1. Is the weight gain steady?
    2. Is the height increase normal?
    3. Calculate latest BMI if possible.
    4. Give 1 one-sentence tip for this age.
    
    Keep the tone encouraging. Max 4-5 sentences.
    """
    
    # Reuse the same model fallback logic
    # Reuse the same model fallback logic (Prioritizing LITE for quota)
    models_to_try = [
        'gemini-2.0-flash-lite',          # Try Lite first (usually higher limits)
        'gemini-flash-latest',            # Generic alias
        'gemini-2.0-flash',               # Fancy one
        'gemini-2.5-flash-lite',          # Newer Lite
        'gemini-2.5-flash'                # Newer Fancy (likely capped)


        
    ]
    
    last_error = None

    for model_name in models_to_try:
        try:
            print(f"🤖 [Analysis] Trying AI Model: {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            print(f"✅ [Analysis] Success with {model_name}!")
            return jsonify({"analysis": response.text.replace("*", "")})
        except Exception as e:
            print(f"⚠️ [Analysis] Failed with {model_name}: {e}")
            last_error = e
            continue
            
    print(f"❌ [Analysis] All models failed. Last error: {last_error}")
    return jsonify({"analysis": f"AI Service Currently Unavailable. (Error: {str(last_error)})"}), 500

@app.route('/api/nutrition_ai', methods=['POST'])
def nutrition_ai():
    if 'user' not in session: return jsonify({"error": "Login required"}), 401
    
    prof = db.get_profile(session['user'])
    if not prof: return jsonify({"error": "Profile not found"}), 400
    
    data = request.json
    action = data.get('action')
    query = data.get('query', '')
    age_months = calculate_age(prof['dob'])
    
    if action == 'check_safety':
        prompt = f"""
        Act as a pediatric nutritionist.
        Food: "{query}"
        Child Age: {age_months} months.
        
        Is this food safe? 
        1. Status: SAFE / CAUTION / UNSAFE / AVOID.
        2. Preparation: How to serve safely (e.g. mash, steam).
        3. Benefits: Key vitamin/nutrient (1-2 words).
        
        Format as JSON: {{ "status": "...", "prep": "...", "benefit": "..." }}
        """
        
    elif action == 'meal_plan':
        prompt = f"""
        Create a 1-day meal plan for a {age_months} month old baby.
        Culture: Indian/Global mix.
        Vegetarian: No (include eggs/chicken if age appropriate).
        
        Format as JSON array of objects: 
        [ {{ "meal": "Breakfast", "food": "...", "desc": "..." }}, ... ]
        """
        
    elif action == 'recipe':
        prompt = f"""
        Give a simple baby recipe for: "{query}".
        Age: {age_months} months.
        Keep it very simple (3-4 steps).
        Format as JSON: {{ "title": "...", "ingredients": ["..."], "steps": ["..."] }}
        """
    else:
        return jsonify({"error": "Invalid action"}), 400

    # --- ROBUST AI FALLBACK SYSTEM (Keys x Models) ---
    
    # 1. Gather all potential Google Keys from Environment
    potential_keys = []
    
    # Add standard keys (SUPPORT GEMINI_Naming too!)
    if os.getenv("GOOGLE_API_KEY"): potential_keys.append(os.getenv("GOOGLE_API_KEY"))
    if os.getenv("GEMINI_API_KEY"): potential_keys.append(os.getenv("GEMINI_API_KEY"))
    if os.getenv("NUTRITION_API_KEY"): potential_keys.append(os.getenv("NUTRITION_API_KEY"))
    
    # Filter for valid Google Keys (Start with AIza)
    valid_keys = [k for k in potential_keys if k and k.startswith("AIza")]
    
    # Deduplicate
    valid_keys = list(set(valid_keys))
    
    if not valid_keys:
        return jsonify({"error": "No valid Google API Keys found. Check .env for GEMINI_API_KEY"}), 500

    # 2. Define Models to Try
    models_to_try = [
        'gemini-2.0-flash-lite',
        'gemini-2.0-flash',
        'gemini-flash-latest',
        'gemini-1.5-flash',
        'gemini-pro'
    ]
    
    last_error = None
    
    # 3. Matrix Execution: Try Every Key with Every Model
    for key in valid_keys:
        genai.configure(api_key=key)
        
        for model_name in models_to_try:
            try:
                # print(f"🤖 Trying Key ending in ...{key[-4:]} with Model {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
                
                # If we get here, it worked!
                # Restore default key for other parts of app if needed, or leave it.
                # (Ideally usage is localized, but for global config we leave it as last working)
                return jsonify(json.loads(response.text))
                
            except Exception as e:
                # print(f"⚠️ Failed: Key ...{key[-4:]} | Model {model_name} | Error: {e}")
                last_error = e
                # Check if it's a key error (400/403) -> break inner loop to switch key
                if "API_KEY" in str(e) or "400" in str(e) or "403" in str(e):
                    break # Key is bad, try next key
                continue # Model might be bad, try next model with same key

    return jsonify({"error": f"AI unavailable. (Last Error: {str(last_error)})"}), 500

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
        exercises = [{"name": "Tummy Time", "desc": "Place baby on stomach while awake to build neck strength.", "benefit": "Strengthens neck & shoulders", "video_id": "bq0S_nulAyk"}]
        
        if age == 1:
            exercises.append({"name": "Leg Bicycle", "desc": "Gently cycle baby's legs towards tummy to relieve gas.", "benefit": "Relieves gas & constipations", "video_id": "nzix4pZtdXs"})
        else:
            exercises.append({"name": "Visual Tracking", "desc": "Move a high-contrast toy slowly side-to-side.", "benefit": "Improves eye coordination", "video_id": "eNl_cR4yM0c"})
            
        return exercises
    elif age < 6:
        return [
            {"name": "Rolling: Tummy to Back", "desc": "Gently guide baby from tummy to back using a toy to lead their head.", "benefit": "Core strength & Coordination", "video_id": "F81VylqnzGE"},
            {"name": "Tummy Time Play", "desc": "Use a mirror or high-contrast cards during tummy time.", "benefit": "Neck strength & reduces flat head", "video_id": "Q9oxTiUOXVY"}
        ]
    elif age < 9:
        return [
            {"name": "Peek-a-Boo", "desc": "Hide face behind hands or cloth to teach object permanence.", "benefit": "Cognitive development", "video_id": "lVFj91Z1AfM"}, 
            {"name": "Obstacle Course", "desc": "Use pillows on floor for baby to crawl over.", "benefit": "Motor skills", "video_id": "9T52OgZ6Rxg"}
        ]
    else: # 9-12+
        return [
            {"name": "Cruising", "desc": "Place toys on sofa to encourage standing and stepping sideways.", "benefit": "Leg strength for walking", "video_id": "-V8PSjYCyAs"},
            {"name": "Stacking Blocks", "desc": "Encourage baby to stack 2-3 blocks.", "benefit": "Fine motor skills", "video_id": "5sFRQRi7OMc"}
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=False)