import datetime
import hmac
import bcrypt
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import os
import certifi
from dotenv import load_dotenv

load_dotenv()
# CONFIG
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    print("⚠️ WARNING: MONGO_URI not found in .env file!")
DB_NAME = "ycry"

client = None
db = None

def init_db():
    global client, db
    try:
        # Set timeout to 20 seconds to allow for slower networks
        client = MongoClient(MONGO_URI, 
                             tlsCAFile=certifi.where(), 
                             serverSelectionTimeoutMS=20000,
                             connectTimeoutMS=20000)
        db = client[DB_NAME]
        # Quick check
        client.admin.command('ping')
        print("✅ Connected to MongoDB Atlas!")
    except Exception as e:
        print(f"❌ MongoDB Connection Failed: {e}")

# --- USER FUNCTIONS ---
# --- HELPERS ---
def get_next_sequence(name):
    """Generates the next unique numeric ID"""
    if db is None: init_db()
    ret = db.counters.find_one_and_update(
        {"_id": name},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=True
    )
    return ret['seq']

# --- USER FUNCTIONS ---
def create_user(username, password, name, email):
    if db is None: init_db()
    users = db.users
    
    # Check if username exists (now querying 'username' field)
    # Also check if email exists to be safe
    if users.find_one({"username": username}):
        return False
        
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    # Generate Numeric ID
    user_id = get_next_sequence("userid")
    
    user_doc = {
        "_id": user_id,
        "username": username,
        "password": hashed_pw,
        "caregiver_name": name,
        "email": email,
        "profile": {},
        "growth": [],
        "vaccines": [],
        "food_log": [],
        "medical_id": {}
    }
    
    try:
        users.insert_one(user_doc)
        return True
    except Exception as e:
        print(f"Create User Error: {e}")
        return False

def delete_user(username):
    if db is None: init_db()
    try:
        # Find user
        user = db.users.find_one({"username": username})
        if not user:
            return False, None
            
        # Delete user
        db.users.delete_one({"username": username})
        
        # Get profile pic if exists
        pic = None
        if user.get('profile') and user['profile'].get('profile_pic'):
            pic = user['profile']['profile_pic']
            
        return True, pic
    except Exception as e:
        print(f"Delete User Error: {e}")
        return False, None

def login_user(username, password):
    if db is None: init_db()
    try:
        user = db.users.find_one({"username": username})
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            # Return tuple: (Name, is_admin)
            return user['caregiver_name'], user.get('is_admin', False)
    except ServerSelectionTimeoutError:
        print("❌ DB Timeout during login")
        return "DB_ERROR", False
    except Exception as e:
        print(f"Login Error: {e}")
    return None, False

def get_all_users():
    if db is None: init_db()
    try:
        # Return list of user documents with just relevant fields
        return list(db.users.find({}, {"username": 1, "caregiver_name": 1, "is_admin": 1}))
    except Exception as e:
        print(f"Get All Users Error: {e}")
        return []

def make_admin(username):
    if db is None: init_db()
    try:
        db.users.update_one({"username": username}, {"$set": {"is_admin": True}})
        return True
    except Exception as e:
        print(f"Make Admin Error: {e}")
        return False

def get_user_by_email(email):
    if db is None: init_db()
    try:
        return db.users.find_one({"email": email})
    except Exception as e:
        print(f"Get User by Email Error: {e}")
        return None

def save_reset_otp(username, otp):
    if db is None: init_db()
    try:
        # OTP expires in 15 minutes
        expiry = datetime.datetime.now() + datetime.timedelta(minutes=15)
        # Note: We update by username now
        db.users.update_one(
            {"username": username},
            {"$set": {"reset_otp": otp, "reset_otp_expiry": expiry}}
        )
        return True
    except Exception as e:
        print(f"Save Reset OTP Error: {e}")
        return False

def verify_reset_otp(username, otp):
    if db is None: init_db()
    try:
        user = db.users.find_one({"username": username})
        if not user:
            return False
        
        saved_otp = user.get("reset_otp")
        expiry = user.get("reset_otp_expiry")

        if saved_otp and hmac.compare_digest(str(saved_otp), str(otp)) and expiry and expiry > datetime.datetime.now():
            return True
        return False
    except Exception as e:
        print(f"Verify Reset OTP Error: {e}")
        return False

def update_password(username, new_password):
    if db is None: init_db()
    try:
        hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        db.users.update_one(
            {"username": username},
            {"$set": {"password": hashed_pw}, "$unset": {"reset_otp": "", "reset_otp_expiry": ""}}
        )
        return True
    except Exception as e:
        print(f"Update Password Error: {e}")
        return False

# --- PROFILE FUNCTIONS ---
def save_full_profile(username, data):
    """
    data = (name, dob, gender, blood, weight, height, head, chest, apgar, summary, pic_path)
    We will store this as a dict in MongoDB.
    """
    if db is None: init_db()
    
    # Map tuple to meaningful keys
    profile_doc = {
        "baby_name": data[0],
        "dob": data[1],
        "gender": data[2],
        "blood_group": data[3],
        "weight_birth": data[4],
        "height_birth": data[5],
        "head_circ": data[6],
        "chest_circ": data[7],
        "apgar_score": data[8],
        "health_summary": data[9],
        "profile_pic": data[10]
    }
    
    db.users.update_one(
        {"username": username},
        {"$set": {"profile": profile_doc}}
    )

def get_profile(username):
    if db is None: init_db()
    user = db.users.find_one({"username": username})
    if user and user.get('profile'):
        return user['profile']
    return None

# --- GROWTH & VACCINE FUNCTIONS ---
def add_growth_record(username, weight, height, date_str=None):
    if db is None: init_db()
    today = date_str if date_str else str(datetime.date.today())
    
    # Check if a record for today already exists
    query = {"username": username, "growth.date": today}
    update = {"$set": {"growth.$.weight": weight, "growth.$.height": height}}
    
    result = db.users.update_one(query, update)
    
    # If no record matched (result.matched_count == 0), push a new one
    if result.matched_count == 0:
        record = {
            "date": today,
            "weight": weight,
            "height": height
        }
        db.users.update_one(
            {"username": username},
            {"$push": {"growth": record}}
        )

def get_latest_growth(username):
    if db is None: init_db()
    user = db.users.find_one({"username": username})
    if user and user.get('growth'):
        latest = user['growth'][-1]
        return {"weight": latest['weight'], "height": latest['height']}
    return None

def get_growth_history(username):
    if db is None: init_db()
    user = db.users.find_one({"username": username})
    if user and user.get('growth'):
        return user['growth']
    return []

def mark_vaccine_done(username, vaccine_name):
    if db is None: init_db()
    today = str(datetime.date.today())
    
    # Check if already exists to avoid duplicates (idempotent)
    user = db.users.find_one({"username": username, "vaccines.name": vaccine_name})
    if not user:
        db.users.update_one(
            {"username": username},
            {"$push": {"vaccines": {"name": vaccine_name, "date": today}}}
        )

def get_completed_vaccines(username):
    if db is None: init_db()
    user = db.users.find_one({"username": username})
    if user and user.get('vaccines'):
        return [v['name'] for v in user['vaccines']]
    return []

# --- MEDICAL ID FUNCTIONS ---
def save_medical_id(username, data):
    """
    data = {"doctor_name": ..., "doctor_phone": ..., "insurance_provider": ..., "policy_number": ..., "allergies": ...}
    """
    if db is None: init_db()
    db.users.update_one(
        {"username": username},
        {"$set": {"medical_id": data}}
    )

def get_medical_id(username):
    if db is None: init_db()
    user = db.users.find_one({"username": username})
    return user.get('medical_id', {}) if user else {}

# --- FOOD LOG FUNCTIONS ---
def add_food_log(username, food, reaction):
    if db is None: init_db()
    today = str(datetime.date.today())
    
    log_entry = {
        "food": food,
        "reaction": reaction,
        "date": today
    }
    
    # 1. Remove existing if any (Overwrite logic)
    db.users.update_one(
        {"username": username},
        {"$pull": {"food_log": {"food": food}}}
    )
    
    # 2. Add new entry at top
    db.users.update_one(
        {"username": username},
        {"$push": {"food_log": {"$each": [log_entry], "$position": 0}}}
    )

    # 3. AUTO-SYNC: If Allergy, add to Medical ID
    if "Allergy" in reaction:
        user = db.users.find_one({"username": username})
        mid = user.get('medical_id', {})
        current_allergies = mid.get('allergies', '')
        
        # Avoid duplicate appending
        if food.lower() not in current_allergies.lower():
            new_allergies = f"{current_allergies}, {food}" if current_allergies else food
            mid['allergies'] = new_allergies
            
            db.users.update_one(
                {"username": username},
                {"$set": {"medical_id": mid}}
            )

def remove_food_log(username, food):
    if db is None: init_db()
    db.users.update_one(
        {"username": username},
        {"$pull": {"food_log": {"food": food}}}
    )

def get_food_log(username):
    if db is None: init_db()
    user = db.users.find_one({"username": username})
    return user.get('food_log', []) if user else []