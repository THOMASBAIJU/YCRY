import datetime
import bcrypt
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import os
import certifi

# CONFIG
MONGO_URI = "mongodb+srv://thomasbaiju02_db_user:abm95KG2rEuItPWP@cluster0.9iivpdx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
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
                             connectTimeoutMS=20000,
                             tlsAllowInvalidCertificates=True)
        db = client[DB_NAME]
        # Quick check
        client.admin.command('ping')
        print("✅ Connected to MongoDB Atlas!")
    except Exception as e:
        print(f"❌ MongoDB Connection Failed: {e}")

# --- USER FUNCTIONS ---
def create_user(username, password, name):
    if db is None: init_db()
    users = db.users
    
    if users.find_one({"_id": username}):
        return False
        
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    user_doc = {
        "_id": username,
        "password": hashed_pw,
        "caregiver_name": name,
        "profile": {},
        "growth": [],
        "vaccines": []
    }
    
    try:
        users.insert_one(user_doc)
        return True
    except Exception as e:
        print(f"Create User Error: {e}")
        return False

def login_user(username, password):
    if db is None: init_db()
    try:
        user = db.users.find_one({"_id": username})
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            return user['caregiver_name']
    except ServerSelectionTimeoutError:
        print("❌ DB Timeout during login")
        return "DB_ERROR"
    except Exception as e:
        print(f"Login Error: {e}")
    return None

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
        {"_id": username},
        {"$set": {"profile": profile_doc}}
    )

def get_profile(username):
    if db is None: init_db()
    user = db.users.find_one({"_id": username})
    if user and user.get('profile'):
        return user['profile']
    return None

# --- GROWTH & VACCINE FUNCTIONS ---
def add_growth_record(username, weight, height):
    if db is None: init_db()
    today = str(datetime.date.today())
    
    # Check if a record for today already exists
    query = {"_id": username, "growth.date": today}
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
            {"_id": username},
            {"$push": {"growth": record}}
        )

def get_latest_growth(username):
    if db is None: init_db()
    user = db.users.find_one({"_id": username})
    if user and user.get('growth'):
        latest = user['growth'][-1]
        return {"weight": latest['weight'], "height": latest['height']}
    return None

def get_growth_history(username):
    if db is None: init_db()
    user = db.users.find_one({"_id": username})
    if user and user.get('growth'):
        return user['growth']
    return []

def mark_vaccine_done(username, vaccine_name):
    if db is None: init_db()
    today = str(datetime.date.today())
    
    # Check if already exists to avoid duplicates (idempotent)
    user = db.users.find_one({"_id": username, "vaccines.name": vaccine_name})
    if not user:
        db.users.update_one(
            {"_id": username},
            {"$push": {"vaccines": {"name": vaccine_name, "date": today}}}
        )

def get_completed_vaccines(username):
    if db is None: init_db()
    user = db.users.find_one({"_id": username})
    if user and user.get('vaccines'):
        return [v['name'] for v in user['vaccines']]
    return []