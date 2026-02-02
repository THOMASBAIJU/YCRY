import database as db
import bcrypt

def setup_admin():
    username = "admin"
    password = "123456"
    name = "System Admin"
    
    print(f"--- 🛠️ SETTING UP ADMIN USER: {username} ---")
    
    if db.db is None: db.init_db()
    
    # 1. Check if user exists
    user = db.db.users.find_one({"_id": username})
    
    if user:
        print(f"ℹ️ User '{username}' already exists.")
    else:
        print(f"Creating new user '{username}'...")
        # Manually inserting to ensure it works even if create_user logic changes
        # But allow using the standard function for hashing consistency if possible
        # actually, I'll just use the db functions I already made to be safe
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        user_doc = {
            "_id": username,
            "password": hashed_pw,
            "caregiver_name": name,
            "profile": {}, # Empty profile
            "growth": [],
            "vaccines": [],
            "is_admin": True # DIRECTLY SET ADMIN HERE
        }
        
        try:
            db.db.users.insert_one(user_doc)
            print(f"✅ User '{username}' created successfully.")
            return
        except Exception as e:
            print(f"❌ Failed to create user: {e}")
            return

    # 2. Ensure Admin Status
    print("Promoting to Admin...")
    if db.make_admin(username):
        print(f"👑 SUCCESS! User '{username}' is now an Administrator.")
    else:
        print("❌ Failed to set admin status.")

if __name__ == "__main__":
    setup_admin()
