import database as db
import pymongo

def reset_and_seed():
    print("⚠️  STARTING DATABASE RESET ⚠️")
    
    # Initialize connection
    db.init_db()
    
    print("🗑️  Dropping 'users' collection...")
    db.db.users.drop()
    
    print("🗑️  Dropping 'counters' collection...")
    db.db.counters.drop()
    
    print("✅ Collections dropped.")
    
    # Create Admin
    username = "admin"
    password = "123456"
    name = "System Administrator"
    email = "admin@ycry.com"
    
    print(f"👤 Creating Admin User: {username}")
    if db.create_user(username, password, name, email):
        # Manually promote to admin
        print("🔧 Promoting to Admin...")
        db.make_admin(username)
        print("✅ Success! Admin created.")
        print(f"👉 Username: {username}")
        print(f"👉 Password: {password}")
    else:
        print("❌ Failed to create admin user.")

if __name__ == "__main__":
    reset_and_seed()
