import database as db
import sys

def create_admin_interactive():
    print("--- 🛡️ CREATE/PROMOTE ADMIN USER ---")
    
    username = input("Enter username to promote to Admin: ").strip()
    
    if not username:
        print("❌ Username cannot be empty.")
        return

    # Check if user exists
    if db.db is None: db.init_db()
    
    user = db.db.users.find_one({"_id": username})
    
    if user:
        print(f"✅ User '{username}' found.")
        if db.make_admin(username):
            print(f"👑 SUCCESS! User '{username}' is now an Administrator.")
        else:
            print("❌ Failed to update user.")
    else:
        print(f"⚠️ User '{username}' does not exist.")
        create_new = input("Do you want to create a NEW user with this name? (yes/no): ").lower().strip()
        
        if create_new == 'yes':
            password = input("Enter Password: ").strip()
            name = input("Enter Caregiver Name: ").strip()
            
            if len(password) < 6:
                print("❌ Password must be at least 6 chars.")
                return
                
            if db.create_user(username, password, name):
                print("✅ User created.")
                if db.make_admin(username):
                    print(f"👑 SUCCESS! User '{username}' is now an Administrator.")
                else:
                    print("❌ Failed to promote to admin.")
            else:
                print("❌ Failed to create user.")
        else:
            print("Operation cancelled.")

if __name__ == "__main__":
    create_admin_interactive()
