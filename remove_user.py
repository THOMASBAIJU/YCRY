import database as db
import os
import sys

def remove_user_interactive():
    print("--- REMOVE USER UTILITY ---")
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        username = input("Enter username to delete: ").strip()
    
    if not username:
        print("❌ No username provided.")
        return

    print(f"Are you sure you want to delete user '{username}'? This cannot be undone.")
    confirm = input("Type 'yes' to confirm: ").strip().lower()
    
    if confirm != 'yes':
        print("❌ Operation cancelled.")
        return

    print(f"Attempting to delete {username}...")
    success, pic_filename = db.delete_user(username)
    
    if success:
        print(f"✅ User '{username}' removed from database.")
        
        if pic_filename:
            pic_path = os.path.join("static", "profile_pics", pic_filename)
            if os.path.exists(pic_path):
                try:
                    os.remove(pic_path)
                    print(f"✅ Profile picture '{pic_filename}' deleted.")
                except Exception as e:
                    print(f"⚠️ Could not delete profile picture: {e}")
            else:
                print(f"ℹ️ Profile picture file not found at {pic_path}")
    else:
        print(f"❌ User '{username}' not found or database error.")

if __name__ == "__main__":
    remove_user_interactive()
