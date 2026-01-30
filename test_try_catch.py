import database
import sys
from pymongo.errors import ServerSelectionTimeoutError

print(f"Imported database from: {database.__file__}")

def test_login_crash():
    print("Testing login_user crash handling...")
    try:
        # This triggers the DB connection
        result = database.login_user("nonexistent_user", "password")
        print(f"Result: {result}")
        if result == "DB_ERROR":
            print("✅ Successfully caught DB handling!")
        else:
            print(f"❌ Failed: Expected 'DB_ERROR' but got {result}")
    except ServerSelectionTimeoutError:
        print("❌ CRITICAL: Exception caught OUTSIDE login_user! The try-block inside didn't work.")
    except Exception as e:
        print(f"❌ CRITICAL: Uncaught exception: {type(e)} - {e}")

if __name__ == "__main__":
    test_login_crash()
