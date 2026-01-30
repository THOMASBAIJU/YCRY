import pymongo
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure
import certifi
import ssl

MONGO_URI = "mongodb+srv://thomasbaiju02_db_user:abm95KG2rEuItPWP@cluster0.9iivpdx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

def try_connect(name, kwargs):
    print(f"\n🧪 Testing Strategy: {name}")
    try:
        client = pymongo.MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            **kwargs
        )
        client.admin.command('ping')
        print(f"✅ SUCCESS with {name}!")
        return True
    except Exception as e:
        print(f"❌ FAILED with {name}")
        print(f"   Error: {e}")
        return False

def run_tests():
    print("🚀 Starting Connectivity Diagnostics...")
    
    # 1. Original Strategy (as in database.py)
    try_connect("Original (Certifi + Allow Invalid)", {
        "tlsCAFile": certifi.where(),
        "tlsAllowInvalidCertificates": True
    })

    # 2. No Certifi (System Certs)
    try_connect("System Certs (No Certifi)", {
        "tlsAllowInvalidCertificates": True
    })

    # 3. Strict SSL (No Allow Invalid)
    try_connect("Strict SSL (Standard)", {
        "tlsCAFile": certifi.where()
    })
    
    # 4. No SSL checks (last resort test, often rejected by Atlas but good to test)
    try_connect("Insecure (No SSL Verify)", {
        "tls": True,
        "tlsAllowInvalidCertificates": True,
        "tlsAllowInvalidHostnames": True
    })

if __name__ == "__main__":
    run_tests()
