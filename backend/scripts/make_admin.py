# backend/scripts/make_admin.py
import os
from pymongo import MongoClient

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/cardamom_db")
client = MongoClient(MONGODB_URI)
users = client["cardamom_db"]["users"]

all_users = list(users.find({}, {"_id": 1, "username": 1, "email": 1, "is_admin": 1}))

if not all_users:
    print("No users found.")
    exit()

print("\n=== CURRENT USERS ===")
for u in all_users:
    admin = " (ADMIN)" if u.get("is_admin") else ""
    print(f"  ID:{u['_id']}  {u.get('username','?')}  {u.get('email','?')}{admin}")

print("\nEnter the email of the user to make admin:")
email = input("Email: ").strip()
user = users.find_one({"email": email})

if not user:
    print(f"User '{email}' not found.")
else:
    users.update_one({"email": email}, {"$set": {"is_admin": True}})
    print(f"\n✅ {user.get('username', email)} is now an admin!")

client.close()