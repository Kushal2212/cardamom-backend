# backend/database/mongo.py
import os
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure

MONGODB_URI = os.environ.get(
    'MONGODB_URI',
    'mongodb://localhost:27017/cardamom_db'
)

try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print('✅ MongoDB connected')
except ConnectionFailure:
    print('❌ MongoDB connection failed')
    client = None

mongo_db = client['cardamom_db'] if client else None

# Primary collections

users        = mongo_db['users']
predictions  = mongo_db['predictions']
contacts     = mongo_db['contact_messages']
newsletter   = mongo_db['newsletter_subscribers']
farmers      = mongo_db['farmers']

# Alternative names (backward compatibility)
messages     = contacts
users_col    = users
predictions_col = predictions
messages_col = contacts
newsletter_col  = newsletter
farmers_col     = farmers

def create_indexes():
    try:
        users_col.create_index('email', unique=True)
        predictions_col.create_index([('user_id', ASCENDING),
                                       ('created_at', DESCENDING)])
        newsletter_col.create_index('email', unique=True)
        farmers_col.create_index('phone', unique=True)
        print('✅ MongoDB indexes created')
    except Exception as e:
        print(f'Index creation error: {e}')