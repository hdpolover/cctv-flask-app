import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

def initialize_firebase():
    """Initialize Firebase."""
    if not firebase_admin._apps:
        cred = credentials.Certificate('cctv_app/cctv-app-flask-firebase-adminsdk-xdxtx-8e5ea88cd9.json')
        firebase_admin.initialize_app(cred)
    return firestore.client()

def save_to_firestore(db, people_in, people_out):
    """
    Save count data to Firestore based on the day.
    If a record exists for the current day, update it.
    Otherwise, create a new record.
    """
    today_date = datetime.now().strftime("%Y-%m-%d")
    collection_ref = db.collection("people_counter")
    doc_ref = collection_ref.document(today_date)
    doc = doc_ref.get()
    
    if doc.exists:
        doc_ref.update({
            "people_in": firestore.Increment(people_in),
            "people_out": firestore.Increment(people_out),
            "last_updated": firestore.SERVER_TIMESTAMP
        })
    else:
        doc_ref.set({
            "date": today_date,
            "people_in": people_in,
            "people_out": people_out,
            "last_updated": firestore.SERVER_TIMESTAMP
        })