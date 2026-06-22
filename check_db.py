# check_db.py
import sys
import os
from pathlib import Path

# Fix encoding issue on Windows terminal
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Set Path to access backend
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from backend.services.vector_store import get_vector_store

def check():
    try:
        vectordb = get_vector_store()
        count = vectordb._collection.count()
        print(f"Total documents in collection: {count}")
        
        if count > 0:
            sample = vectordb._collection.get(limit=1)
            print(f"Sample document ID: {sample['ids'][0]}")
            print(f"Sample metadata: {sample['metadatas'][0]}")
    except Exception as e:
        print(f"Error checking DB: {e}")

if __name__ == "__main__":
    check()
