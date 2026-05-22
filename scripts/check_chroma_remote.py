#!/usr/bin/env python3
import chromadb

c = chromadb.HttpClient(host='localhost', port=8000)
cols = c.list_collections()
print('=== Collections ===')
for col in cols:
    print(f'Name: {col.name}  Count: {col.count()}')
    # sample 2 items to see metadata
    try:
        items = col.get(limit=2)
        if items and items.get('metadatas'):
            print(f'  Sample metadata: {items["metadatas"][0]}')
    except Exception as e:
        print(f'  Error sampling: {e}')
print('=== DONE ===')
