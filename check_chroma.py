import chromadb
c = chromadb.HttpClient(host='hybrid_rag_chromadb', port=8000)
try:
    col = c.get_collection('yru_planning_data')
    print('Current docs:', col.count())
except Exception as e:
    print('Error:', e)
