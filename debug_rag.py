import sys
sys.path.insert(0,'/app')
from backend.services.vector_store import search_similar
from backend.services.rag import _rerank_documents, _filter_relevant_docs, MIN_SCORE_THRESHOLD

query = 'โครงการมีอะไรบ้าง'
raw = search_similar(query, k=15)
print('raw docs:', len(raw))
reranked = _rerank_documents(query, raw, 10)
print('reranked docs:', len(reranked))
for i, d in enumerate(reranked[:5]):
    src = d.metadata.get('source','?')
    ai  = d.metadata.get('ai_score', 0)
    kw  = d.metadata.get('keyword_score', 0)
    print(f'  [{i}] src={src} ai_score={ai:.4f} kw={kw:.1f}')
filtered = _filter_relevant_docs(query, reranked)
print('filtered passed:', len(filtered))
print('MIN_SCORE_THRESHOLD:', MIN_SCORE_THRESHOLD)
