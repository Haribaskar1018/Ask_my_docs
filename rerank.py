from sentence_transformers import CrossEncoder
from hybrid_search import hybrid_search

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question, candidates, top_k=2):
    pairs = [(question, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [c for c, score in ranked[:top_k]]

if __name__ == "__main__":
    candidates = hybrid_search("What is Depends?")
    top_results = rerank("What is Depends?", candidates)
    for r in top_results:
        print(r["source"], "-", r["text"][:80])