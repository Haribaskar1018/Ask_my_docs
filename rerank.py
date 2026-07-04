from sentence_transformers import CrossEncoder
from hybrid_search import hybrid_search

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question, candidates, top_k=2):
    pairs = [(question, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    top = ranked[:top_k]
    # attach the score directly onto each chunk dict so callers can see it
    results = []
    for chunk, score in top:
        chunk_with_score = dict(chunk)
        chunk_with_score["relevance_score"] = float(score)
        results.append(chunk_with_score)
    return results

if __name__ == "__main__":
    candidates = hybrid_search("What is Depends?")
    top_results = rerank("What is Depends?", candidates)
    for r in top_results:
        print(r["source"], "-", round(r["relevance_score"], 3), "-", r["text"][:80])