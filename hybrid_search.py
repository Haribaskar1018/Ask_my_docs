import json
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")

DOMAINS = {
    "fastapi": {"chunks_file": "chunks.json", "collection": "fastapi_docs"},
    "requests": {"chunks_file": "chunks_requests.json", "collection": "requests_docs"},
    "resume": {"chunks_file": "chunks_resume.json", "collection": "resume_docs"},
}

def register_domain(domain_name, chunks_file, collection_name):
    DOMAINS[domain_name] = {"chunks_file": chunks_file, "collection": collection_name}
    if domain_name in _cache:
        del _cache[domain_name]

_cache = {}

def _load_domain(domain):
    if domain not in _cache:
        config = DOMAINS[domain]
        with open(config["chunks_file"], encoding="utf-8") as f:
            chunks = json.load(f)
        tokenized = [c["text"].lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized)
        collection = client.get_collection(config["collection"])
        _cache[domain] = (chunks, bm25, collection)
    return _cache[domain]

def hybrid_search(question, domain="fastapi", top_k=5):
    chunks, bm25, collection = _load_domain(domain)

    bm25_scores = bm25.get_scores(question.lower().split())
    bm25_top_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:top_k]
    bm25_results = [chunks[i] for i in bm25_top_idx]

    q_embedding = embedder.encode([question]).tolist()
    vec_results = collection.query(query_embeddings=q_embedding, n_results=top_k)
    vec_chunk_ids = vec_results["ids"][0]
    vec_results_full = [c for c in chunks if c["id"] in vec_chunk_ids]

    combined = {c["id"]: c for c in bm25_results + vec_results_full}
    return list(combined.values())
