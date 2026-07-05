from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
import json as json_lib
import yaml
import time
import math
import os
import uuid
import json
from werkzeug.utils import secure_filename
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from hybrid_search import hybrid_search, register_domain
from rerank import rerank

app = Flask(__name__)
CORS(app)

# ---------- Existing /ask_stream setup ----------

GPT4O_INPUT_COST_PER_1K = 0.005
GPT4O_OUTPUT_COST_PER_1K = 0.015

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def estimate_tokens(text):
    return int(len(text.split()) * 0.75)

@app.route("/ask_stream", methods=["POST"])
def ask_stream():
    data = request.get_json()
    question = data.get("question", "")
    domain = data.get("domain", "fastapi")
    history = data.get("history", [])

    t0 = time.time()
    candidates = hybrid_search(question, domain=domain)
    t1 = time.time()
    top_contexts = rerank(question, candidates)
    t2 = time.time()

    retrieval_ms = round((t1 - t0) * 1000)
    rerank_ms = round((t2 - t1) * 1000)

    context_block = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in top_contexts)

    history_block = ""
    if history:
        last = history[-1]
        history_block = f"Previous question: {last['question']}\nPrevious answer: {last['answer']}\n\n"

    with open("prompts.yaml", encoding="utf-8") as f:
        template = yaml.safe_load(f)["v3"]["template"]
    prompt = history_block + template.format(context=context_block, question=question)

    scores = [c["relevance_score"] for c in top_contexts]
    min_score, max_score = min(scores), max(scores)

    sources = []
    for c in top_contexts:
        if max_score == min_score:
            confidence_pct = 100
        else:
            confidence_pct = round((c["relevance_score"] - min_score) / (max_score - min_score) * 100)
        sources.append({
            "source": c["source"],
            "text": c["text"][:300],
            "confidence": confidence_pct
        })

    input_tokens = estimate_tokens(prompt)

    def generate():
        yield json_lib.dumps({
            "type": "sources",
            "sources": sources,
            "timing": {"retrieval_ms": retrieval_ms, "rerank_ms": rerank_ms}
        }) + "\n"

        gen_start = time.time()
        full_text = ""
        for chunk in ollama.generate(model="llama3", prompt=prompt, stream=True):
            token = chunk["response"]
            full_text += token
            yield json_lib.dumps({"type": "token", "text": token}) + "\n"
        gen_ms = round((time.time() - gen_start) * 1000)

        output_tokens = estimate_tokens(full_text)
        estimated_cost = (input_tokens / 1000 * GPT4O_INPUT_COST_PER_1K) + \
                          (output_tokens / 1000 * GPT4O_OUTPUT_COST_PER_1K)

        refused = "NOT_FOUND" in full_text
        yield json_lib.dumps({
            "type": "done",
            "refused": refused,
            "timing": {"generation_ms": gen_ms, "total_ms": retrieval_ms + rerank_ms + gen_ms},
            "cost": {"estimated_gpt4o_cost": round(estimated_cost, 5), "actual_cost": 0.0}
        }) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")

# ---------- Upload route ----------

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

upload_embedder = SentenceTransformer("all-MiniLM-L6-v2")
upload_chroma_client = chromadb.PersistentClient(path="./chroma_db")

def extract_text(filepath, filename):
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    else:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            return f.read()

def chunk_text(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    filename = secure_filename(file.filename)
    upload_id = uuid.uuid4().hex[:8]
    saved_path = os.path.join(UPLOAD_FOLDER, f"{upload_id}_{filename}")
    file.save(saved_path)

    text = extract_text(saved_path, filename)
    print(f"DEBUG: extracted {len(text)} characters from {filename}")
    print(f"DEBUG: first 200 chars: {text[:200]!r}")

    if not text or not text.strip():
        return jsonify({"error": "No text could be extracted from this file. If it's a PDF, it may be a scanned image with no selectable text."}), 400

    text_chunks = chunk_text(text)
    if not text_chunks:
        return jsonify({"error": "File was too short to create any chunks."}), 400

    domain_name = f"upload_{upload_id}"
    chunks_file = f"chunks_{domain_name}.json"
    collection_name = f"{domain_name}_docs"

    chunk_dicts = [
        {"id": f"{domain_name}-{i}", "source": filename, "text": c}
        for i, c in enumerate(text_chunks)
    ]
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, indent=2)

    collection = upload_chroma_client.get_or_create_collection(collection_name)
    texts = [c["text"] for c in chunk_dicts]
    ids = [c["id"] for c in chunk_dicts]
    metadatas = [{"source": c["source"]} for c in chunk_dicts]
    embeddings = upload_embedder.encode(texts).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    register_domain(domain_name, chunks_file, collection_name)

    return jsonify({
        "domain": domain_name,
        "filename": filename,
        "chunks_created": len(chunk_dicts)
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)