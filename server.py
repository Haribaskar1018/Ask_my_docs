from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
import json as json_lib
import yaml
import time
import math
import ollama
from hybrid_search import hybrid_search
from rerank import rerank

app = Flask(__name__)
CORS(app)

# Rough GPT-4o pricing for cost comparison (as of public pricing, per 1M tokens)
GPT4O_INPUT_COST_PER_1K = 0.005
GPT4O_OUTPUT_COST_PER_1K = 0.015

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def estimate_tokens(text):
    # rough estimate: ~0.75 tokens per word
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

    sources = []
    for c in top_contexts:
        confidence_pct = round(sigmoid(c["relevance_score"]) * 100)
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

if __name__ == "__main__":
    app.run(port=5000, debug=True)