import os
import json
os.environ["OPENAI_API_KEY"] = "ollama"

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from hybrid_search import hybrid_search
from rerank import rerank
from ask_v2 import ask_with_enforcement

with open("golden_dataset.json", encoding="utf-8") as f:
    golden = json.load(f)

answerable_results = []
refusal_results = []

for item in golden:
    question = item["question"]
    expect_refusal = item.get("expect_refusal", False)

    candidates = hybrid_search(question)
    top_contexts = rerank(question, candidates)
    answer = ask_with_enforcement(question)

    if expect_refusal:
        correctly_refused = "NOT_FOUND" in answer
        refusal_results.append({
            "question": question,
            "answer": answer,
            "correctly_refused": correctly_refused
        })
        print(f"[Refusal check: {'PASS' if correctly_refused else 'FAIL'}] {question}")
    else:
        answerable_results.append({
            "question": question,
            "answer": answer,
            "contexts": [c["text"] for c in top_contexts],
            "ground_truth": item["ground_truth"]
        })
        print(f"[Answerable] {question}")

# --- Score answerable questions with Ragas ---
dataset = Dataset.from_list(answerable_results)

local_llm = ChatOpenAI(model="llama3", base_url="http://localhost:11434/v1", api_key="ollama")
ragas_llm = LangchainLLMWrapper(local_llm)

local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
ragas_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

ragas_scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy], llm=ragas_llm, embeddings=ragas_embeddings)

# --- Score refusal accuracy ---
refusal_accuracy = sum(r["correctly_refused"] for r in refusal_results) / len(refusal_results) if refusal_results else None

print("\n--- FINAL RESULTS ---")
print("Ragas (answerable questions):", dict(ragas_scores))
print(f"Refusal accuracy: {refusal_accuracy}")

with open("eval_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "ragas_scores": dict(ragas_scores),
        "refusal_accuracy": refusal_accuracy
    }, f, indent=2)