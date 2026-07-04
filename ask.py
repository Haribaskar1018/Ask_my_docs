import chromadb
from sentence_transformers import SentenceTransformer
import ollama

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("fastapi_docs")

def ask(question, top_k=3):
    q_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=top_k)

    contexts = results["documents"][0]
    sources = results["metadatas"][0]

    context_block = "\n\n".join(
        f"[Source: {s['source']}]\n{c}" for c, s in zip(contexts, sources)
    )

    prompt = f"""Answer the question using ONLY the context below.
Cite the source file name for your answer.
If the answer is not in the context, say "I don't know based on the provided docs."

Context:
{context_block}

Question: {question}

Answer:"""

    response = ollama.generate(model="llama3", prompt=prompt)
    print(response["response"])

if __name__ == "__main__":
    ask("How do you define a path parameter in FastAPI?")