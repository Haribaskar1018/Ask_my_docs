import os
import json

def chunk_text(text, chunk_size=600, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

all_chunks = []
for filename in os.listdir("docs_requests"):
    with open(f"docs_requests/{filename}", encoding="utf-8") as f:
        text = f.read()
    for i, c in enumerate(chunk_text(text)):
        all_chunks.append({"id": f"{filename}-{i}", "source": filename, "text": c})

with open("chunks_requests.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

print(f"Created {len(all_chunks)} chunks")