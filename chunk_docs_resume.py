import os
import json

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

all_chunks = []
for filename in os.listdir("docs_resume"):
    with open(f"docs_resume/{filename}", encoding="utf-8") as f:
        text = f.read()
    for i, c in enumerate(chunk_text(text)):
        all_chunks.append({"id": f"{filename}-{i}", "source": filename, "text": c})

with open("chunks_resume.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

print(f"Created {len(all_chunks)} chunks")