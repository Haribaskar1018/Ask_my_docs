import os
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

def ingest_web(url, output_name, output_dir="docs"):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    article = soup.find("article") or soup.find("div", {"class": "body"}) or soup
    text = article.get_text(separator="\n")
    _save(text, output_name, output_dir)

def ingest_pdf(pdf_path, output_name, output_dir="docs"):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    _save(text, output_name, output_dir)

def ingest_markdown(md_path, output_name, output_dir="docs"):
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    _save(text, output_name, output_dir)

def _save(text, output_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/{output_name}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved {path}")

if __name__ == "__main__":
    ingest_pdf("haribaskar_resume.pdf", "resume", output_dir="docs_resume")