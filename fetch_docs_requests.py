import requests
from bs4 import BeautifulSoup
import os

os.makedirs("docs_requests", exist_ok=True)

urls = {
    "quickstart": "https://requests.readthedocs.io/en/latest/user/quickstart/",
    "advanced": "https://requests.readthedocs.io/en/latest/user/advanced/",
    "authentication": "https://requests.readthedocs.io/en/latest/user/authentication/",
}

for name, url in urls.items():
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    article = soup.find("div", {"class": "body"}) or soup.find("article") or soup
    text = article.get_text(separator="\n")
    with open(f"docs_requests/{name}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved {name}")