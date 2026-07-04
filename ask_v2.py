import ollama
import yaml
from hybrid_search import hybrid_search
from rerank import rerank

def ask_with_enforcement(question):
    candidates = hybrid_search(question)
    top_contexts = rerank(question, candidates)

    context_block = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in top_contexts
    )

    with open("prompts.yaml", encoding="utf-8") as f:
        template = yaml.safe_load(f)["v3"]["template"]
    prompt = template.format(context=context_block, question=question)

    response = ollama.generate(model="llama3", prompt=prompt)
    return response["response"]

if __name__ == "__main__":
    print(ask_with_enforcement("What is Depends used for?"))
    print("---")
    print(ask_with_enforcement("What is the capital of France?"))