import json
from pathlib import Path

FAQ_FILE = Path(__file__).resolve().parent.parent / "data" / "faqs.json"

def load_faqs():
    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    faqs = load_faqs()
    for faq in faqs:
        print(f"Q: {faq['question']}")
        print(f"A: {faq['answer']}")
        print(f"Source: {faq['source']}")
        print("----")