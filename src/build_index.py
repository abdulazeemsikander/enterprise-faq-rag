import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load env vars
load_dotenv()

# Paths
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "faqs.json"
INDEX_DIR = Path(__file__).resolve().parent.parent / "var" / "index"

INDEX_DIR.mkdir(parents=True, exist_ok=True)

def load_faqs():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def build_index():
    # 1. Load data
    faqs = load_faqs()
    docs = []
    for faq in faqs:
        text = f"Q: {faq['question']}\nA: {faq['answer']}"
        docs.append(text)

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # characters
        chunk_overlap=50,
    )
    chunks = splitter.split_text("\n\n".join(docs))

    print(f"Total chunks: {len(chunks)}")

    # 3. Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Create FAISS index
    db = FAISS.from_texts(chunks, embeddings)

    # 5. Save index locally
    db.save_local(str(INDEX_DIR))
    print(f"Index saved to {INDEX_DIR}")

if __name__ == "__main__":
    build_index()