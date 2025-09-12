import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

INDEX_DIR = Path(__file__).resolve().parent.parent / "var" / "index"

def test_query(query: str):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=2)

    print(f"🔍 Query: {query}\n")
    for i, d in enumerate(docs, 1):
        print(f"Result {i}:")
        print(d.page_content)
        print("----")

if __name__ == "__main__":
    test_query("How do I get a refund?")