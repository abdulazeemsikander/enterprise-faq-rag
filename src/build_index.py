import json
import shutil
from pathlib import Path
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Load env vars
load_dotenv()

# Paths
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "faqs.json"
INDEX_DIR = Path(__file__).resolve().parent.parent / "var" / "index"

# Ensure index folder exists
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def load_faqs():
    """Load FAQ dataset from JSON file"""
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index():
    # 1. Load data
    faqs = load_faqs()
    docs = []

    for faq in faqs:
        text = f"Q: {faq['question']}\nA: {faq['answer']}"
        # ✅ Attach metadata, force values to str
        metadata = {k: str(v) for k, v in faq.items()}
        docs.append(Document(page_content=text, metadata=metadata))

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # characters
        chunk_overlap=50,
    )
    split_docs = splitter.split_documents(docs)

    print(f"Total chunks: {len(split_docs)}")

    # 3. Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Clean old index (avoid mixing with previous builds)
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)

    # 5. Create and persist Chroma index
    db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(INDEX_DIR),
    )
    db.persist()  # ✅ ensure index is written to disk

    print(f"Chroma index built and saved to {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
