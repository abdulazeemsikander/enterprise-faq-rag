# FAQ Assistant with RAG

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent **FAQ Assistant** built with **Retrieval-Augmented Generation (RAG)** using LangChain, Hugging Face models, FAISS/Chroma, and FastAPI.  
Ask questions in natural language and get accurate answers grounded in your FAQ knowledge base.

---

## 🚀 Features
- **Knowledge Retrieval**: Retrieves relevant FAQ chunks using FAISS (or Chroma fallback).
- **Contextual Answering**: HuggingFace model generates answers with context.
- **API Interface**: Built on FastAPI with Swagger UI docs.
- **Dockerized**: Ready to run anywhere in a container.
- **Unit Tested**: Includes `pytest` tests for retrieval pipeline.
- **Environment Config**: `.env.example` provided for setup.

---

## 📂 Project Structure
```
enterprise-faq-rag/
├── src/
│   ├── rag_pipeline.py       # RAG logic (retriever + generator)
│   ├── rag_graph.py          # LangGraph integration
│   ├── main.py               # FastAPI entrypoint
│   ├── test_conversation.py  # Example interactive script
│   └── ...
├── tests/
│   └── test_retriever.py     # Unit tests for retriever
├── data/
│   └── faqs.json             # FAQ knowledge base
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚡ Setup & Run (Local)

### 1️⃣ Clone the repo
```bash
git clone https://github.com/<your-username>/enterprise-faq-rag.git
cd enterprise-faq-rag
```

### 2️⃣ Create and activate venv
```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Run FastAPI server
```bash
uvicorn src.main:app --reload
```

Swagger docs available at:  
👉 http://localhost:8000/docs

---

## 🧪 Run Tests
Run the unit tests to verify retriever and pipeline:

```bash
pytest -q tests/test_retriever.py
```

Example test included:  
✔️ `test_refund_policy()` – verifies the refund FAQ.

---

## 🐳 Run with Docker

### 1️⃣ Build the image
```bash
docker build -t faq-rag .
```

### 2️⃣ Run the container
```bash
docker run -p 8000:8000 faq-rag
```

Access the API at: http://localhost:8000/docs

---

## ✅ Example API Usage

### Curl
```bash
curl -X POST "http://0.0.0.0:8000/ask"      -H "accept: application/json"      -H "Content-Type: application/json"      -d '{"session_id": "1", "question": "What is the refund policy?"}'
```

### Response
```json
{
  "answer": "Refunds are available within 30 days of purchase, provided the product is unused.",
  "context": "Q: What is the refund policy?\nA: Refunds are available within 30 days ..."
}
```

---

## 📦 Deployment

### Push Docker image
```bash
docker tag faq-rag <your-dockerhub-username>/faq-rag:latest
docker push <your-dockerhub-username>/faq-rag:latest
```

### Deploy
You can deploy the container on:
- **Render**
- **Railway**
- **AWS ECS**
- **Google Cloud Run**

---

## 🔧 Environment Variables

Copy `.env.example` → `.env` and configure as needed.

Example:
```env
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
INDEX_DIR=var/index
```

---

## 📜 License
This project is licensed under the MIT License.  
© 2025 Abdul Azeem Sikander
