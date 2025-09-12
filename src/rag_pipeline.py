import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import pipeline

load_dotenv()

INDEX_DIR = Path(__file__).resolve().parent.parent / "var" / "index"

# Load retriever
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 2})

# Load generator (LLM)
def get_generator():
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",   # lightweight model, can replace with a bigger one if GPU available
        tokenizer="google/flan-t5-base"
    )
    return HuggingFacePipeline(pipeline=generator)


# Build RAG answer
def rag_answer(query: str):
    retriever = get_retriever()
    generator = get_generator()

    # Step 1: Retrieve relevant chunks
    docs = retriever.get_relevant_documents(query)

    # Step 2: Construct context
    context = "\n".join([f"{i+1}. {doc.page_content} (Source: {doc.metadata.get('source','N/A')})"
                         for i, doc in enumerate(docs)])

    # Step 3: Build prompt
    prompt = f"""You are an assistant. 
Use the following FAQ information to answer the question. 
If you don't know, say "I don't know."

Context:
{context}

Question: {query}
Answer:"""
    
    result = generator(prompt, max_new_tokens=200)
    # HuggingFace pipeline returns a list of dicts OR a plain string depending on wrapper
    if isinstance(result, list) and "generated_text" in result[0]:
        answer = result[0]["generated_text"]
    elif isinstance(result, str):
        answer = result
    else:
        answer = str(result)

    return answer, context

if __name__ == "__main__":
    questions = [
        "How do I reset my password?",
        "What is the refund policy?",
        "Do you offer technical support?",
        "Can I change my shipping address?"
    ]

    for q in questions:
        answer, used_context = rag_answer(q)
        print("\n==============================")
        print("🔍 Question:", q)
        print("\n📖 Context used:\n", used_context)
        print("\n💡 Answer:\n", answer)
