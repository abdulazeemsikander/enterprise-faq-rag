import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from transformers import pipeline
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from src.memory import ConversationMemory
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Suppress PyTorch Metal (MPS) backend logs
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

load_dotenv()

INDEX_DIR = Path(__file__).resolve().parent.parent / "var" / "index"

# Define state
class ConversationState(TypedDict):
    question: str
    context: str
    answer: str
    history: List[str]

# Retriever
def retrieve_node(state: ConversationState):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(
        persist_directory=str(INDEX_DIR),
        embedding_function=embeddings,
    )
    retriever = db.as_retriever(search_kwargs={"k": 2})

    docs = retriever.invoke(state["question"])
    context = "\n".join([d.page_content for d in docs])
    state["context"] = context
    return state

# Generator
def generate_node(state: ConversationState):
    generator = HuggingFacePipeline(
        pipeline=pipeline("text2text-generation", model="google/flan-t5-base")
    )
    memory_context = "\n".join(state["history"])
    prompt = f"""You are a helpful assistant. 
Conversation so far:
{memory_context}

FAQ Context:
{state['context']}

User question: {state['question']}
Answer:"""

    result = generator.invoke(prompt)
    answer = result[0]["generated_text"] if isinstance(result, list) else str(result)
    state["answer"] = answer
    return state

# Memory updater
def memory_node(state: ConversationState):
    state["history"].append(f"User: {state['question']}\nAssistant: {state['answer']}")
    return state

# Build graph
def build_graph():
    workflow = StateGraph(ConversationState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("memory", memory_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "memory")
    workflow.add_edge("memory", END)

    return workflow.compile()
