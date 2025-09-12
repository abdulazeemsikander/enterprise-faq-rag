import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_graph import build_graph

# Initialize app
app = FastAPI(title="FAQ Assistant with RAG")

# Build LangGraph workflow
graph = build_graph()

# Pydantic models
class QueryRequest(BaseModel):
    session_id: str
    question: str

class QueryResponse(BaseModel):
    answer: str
    context: str

# Session states (simple in-memory storage for now)
sessions = {}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    # Get or create state
    state = sessions.get(request.session_id, {"question": "", "context": "", "answer": "", "history": []})

    # Run through graph
    state["question"] = request.question
    state = graph.invoke(state)

    # Save session
    sessions[request.session_id] = state

    return QueryResponse(answer=state["answer"], context=state["context"])

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)