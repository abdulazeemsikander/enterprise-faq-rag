import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
from src.rag_graph import build_graph
warnings.filterwarnings("ignore", category=UserWarning)

def run_conversation():
    app = build_graph()
    state = {"question": "", "context": "", "answer": "", "history": []}

    while True:
        q = input("\nYou: ")
        if q.lower() in ["exit", "quit"]:
            break
        state["question"] = q
        state = app.invoke(state)
        print(f"Assistant: {state['answer']}")

if __name__ == "__main__":
    run_conversation()