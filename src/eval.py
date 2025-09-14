import json
from pathlib import Path
from src.rag_pipeline import rag_answer

# Small test set
TEST_QUERIES = [
    {
        "question": "What is the refund policy?",
        "expected": "Refunds are available within 30 days"
    },
    {
        "question": "How do I reset my password?",
        "expected": "Forgot password"
    },
    {
        "question": "Do you offer technical support?",
        "expected": "support"
    }
]

def evaluate():
    results = []
    for test in TEST_QUERIES:
        answer, context = rag_answer(test["question"])
        correct = test["expected"].lower() in answer.lower()
        results.append({
            "question": test["question"],
            "expected": test["expected"],
            "answer": answer,
            "correct": correct
        })
    return results

if __name__ == "__main__":
    results = evaluate()
    correct = sum(r["correct"] for r in results)
    total = len(results)
    print(f"\n✅ Correct: {correct}/{total} ({correct/total:.0%})\n")
    for r in results:
        print(f"Q: {r['question']}")
        print(f"Expected: {r['expected']}")
        print(f"Answer: {r['answer']}")
        print(f"Correct: {r['correct']}")
        print("----")