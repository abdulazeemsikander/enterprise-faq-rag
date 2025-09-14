from src.rag_pipeline import rag_answer

def test_refund_policy():
    answer, _ = rag_answer("What is the refund policy?")
    assert "30 days" in answer.lower()
