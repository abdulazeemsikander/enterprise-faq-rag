class ConversationMemory:
    def __init__(self):
        self.history = []

    def add_turn(self, question, answer):
        self.history.append({"question": question, "answer": answer})

    def get_context(self):
        formatted = "\n".join(
            [f"User: {h['question']}\nAssistant: {h['answer']}" for h in self.history]
        )
        return formatted if formatted else "No previous conversation."