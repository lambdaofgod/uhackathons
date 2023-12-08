import dspy


class SimpleRAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(
            "context, question -> answer")

    def forward(self, question, max_tokens=256):
        context = self.retrieve(question).passages
        answer = self.generate_answer(
            context=context, question=question, max_tokens=max_tokens)
        return context, answer
