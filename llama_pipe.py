from typing import List, Dict, Any
from ollama import AsyncClient

class LocalRAGPipeline:
    def __init__(self, model_name: str = "llama3.1:latest", host: str = "http://localhost:11434", retriever=None):
        self.model_name = model_name
        self.client = AsyncClient(host=host)
        self.retriever = retriever  # AsyncRetriever will be set here

    async def set_retriever(self, retriever):
        """Set the retriever for the RAG system."""
        self.retriever = retriever

    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents."""
        if self.retriever is None:
            raise ValueError("Retriever is not set. Use set_retriever() method.")
        return await self.retriever.retrieve(query)

    async def generate(self, prompt: str, context: List[str]) -> str:
        """Generate a response using the local LLaMA model."""
        full_prompt = f"Context:\n{' '.join(context)}\n\nQuestion: {prompt}\n\nAnswer:"
        response = await self.client.generate(model=self.model_name, prompt=full_prompt)
        return response['response']

    async def run(self, query: str) -> str:
        """Run the full RAG pipeline."""
        retrieved_docs = await self.retrieve(query)
        context = [doc['content']['content'] for doc in retrieved_docs if isinstance(doc['content'], dict)]
        return await self.generate(query, context)

# Example usage
async def main():
    # Initialize the pipeline
    rag_pipeline = LocalRAGPipeline()

    # Set up a dummy retriever (replace this with your actual retriever)
    class DummyRetriever:
        async def retrieve(self, query):
            return [{"content": {"content": "This is a dummy retrieved document."}}]

    await rag_pipeline.set_retriever(DummyRetriever())

    # Run a query
    query = "What is the capital of France?"
    response = await rag_pipeline.run(query)
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())