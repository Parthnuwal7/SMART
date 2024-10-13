from llama_index.core import  VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core.prompts.prompts import SimpleInputPrompt
import asyncio
from llama_pipe import LocalRAGPipeline
from retriever import AsyncRetriever


document = SimpleDirectoryReader('./data').load_data()
document = {
    "content" : doc.text for doc in document
}

async def main():
    # Initialize the retriever
    retriever = AsyncRetriever(use_gpu=True)
    await retriever.add_documents([document])

    # Initialize the Local RAG Pipeline and set the retriever
    rag_pipeline = LocalRAGPipeline()
    await rag_pipeline.set_retriever(retriever)

    # Run the pipeline with a query
    query = "what does the financial report say? what columns does it have?"
    response = await rag_pipeline.run(query)

    print("Generated Response:", response)


if __name__ == "__main__":
    asyncio.run(main())