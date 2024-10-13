import faiss
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

class AsyncRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2', use_gpu=True):
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index = None  # Will hold the FAISS index
        self.documents = []
        self.document_embeddings = None

    async def add_documents(self, docs):
        """Adds documents and computes their embeddings."""
        self.documents = docs
        print(f"Generating embeddings for {len(docs)} documents...")
        self.document_embeddings = self.model.encode(docs, show_progress_bar=True, convert_to_tensor=True)
        
        embeddings_np = self.document_embeddings.cpu().detach().numpy()
        
        # Build the FAISS index for retrieval
        self.index = faiss.IndexFlatL2(embeddings_np.shape[1])
        self.index.add(embeddings_np)

    async def retrieve(self, query, top_k=5):
        """Asynchronously retrieves the most relevant documents for the query."""
        query_embedding = self.model.encode(query, convert_to_tensor=True).cpu().detach().numpy().reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                'content': self.documents[idx],  # Assuming 'content' is the relevant field
                'score': 1 - dist  # FAISS uses L2 distance; similarity = 1 - distance
            })
        return results
