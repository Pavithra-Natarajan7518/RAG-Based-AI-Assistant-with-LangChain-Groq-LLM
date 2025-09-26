# vectordb.py
import re
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import traceback
import os

class VectorDB:
    """
    Simple in-memory vector DB using SentenceTransformer embeddings and NumPy similarity search.
    - Not persistent (keeps data in memory)
    - Good for debugging and small datasets
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
        self.embedding_model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        print(f"[VectorDB] Loading embedding model: {self.embedding_model_name}", flush=True)
        # Use convert_to_numpy for consistent numpy arrays
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # in-memory storage
        self.ids: List[str] = []
        self.docs: List[str] = []
        self.metadatas: List[dict] = []
        self.embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)  # shape (n_items, emb_dim)

        print(f"[VectorDB] Initialized in-memory collection: {self.collection_name}", flush=True)

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], []
        length = 0
        for sent in sentences:
            if length + len(sent) > chunk_size and current:
                chunks.append(" ".join(current))
                current, length = [], 0
            current.append(sent)
            length += len(sent)
        if current:
            chunks.append(" ".join(current))
        return chunks

    def add_documents(self, documents: List[dict]) -> None:
        """
        documents: list of {"content": "...", "metadata": {...}}
        """
        print(f"[VectorDB] Processing {len(documents)} documents...", flush=True)
        all_texts, all_metadatas, all_ids = [], [], []

        for doc_index, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {}) or {}
            chunks = self.chunk_text(content)
            for i, chunk in enumerate(chunks):
                chunk_id = f"doc{doc_index}_chunk{i}"
                all_texts.append(chunk)
                # ensure metadata is serializable/simple
                all_metadatas.append({k: str(v) for k, v in metadata.items()})
                all_ids.append(chunk_id)

        if not all_texts:
            print("[VectorDB] No text to add, returning.", flush=True)
            return

        try:
            print("[VectorDB] Encoding embeddings...", flush=True)
            # Ensure numpy array of dtype float32 for consistency
            new_embeddings = self.embedding_model.encode(all_texts, convert_to_numpy=True).astype(np.float32)
            print(f"[VectorDB] Encoded {len(all_texts)} chunks, emb dim = {new_embeddings.shape[1]}", flush=True)
        except Exception as e:
            print("[VectorDB] ERROR during embedding:", e, flush=True)
            traceback.print_exc()
            raise

        # Append to existing in-memory store
        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
        else:
            # verify dims match
            if new_embeddings.shape[1] != self.embeddings.shape[1]:
                raise ValueError("Embedding dimension mismatch.")
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.ids.extend(all_ids)
        self.docs.extend(all_texts)
        self.metadatas.extend(all_metadatas)

        print(f"[VectorDB] Added {len(all_texts)} chunks. Total stored chunks: {len(self.ids)}", flush=True)

    def _cosine_similarity(self, query_emb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        # query_emb: (D,), matrix: (N, D) => return (N,)
        # normalize for cosine
        if matrix.size == 0:
            return np.array([])
        q = query_emb / (np.linalg.norm(query_emb) + 1e-12)
        m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
        sims = np.dot(m, q)
        return sims

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Returns top-n most similar document chunks.
        """
        if len(self.ids) == 0:
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}

        # encode query
        try:
            q_emb = self.embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)[0]
        except Exception as e:
            print("[VectorDB] ERROR encoding query:", e, flush=True)
            traceback.print_exc()
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}

        sims = self._cosine_similarity(q_emb, self.embeddings)  # higher = better
        # convert to "distance" like (1 - similarity)
        distances = 1.0 - sims

        # get top n_results
        idx_sorted = np.argsort(distances)[:n_results]  # smallest distance first
        results_docs = [self.docs[i] for i in idx_sorted.tolist()]
        results_metas = [self.metadatas[i] for i in idx_sorted.tolist()]
        results_ids = [self.ids[i] for i in idx_sorted.tolist()]
        results_distances = distances[idx_sorted].tolist()

        return {
            "documents": results_docs,
            "metadatas": results_metas,
            "distances": results_distances,
            "ids": results_ids,
        }
