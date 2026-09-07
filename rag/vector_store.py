import faiss
import numpy as np


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []

    def build_index(self, embeddings, chunks):
        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        self.chunks = chunks

    def search(self, query_embedding, top_k=5):
        query_embedding = np.asarray(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)

            results.append(chunk)

        return results