import os
import json
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
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized.")

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

    def save(self, storage_dir="storage"):
        if self.index is None:
            raise RuntimeError("No FAISS index to save.")

        os.makedirs(
            storage_dir,
            exist_ok=True
        )

        index_path = os.path.join(
            storage_dir,
            "index.faiss"
        )

        chunks_path = os.path.join(
            storage_dir,
            "chunks.json"
        )

        faiss.write_index(
            self.index,
            index_path
        )

        with open(
            chunks_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                self.chunks,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(f"FAISS index saved to: {index_path}")
        print(f"Chunks saved to: {chunks_path}")

    def load(self, storage_dir="storage"):
        index_path = os.path.join(
            storage_dir,
            "index.faiss"
        )

        chunks_path = os.path.join(
            storage_dir,
            "chunks.json"
        )

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not os.path.exists(chunks_path):
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_path}"
            )

        self.index = faiss.read_index(
            index_path
        )

        with open(
            chunks_path,
            "r",
            encoding="utf-8"
        ) as f:
            self.chunks = json.load(f)

        print(f"Loaded {len(self.chunks)} chunks")
        print(f"FAISS vectors: {self.index.ntotal}")