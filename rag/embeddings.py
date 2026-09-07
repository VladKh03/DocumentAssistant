from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name="BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)

    def encode_chunks(self, chunks):
        """
        Takes a list of chunks and returns embeddings for their text
        """

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings

    def encode_query(self, query):
        """
        Accepts a user question and returns one embedding
        """

        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        return embedding