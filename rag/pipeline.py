from rag.parser import parse_pdf
from rag.chunker import create_chunks
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.retrieval import Retriever
from llm.model import QwenLLM


class RAGPipeline:
    def __init__(self, top_k=5):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        self.retriever = Retriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            top_k=top_k
        )

        self.llm = QwenLLM()

        self.chunks = []
        self.index_ready = False

    def index_pdf(self, file_path):
        # 1 PDF to pages
        pages = parse_pdf(file_path)

        # 2 pages to chunks
        self.chunks = create_chunks(pages)

        # 3 chunks to embeddings
        embeddings = self.embedding_model.encode_chunks(
            self.chunks
        )

        # 4 embeddings to FAISS
        self.vector_store.build_index(
            embeddings=embeddings,
            chunks=self.chunks
        )

        # 5 save FAISS index + chunks
        self.vector_store.save(
            "storage"
        )

        self.index_ready = True

        return {
            "pages": len(pages),
            "chunks": len(self.chunks)
        }

    def ask(self, question):
        if not self.index_ready:
            raise RuntimeError(
                "No PDF has been indexed yet."
            )

        retrieved_chunks = self.retriever.retrieve(
            question
        )

        answer = self.llm.generate(
            question=question,
            chunks=retrieved_chunks
        )

        sources = sorted(
            {
                chunk["page"]
                for chunk in retrieved_chunks
            }
        )

        return {
            "answer": answer,
            "sources": sources,
            "chunks": retrieved_chunks
        }
    
    def load_index(self):
        self.vector_store.load(
            "storage"
        )

        self.chunks = self.vector_store.chunks

        self.index_ready = True