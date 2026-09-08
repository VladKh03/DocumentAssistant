import os

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

    def add_documents(self, file_paths):
        if not file_paths:
            raise ValueError(
                "No documents provided."
            )

        existing_chunks = self.vector_store.chunks

        if existing_chunks:
            next_chunk_id = max(
                chunk["chunk_id"]
                for chunk in existing_chunks
            ) + 1
        else:
            next_chunk_id = 0


        new_chunks = []

        total_pages = 0

        processed_documents = []

        for file_path in file_paths:
            document_name = os.path.basename(
                file_path
            )

            pages = parse_pdf(
                file_path
            )

            total_pages += len(pages)

            document_chunks = create_chunks(
                pages=pages,
                document_name=document_name,
                start_chunk_id=next_chunk_id
            )

            new_chunks.extend(
                document_chunks
            )

            next_chunk_id += len(
                document_chunks
            )

            processed_documents.append(
                document_name
            )

        if not new_chunks:
            raise RuntimeError(
                "No text chunks were created from the uploaded documents."
            )

        embeddings = self.embedding_model.encode_chunks(
            new_chunks
        )

        self.vector_store.add(
            embeddings=embeddings,
            chunks=new_chunks
        )

        self.vector_store.save(
            "storage"
        )

        self.index_ready = True

        return {
            "documents_added": len(file_paths),
            "document_names": processed_documents,
            "pages_added": total_pages,
            "chunks_added": len(new_chunks),
            "total_chunks": len(
                self.vector_store.chunks
            )
        }

    def ask(self, question):
        if not self.index_ready:
            raise RuntimeError(
                "No documents have been indexed yet."
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
                (
                    chunk["document"],
                    chunk["page"]
                )
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

        self.index_ready = True

        return {
            "chunks": len(
                self.vector_store.chunks
            ),
            "documents": len(
                {
                    chunk["document"]
                    for chunk
                    in self.vector_store.chunks
                }
            )
        }