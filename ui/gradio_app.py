import gradio as gr

from rag.pipeline import RAGPipeline


rag = RAGPipeline(
    top_k=5
)

try:
    info = rag.load_index()

    print("Existing index loaded successfully")

    print(f"Documents: {info['documents']}")

    print( f"Chunks: {info['chunks']}")

except FileNotFoundError:
    print(
        "No saved index found. "
        "Waiting for documents."
    )

except Exception as e:
    print(f"Failed to load saved index: {e}")

def process_documents(files):
    if not files:
        return (
            "Please upload at least one PDF."
        )

    try:
        file_paths = []

        for file in files:
            if isinstance(file, str):
                file_paths.append(file)

            else:
                file_paths.append(
                    file.name
                )

        info = rag.add_documents(
            file_paths
        )

        document_names = "\n".join(
            f"- {name}"
            for name in info["document_names"]
        )

        return (
            "Documents processed successfully\n\n"
            f"Documents added: "
            f"{info['documents_added']}\n\n"
            f"{document_names}\n\n"
            f"Pages added: "
            f"{info['pages_added']}\n"
            f"Chunks added: "
            f"{info['chunks_added']}\n"
            f"Total chunks: "
            f"{info['total_chunks']}"
        )

    except Exception as e:
        return (
            "Error while processing documents:\n"
            f"{str(e)}"
        )

def ask_document(question):
    if not question.strip():
        return "", ""

    try:
        result = rag.ask(
            question
        )

        answer = result["answer"]

        sources = "\n".join(
            f"{document} — Page {page}"
            for document, page
            in result["sources"]
        )

        return (
            answer,
            sources
        )

    except Exception as e:
        return (
            f"Error:\n{str(e)}",
            ""
        )

def debug_retrieval(question):
    if not question.strip():
        return "Please enter a question."

    try:
        results = rag.retriever.retrieve(question)

        if not results:
            return "No results found."

        output = []

        for i, result in enumerate(results, start=1):
            output.append(
                f"Result {i}\n"
                f"Document: {result.get('document', 'Unknown')}\n"
                f"Page: {result['page']}\n"
                f"Score: {result['score']:.4f}\n\n"
                f"{result['text']}\n"
                f"{'-' * 80}"
            )

        return "\n\n".join(output)

    except Exception as e:
        return f"Error:\n{str(e)}"

with gr.Blocks(
    title="AI Document Assistant"
) as demo:

    gr.Markdown(
        """
        # AI Document Assistant

        Upload PDF documents and ask questions
        about their content.
        """
    )

    with gr.Tabs():
        with gr.Tab("Chat"):
            question = gr.Textbox(
                label="Question",
                placeholder=(
                    "Ask something "
                    "about your documents..."
                ),
                lines=3
            )

            ask_button = gr.Button(
                "Ask",
                variant="primary"
            )

            answer = gr.Textbox(
                label="Answer",
                lines=10
            )

            sources = gr.Textbox(
                label="Sources",
                lines=5
            )

            ask_button.click(
                fn=ask_document,
                inputs=question,
                outputs=[
                    answer,
                    sources
                ]
            )

            question.submit(
                fn=ask_document,
                inputs=question,
                outputs=[
                    answer,
                    sources
                ]
            )

        with gr.Tab("Documents"):
            documents = gr.File(
                label="Upload PDFs",
                file_types=[
                    ".pdf"
                ],
                file_count="multiple"
            )

            process_button = gr.Button(
                "Add documents",
                variant="primary"
            )

            status = gr.Textbox(
                label="Status",
                lines=10
            )

            process_button.click(
                fn=process_documents,
                inputs=documents,
                outputs=status
            )
        
        with gr.Tab("Retrieval"):
            retrieval_question = gr.Textbox(
                label="Question",
                placeholder="Enter a question to inspect retrieval...",
                lines=3
            )

            retrieval_button = gr.Button(
                "Search",
                variant="primary"
            )

            retrieval_output = gr.Textbox(
                label="Top retrieved chunks",
                lines=25
            )

            retrieval_button.click(
                fn=debug_retrieval,
                inputs=retrieval_question,
                outputs=retrieval_output
            )

            retrieval_question.submit(
                fn=debug_retrieval,
                inputs=retrieval_question,
                outputs=retrieval_output
            )