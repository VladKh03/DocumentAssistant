import gradio as gr

from rag.pipeline import RAGPipeline

rag = RAGPipeline(top_k=5)

try:
    rag.load_index()
    print("Existing index loaded successfully")
except FileNotFoundError:
    print("No saved index found")
except Exception as e:
    print(f"Failed to load saved index: {e}")

def process_document(file):
    if file is None:
        return "Please upload a PDF first."

    try:
        info = rag.index_pdf(file.name)

        return (
            "Document processed successfully\n\n"
            f"Pages: {info['pages']}\n"
            f"Created chunks: {info['chunks']}"
        )

    except Exception as e:
        return f"Error while processing document:\n{str(e)}"


def ask_document(question):
    if not question.strip():
        return "", ""

    try:
        result = rag.ask(question)

        answer = result["answer"]

        sources = "\n".join(
            f"Page {page}"
            for page in result["sources"]
        )

        return answer, sources

    except Exception as e:
        return f"Error:\n{str(e)}", ""


with gr.Blocks(title="AI Document Assistant") as demo:

    gr.Markdown(
        """
        # AI Document Assistant

        Upload a PDF and ask questions about its content.
        """
    )

    with gr.Tabs():

        with gr.Tab("Chat"):

            question = gr.Textbox(
                label="Question",
                placeholder="Ask something about the document...",
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
                lines=4
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

            document = gr.File(
                label="Upload PDF",
                file_types=[".pdf"]
            )

            process_button = gr.Button(
                "Process document",
                variant="primary"
            )

            status = gr.Textbox(
                label="Status",
                lines=5
            )

            process_button.click(
                fn=process_document,
                inputs=document,
                outputs=status
            )