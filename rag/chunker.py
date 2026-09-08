def create_chunks(
    pages,
    document_name,
    chunk_size=300,
    overlap=50,
    start_chunk_id=0
):
    chunks = []

    chunk_id = start_chunk_id

    for page in pages:
        page_number = page["page"]
        words = page["text"].split()

        start = 0

        while start < len(words):
            end = start + chunk_size

            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            if chunk_text.strip():
                chunks.append({
                    "chunk_id": chunk_id,
                    "document": document_name,
                    "page": page_number,
                    "text": chunk_text
                })

                chunk_id += 1

            start += chunk_size - overlap

    return chunks