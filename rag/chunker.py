def create_chunks(pages, chunk_size=300, overlap=50):
    chunks = []

    chunk_id = 0

    for page in pages:
        page_number = page["page"]
        text = page["text"]

        words = text.split()

        start = 0

        while start < len(words):
            end = start + chunk_size

            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            if chunk_text.strip():
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "text": chunk_text
                })

                chunk_id += 1

            start += chunk_size - overlap

    return chunks