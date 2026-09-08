import pymupdf

def parse_pdf(file_path):
    document = pymupdf.open(file_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        pages.append({
            "page": page_number,
            "text": page.get_text()
        })

    document.close()

    return pages