import pymupdf
from sentence_transformers import SentenceTransformer

def extract_text(pdf_path):
    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        pages.append({
            "text": text,
            "page": page_number
        })

    return pages


def create_chunks(pages, chunk_size=500, overlap=50):
    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "page": page_number
            })

            start += chunk_size - overlap

    return chunks

# finds and opens my pdf
pdf_path = "data/employee_handbook.pdf"

# this extracts the pdf into pages
pages = extract_text(pdf_path)

#breakes my pages into the chunksss
chunks = create_chunks(pages)

# loads the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

#takes the first chunk
first_chunk = chunks[0]["text"]

#converts the first chunk into a vector
embedding = model.encode(first_chunk)




print()
print("Embedding:")
print(embedding)

print()
print("Embedding length:", len(embedding))

print("Number of pages:", len(pages))
print("Number of chunks:", len(chunks))
print()
print("First chunk:")
print(chunks[0])

