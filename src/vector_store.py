import chromadb
from sentence_transformers import SentenceTransformer
from pdf_reader import extract_text, create_chunks


# Connect to our local Chroma database
client = chromadb.PersistentClient(path="./chroma_db")

# Create or open a collection for our document chunks
collection = client.get_or_create_collection(name="documents")

# Load the embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


# Get pages and chunks from our PDF
pdf_path = "data/employee_handbook.pdf"

pages = extract_text(pdf_path)
chunks = create_chunks(pages)


# Store all chunks in Chroma
for index, chunk in enumerate(chunks):

    chunk_text = chunk["text"]
    page_number = chunk["page"]

#creating the embedding and puttimg it in a list of 384 vectors
    embedding = model.encode(chunk_text).tolist()

    collection.add(
        ids=[f"chunk_{index}"],
        documents=[chunk_text],
        embeddings=[embedding],
        metadatas=[{"page": page_number}]
    )


print("Number of chunks stored:", len(chunks))