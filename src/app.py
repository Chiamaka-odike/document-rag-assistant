import hashlib
import os

import chromadb
import pymupdf
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


load_dotenv()

st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="📄"
)

st.title("📄 Document RAG Assistant")
st.write("Upload a PDF and ask questions about its contents.")


# -----------------------------
# Load embedding model
# -----------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# -----------------------------
# Connect to OpenRouter
# -----------------------------

api_key = os.getenv("OPENROUTER_API_KEY")

llm_client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# -----------------------------
# PDF extraction
# -----------------------------

def extract_pages(pdf_bytes):
    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text().strip()

        if text:
            pages.append({
                "text": text,
                "page": page_number
            })

    return pages


# -----------------------------
# Chunking
# -----------------------------

def create_chunks(pages, chunk_size=500, overlap=50):
    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            start += chunk_size - overlap

    return chunks


# -----------------------------
# Build vector store
# -----------------------------

def build_vector_store(chunks):
    if not chunks:
        return None

    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="document_chunks"
    )

    existing = collection.get()

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(texts).tolist()

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    metadata = [
        {"page": chunk["page"]}
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadata
    )

    return collection


# -----------------------------
# Retrieve relevant chunks
# -----------------------------

def retrieve_chunks(collection, question, number_of_results=5):
    # Add a simple semantic hint for common document questions.
    retrieval_question = question

    lower_question = question.lower()

    if "work from home" in lower_question:
        retrieval_question = (
            question + " remote working work remotely flexible working"
        )

    elif "vacation" in lower_question:
        retrieval_question = question + " annual leave holidays time off"

    elif "sick" in lower_question or "ill" in lower_question:
        retrieval_question = question + " sick leave illness medical absence"

    question_embedding = embedding_model.encode(
        retrieval_question
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_results
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return documents, metadatas


# -----------------------------
# Generate answer
# -----------------------------

def generate_answer(question, documents):
    context = "\n\n".join(documents)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information in the context.

Do not show reasoning or analysis.
Return only the final answer.

If the context does not contain enough information, say:
"I could not find that information in the document."

Context:
{context}

Question:
{question}
"""

    response = llm_client.chat.completions.create(
        model="poolside/laguna-s-2.1:free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# -----------------------------
# Upload document
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    # Create an ID for this particular uploaded file.
    file_id = hashlib.md5(pdf_bytes).hexdigest()

    # Only process the PDF once per uploaded file.
    if st.session_state.get("file_id") != file_id:

        pages = extract_pages(pdf_bytes)
    chunks = create_chunks(pages)

    collection = build_vector_store(chunks)

    if collection is None:
        st.error(
            "I couldn't extract readable text from this PDF. "
            "Please upload a text-based PDF."
        )
        st.stop()

    st.session_state.file_id = file_id
    st.session_state.collection = collection
    st.session_state.pages = len(pages)
    st.session_state.chunks = len(chunks)

else:
    collection = st.session_state.collection
    st.success(
        f"Document processed: "
        f"{st.session_state.pages} pages, "
        f"{st.session_state.chunks} chunks."
    )


    question = st.text_input(
        "Ask a question about the document"
    )


    if question:

        documents, metadatas = retrieve_chunks(
            collection,
            question
        )

        answer = generate_answer(
            question,
            documents
        )

        st.subheader("Answer")
        st.write(answer)

        source_pages = sorted(
            set(metadata["page"] for metadata in metadatas)
        )

        st.caption(
            "Retrieved source pages: "
            + ", ".join(str(page) for page in source_pages)
        )