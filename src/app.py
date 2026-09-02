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
    page_title="Parser",
    page_icon="◈",
    layout="centered"
)

st.markdown(
    """
    <style>

    /* Page */
    .stApp {
        background-color: #f6f7fb;
    }

    .block-container {
        max-width: 850px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    /* Remove some default Streamlit elements */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Header */
    .header {
        text-align: center;
        margin-bottom: 2.5rem;
    }

    .logo {
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }

    .title {
        font-size: 2.7rem;
        font-weight: 750;
        color: #18181b;
        margin: 0;
    }

    .subtitle {
        margin-top: 0.6rem;
        color: #667085;
        font-size: 1rem;
    }

    /* Section titles */
    .section-label {
        font-size: 0.9rem;
        font-weight: 650;
        color: #344054;
        margin-bottom: 0.5rem;
    }
    /* Upload section */

.upload-heading {
    font-size: 1rem;
    font-weight: 650;
    color: #344054;
    margin-bottom: 0.6rem;
}

[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #cfd4dc;
    border-radius: 18px;
    padding: 1.5rem;
    box-shadow: 0 3px 14px rgba(16, 24, 40, 0.04);
}

/* Uploaded document card */

.document-card {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: white;
    border: 1px solid #e4e7ec;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    margin-bottom: 2rem;
    box-shadow: 0 3px 14px rgba(16, 24, 40, 0.04);
}

.document-icon {
    font-size: 1.8rem;
}

.document-info {
    flex: 1;
}

.document-name {
    font-size: 0.95rem;
    font-weight: 600;
    color: #18181b;
}

.document-meta {
    margin-top: 0.25rem;
    font-size: 0.82rem;
    color: #667085;
}

.document-meta span {
    margin: 0 0.3rem;
}

.document-status {
    font-size: 0.8rem;
    font-weight: 600;
    color: #087443;
    background: #ecfdf3;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
}

/* Question input */

div[data-testid="stTextInput"] input {
    background-color: white;
    color: #18181b;
    caret-color: #18181b;
    border: 1px solid #d0d5dd;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    font-size: 1rem;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #98a2b3;
}


div[data-testid="stTextInput"] input:focus {
    border-color: #667085;
    box-shadow: 0 0 0 1px #667085;
    color: #18181b;
}

/* Upload button text */
[data-testid="stFileUploader"] button {
    color: #18181b !important;
}

/* Ask Parser button */
button[kind="formSubmit"] {
    color: #ffffff !important;
}

/* Global text color */

.stApp, 
.stApp p,
.stApp label,
.stMarkdown,
.stText,
.stCaption,
.stTextInput,
.stTextInput label {
    color: #18181b;
}
/* Make all Streamlit buttons readable */

.stButton > button {
    background-color: #18181b !important;
    color: #ffffff !important;
    border: 1px solid #18181b !important;
    border-radius: 12px !important;
}

.stButton > button p {
    color: #ffffff !important;
}


/* File uploader button */

[data-testid="stFileUploader"] button {
    background-color: #ffffff !important;
    color: #18181b !important;
    border: 1px solid #d0d5dd !important;
}

[data-testid="stFileUploader"] button p {
    color: #18181b !important;
}
/* Form submit button */

[data-testid="stFormSubmitButton"] button {
    background-color: #18181b !important;
    color: #ffffff !important;
    border: 1px solid #18181b !important;
    border-radius: 12px !important;
}

[data-testid="stFormSubmitButton"] button p {
    color: #ffffff !important;
}


/* File uploader text */

[data-testid="stFileUploader"] {
    color: #18181b !important;
}

[data-testid="stFileUploader"] p {
    color: #18181b !important;
}

[data-testid="stFileUploader"] span {
    color: #18181b !important;
}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="header">
        <div class="logo">◈</div>
        <div class="title">Parser</div>
        <div class="subtitle">
            Ask questions. Find answers in your documents.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
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
    ],
    extra_body={
        "models": [
            "poolside/laguna-s-2.1:free",
            "openai/gpt-oss-20b:free",
            "openrouter/free"
        ]
    }
)
    return response.choices[0].message.content


# -----------------------------
# Upload document
# -----------------------------

st.markdown(
    """
    <div class="upload-heading">Upload a document</div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Drop your PDF here or browse your files",
    type=["pdf"],
    label_visibility="collapsed"
)


if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    # Create an ID for this particular uploaded file.
    file_id = hashlib.md5(pdf_bytes).hexdigest()

    # Only process the PDF if it is a new upload.
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
        # Reuse the existing collection for the same document.
        collection = st.session_state.collection


    st.success(
    f"✓ {uploaded_file.name} — "
    f"{st.session_state.pages} pages • "
    f"{st.session_state.chunks} chunks"
)

    st.markdown(
    "### Ask a question"
)

   
with st.form("question_form"):
    question = st.text_input(
        "Ask something about your document...",
        placeholder="e.g. Can I work from home?",
        label_visibility="collapsed"
    )

    submitted = st.form_submit_button("Ask Parser")


if submitted and question:

    # st.write("✅ Question submitted:", question)

    with st.spinner("Searching your document..."):

        documents, metadatas = retrieve_chunks(
            collection,
            question
        )

    # st.write("✅ Retrieved", len(documents), "chunks")

    with st.spinner("Generating answer..."):

        try:
            answer = generate_answer(
                question,
                documents
            )

            # st.write("✅ LLM responded")

        except Exception as e:
            st.error("LLM error:")
            st.exception(e)
            st.stop()

    st.markdown("### Answer")

    with st.container(border=True):
        st.write(answer)

    source_pages = sorted(
        set(metadata["page"] for metadata in metadatas)
    )

    st.markdown("**Sources**")

    source_text = " · ".join(
        f"Page {page}" for page in source_pages
    )

    st.caption(source_text)