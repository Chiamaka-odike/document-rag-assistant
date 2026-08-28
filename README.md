# Document RAG Assistant

A simple Retrieval-Augmented Generation (RAG) application that lets users upload a PDF and ask questions about the document.

I built this project to understand how a basic RAG pipeline works from start to finish, including document processing, text chunking, embeddings, vector search, and LLM-based answer generation.

## How it works

```text
PDF
 ↓
PyMuPDF
 ↓
Text extraction
 ↓
Chunking
 ↓
Sentence Transformer
 ↓
Embeddings
 ↓
ChromaDB
 ↓
Similarity search
 ↓
OpenRouter LLM
 ↓
Answer
```

When a PDF is uploaded, the application extracts its text using PyMuPDF and splits it into smaller chunks.

Each chunk is converted into an embedding using `all-MiniLM-L6-v2` and stored in ChromaDB together with its page number.

When a user asks a question, the question is also converted into an embedding. ChromaDB searches for the most similar document chunks, and those chunks are passed to an LLM through OpenRouter to generate the answer.

## Built With

* Python
* Streamlit
* PyMuPDF
* Sentence Transformers
* ChromaDB
* OpenRouter
* python-dotenv

## Features

* Upload a PDF and ask questions about it
* Extract text from PDF documents
* Split documents into smaller chunks
* Generate text embeddings locally
* Retrieve relevant document sections using vector similarity
* Generate answers using retrieved context
* Show the pages associated with retrieved information
* Handle PDFs where no readable text can be extracted

## Project Structure

```text
document-rag-assistant/
├── data/
│   └── employee_handbook.pdf
├── src/
│   ├── app.py
│   ├── pdf_reader.py
│   ├── rag.py
│   ├── search.py
│   └── vector_store.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Running the Project

### Clone the repository

```bash
git clone https://github.com/Chiamaka-odike/document-rag-assistant.git
cd document-rag-assistant
```

### Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

### Install the dependencies

```bash
pip install -r requirements.txt
```

### Add your API key

Create a `.env` file in the project root:

```text
OPENROUTER_API_KEY=your_api_key_here
```

### Run the application

```bash
streamlit run src/app.py
```

## Example

**Question:**
Can I work from home?

**Answer:**
Yes, remote working options may be available to eligible employees, subject to approval based on role requirements and individual performance.

## Limitations

The current version uses simple character-based chunking, so some chunks may split sentences or sections awkwardly.

The application works best with text-based PDFs. Scanned PDFs without selectable text are not currently supported.

## What I Learned

This project helped me get practical experience with:

* PDF text extraction
* Document chunking
* Text embeddings
* Vector databases
* Semantic search
* Retrieval-Augmented Generation
* Integrating an LLM API
* Building a simple AI application with Streamlit
