import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


load_dotenv()

# Connect to our local Chroma database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="documents")

# Load the same embedding model used for our document chunks
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to OpenRouter
api_key = os.getenv("OPENROUTER_API_KEY")

llm_client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


question = "Can I work from home?"

# Convert the question into a vector
question_embedding = embedding_model.encode(question).tolist()

# Retrieve the 3 most relevant chunks
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

retrieved_chunks = results["documents"][0]

# Combine the retrieved chunks into one piece of context
context = "\n\n".join(retrieved_chunks)

# Build the prompt for the LLM
prompt = f"""
Answer the user's question using only the information in the context below.

Context:
{context}

Question:
{question}

If the answer is not contained in the context, say:
"I could not find that information in the document."
"""

# Send the context + question to the LLM
response = llm_client.chat.completions.create(
    model="openrouter/free",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

answer = response.choices[0].message.content

print("Question:")
print(question)

print("\nAnswer:")
print(answer)