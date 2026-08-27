import chromadb
from sentence_transformers import SentenceTransformer


client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(name="documents")

model = SentenceTransformer("all-MiniLM-L6-v2")


question = "Can I work from home?"

question_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

print("Question:")
print(question)

print("\nRetrieved chunks:")

for i, document in enumerate(results["documents"][0]):
    print(f"\nResult {i + 1}:")
    print(document)