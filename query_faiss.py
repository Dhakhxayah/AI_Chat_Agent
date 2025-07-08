from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load the embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the saved FAISS vector store
db = FAISS.load_local("rag_models/faiss_index", embedding, allow_dangerous_deserialization=True)

while True:
    query = input("\n❓ Enter your question (or type 'exit' to quit): ").strip()

    if query.lower() == "exit":
        print("👋 Exiting...")
        break

    # Perform similarity search
    results = db.similarity_search(query, k=3)

    print("\n🔎 Top Relevant Chunks:")
    for i, doc in enumerate(results, start=1):
        print(f"\n--- Chunk {i} ---\n{doc.page_content}")
