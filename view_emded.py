from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load the FAISS vector store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("rag_models/faiss_index", embedding, allow_dangerous_deserialization=True)

# View embedded document chunks
docs = db.similarity_search("show me stored text", k=5)

for i, doc in enumerate(docs, start=1):
    print(f"\n🔹 Document Chunk {i}:\n{doc.page_content}")
