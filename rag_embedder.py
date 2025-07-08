import os
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def process_file(file_path):
    # Load the document
    if file_path.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")  # 🛠 Fix encoding issue
    else:
        print("❌ Unsupported file type. Please upload a .pdf or .txt file.")
        return

    # Load content
    try:
        documents = loader.load()
    except Exception as e:
        print(f"❌ Failed to load document: {e}")
        return

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Embedding model
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create and save FAISS index
    db = FAISS.from_documents(chunks, embedding)
    
    # Ensure output directory exists
    os.makedirs("rag_models", exist_ok=True)
    db.save_local("rag_models/faiss_index")

    print("✅ File processed and vector store saved successfully!")

if __name__ == "__main__":
    file_path = input("📂 Enter the full path to your .pdf or .txt file: ").strip()

    if not os.path.exists(file_path):
        print("❌ File does not exist.")
    else:
        process_file(file_path)
