import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load the Code Files
print("Loading files from Python-Files directory...")
loader = DirectoryLoader("./Python-Files", glob="**/*.py", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
documents = loader.load()
print(f"Loaded {len(documents)} Python files.")

# 2. The "Dumb" Chunking (Baseline)
# Cuts strictly by character count, often breaking functions in half.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
baseline_chunks = text_splitter.split_documents(documents)
print(f"Created {len(baseline_chunks)} baseline chunks.")

# 3. Setup Local Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create and Save the Baseline ChromaDB
print("Building Baseline Vector Database...")
Chroma.from_documents(
    documents=baseline_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_baseline",
)
print("Baseline Database saved to './chroma_db_baseline'!")
