import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load the Code Files
print("Loading files from Python-Files directory...")
loader = DirectoryLoader("./Python-Files", glob="**/*.py", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
documents = loader.load()

# 2. The "Smart" Chunking (Semantic/AST-Aware)
# Prioritizes splitting at classes and functions before standard text.
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
)
semantic_chunks = python_splitter.split_documents(documents)
print(f"Created {len(semantic_chunks)} semantic chunks.")

# 3. Setup Local Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create and Save the Semantic ChromaDB
print("Building Semantic Vector Database...")
Chroma.from_documents(
    documents=semantic_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_semantic",
)
print("Semantic Database saved to './chroma_db_semantic'!")
