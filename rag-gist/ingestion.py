import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader("mediumblog1.txt", encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Ingesting...")
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=os.environ["PINECONE_INDEX_NAME"])
    print("Done!")