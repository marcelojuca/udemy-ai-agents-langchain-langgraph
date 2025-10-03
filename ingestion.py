import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingesting data...")
    loader = TextLoader("medium-blog.txt")
    document = loader.load()
    print(document)

    print("-----" * 10)
    print("Splitting data...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(document)
    print(chunks)

    print("-----" * 10)
    print("Embedding data...")
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
    )
    print(embeddings)

    print("-----" * 10)
    print("Ingesting data...")
    vectorstore = PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=os.getenv("PINECONE_INDEX_NAME")
    )
    print(vectorstore)

    print("-----" * 10)
    print("Finished ingestion...")
