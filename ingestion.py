import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    ##################  Ingestion: Loading data ##################
    print("-----"*10)
    print("Ingestion: Started")
    print("Lesson 48")
    print("Ingestion: Loading data...")
    loader = TextLoader("medium-blog.txt")
    document = loader.load()
    print(document)

    ##################  Ingestion: Splitting data ##################
    print("-----" * 10)
    print("Ingestion: Splitting data...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(document)
    print(chunks)

    ##################  Ingestion: Embedding data ##################
    print("-----" * 10)
    print("Ingestion: Embedding data...")
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
    )
    print(embeddings)

    ##################  Ingestion: Saving data ##################
    print("-----" * 10)
    print("Ingestion: Saving data...")
    vectorstore = PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=os.getenv("PINECONE_INDEX_NAME")
    )
    print(vectorstore)

    ##################  Ingestion: Finished ##################
    print("-----" * 10)
    print("Ingestion: Finished")
