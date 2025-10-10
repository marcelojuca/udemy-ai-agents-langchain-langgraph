from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [doc for doc_list in docs for doc in doc_list]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
docs_splits = text_splitter.split_documents(docs_list)

# vectorstore = Chroma.from_documents(
#     documents=docs_splits,
#     collection_name="rag_chroma",
#     embedding=OpenAIEmbeddings(),
#     persist_directory="./.chroma_db",
# )

retriever = Chroma(
    collection_name="rag_chroma",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./.chroma_db",
).as_retriever()


def main():
    print("Hi")
