from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain # executes the augmentation of the retrieved documents
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

INDEX_NAME = "langchain-docs-2025"

# retrieval function that retrieves the documents from the vector store
def run_llm(query: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0, model="gpt-5-mini")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    
    retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), combine_docs_chain=stuff_documents_chain)
    return retrieval_chain.invoke({"input": query})


if __name__ == "__main__":
    res = run_llm(query = "What is LangChain chain?")
    print(res["answer"])
