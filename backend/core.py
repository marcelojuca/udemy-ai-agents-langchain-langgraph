import os
from typing import Any

from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

from typing import Any, Dict, List

from langchain import hub
from langchain.chains.combine_documents import \
    create_stuff_documents_chain  # executes the augmentation of the retrieved documents
from langchain.chains.history_aware_retriever import \
    create_history_aware_retriever
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


# retrieval function that retrieves the documents from the vector store
def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0, model="gpt-4o-mini")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(llm=chat, prompt=retrieval_qa_chat_prompt)

    # lesson 60, minute 04:30
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(llm=chat, prompt=rephrase_prompt,
        retriever=vector_store.as_retriever()
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )
    result = retrieval_chain.invoke({"input": query, "chat_history": chat_history})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result


if __name__ == "__main__":
    res = run_llm(query="What is LangChain chain?")
    print(res["result"])
