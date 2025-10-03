
import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    
if __name__ == "__main__":
    print("-----"*10)
    print("OpenAI LLM response w/o Pinecone data...")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
    query = "what is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result)

    print("-----"*10)
    print("OpenAI LLM response retrieving Pinecone data...")

    vectorstore = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)
    result = retrieval_chain.invoke(input={"input": query})
    print(result)

    print("-----"*10)
    print("RAG chain...")

    template = """Use the following pieces of retrieved context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer concise as possible.
    Always say "Thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template=template)
    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )
    result = rag_chain.invoke(query)
    print(result)

    print("-----"*10)
    print("Finished RAG chain...")