import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


if __name__ == "__main__":
    print("OpenAI LLM response with RAG FAISS - lesson 46")
    print("Loading data...")
    pdf_path = "/Users/mhcj/Downloads/GitHub/langchain-course/2210.03629v3.pdf"
    loader = PyPDFLoader(pdf_path)
    doc = loader.load()
    print(f"Loaded {len(doc)} pages")

    print("Splitting data...")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    chunks = text_splitter.split_documents(doc)
    print(f"Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
    )
    print("Embedding data...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"Embedded {len(chunks)} chunks")

    print("Saving data...")
    vectorstore.save_local("faiss_index")
    print("Saved data...")

    print("Loading data...")
    new_vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # print(f"Loaded {len(new_vectorstore.index.ntotal)} chunks")

    # print("Querying data...")
    # query = "What is the main idea of the paper?"
    # docs = new_vectorstore.similarity_search(query)
    # print(docs)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini"),
        prompt=retrieval_qa_chat_prompt,
    )
    retrieval_chain = create_retrieval_chain(
        retriever=new_vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    result = retrieval_chain.invoke(
        input={"input": "Give me the gist of ReAct in 3 sentences"}
    )
    print(result["answer"])
