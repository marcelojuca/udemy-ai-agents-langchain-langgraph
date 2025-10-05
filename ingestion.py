import asyncio
import os
from re import M
import ssl
from typing import Any, Dict, List

import certifi
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

# Configure SSL context to handle certificate verification issues
def configure_ssl():
    """Configure SSL context with proper certificate handling."""
    try:
        # Try to use certifi certificates first
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
        log_info("SSL: Using certifi certificates")
        return ssl_context
    except Exception as e:
        log_warning(f"SSL: Failed to use certifi certificates: {e}")
        try:
            # Fallback to system certificates
            ssl_context = ssl.create_default_context()
            log_info("SSL: Using system certificates")
            return ssl_context
        except Exception as e2:
            log_warning(f"SSL: Failed to use system certificates: {e2}")
            # Last resort: disable verification (not recommended for production)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            log_warning("SSL: Disabled certificate verification (not recommended for production)")
            return ssl_context

# Configure SSL
ssl_context = configure_ssl()

# Configure requests to use the same SSL configuration
requests.adapters.DEFAULT_RETRIES = 3
session = requests.Session()
session.verify = certifi.where() if certifi.where() else True

embeddings = OpenAIEmbeddings( model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10)

# for local vector store
# vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
# for online vector store
vector_store = PineconeVectorStore(index_name="langchain-docs-2025", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    """Chunk URLs into smaller lists of a specified size."""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i : i + chunk_size]
        chunks.append(chunk)
    return chunks


async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Coroutine to extract documents from a batch of URLs."""
    try:
        log_info(f"TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs")
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        results = docs.get("results", [])
        log_success(f"TavilyExtract: Batch {batch_num} completed - extracted {len(results)} documents")
        return results
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num}: {e}")
        return []


async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(f"Starting to extract documents from {len(url_batches)} batches of URLs")

    # coroutine to extract documents from each batch concurrently
    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exception and flatten the results
    all_pages = []
    failed_batches = 0
    for batch_result in batch_results:
        if isinstance(batch_result, Exception):
            log_error(
                f"TavilyExtract: Failed to extract batch {batch_result}"
            )
            failed_batches += 1
        else:
            for extracted_page in batch_result:
                document = Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]},
                )
                all_pages.append(document)

    log_success(f"TavilyExtract: Extraction complete - Successfully extracted {len(all_pages)} documents from {len(url_batches)} batches")

    if failed_batches > 0:
        log_warning(f"TavilyExtract: Failed to extract {failed_batches} batches")
        log_warning(f"TavilyExtract: Consider increasing the batch size or retrying failed batches")
    
    return all_pages


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches and index them into the vector store."""
    log_info(f"VectorStore Indexing: Preparing to add {len(documents)} documents into the vector store")

    # Create a list of batches
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    log_info(f"VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each")

    # Coroutine to process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            log_info(f"VectorStore Indexing: Adding batch {batch_num} of {len(batch)} documents")
            await vector_store.aadd_documents(batch)
            log_success(f"VectorStore Indexing: Batch {batch_num} added successfully")
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num}: {e}")
            raise e

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    log_success(f"VectorStore Indexing: Added {len(documents)} documents into the vector store")

    # Count successful batches
    successful_batches = sum(1 for result in results if not isinstance(result, Exception))
    log_success(f"VectorStore Indexing: Successfully added {successful_batches} batches of {batch_size} documents each")

    # Count failed batches
    failed_batches = sum(1 for result in results if isinstance(result, Exception))
    log_warning(f"VectorStore Indexing: Failed to add {failed_batches} batches of {batch_size} documents each")
    log_warning(f"VectorStore Indexing: Consider increasing the batch size or retrying failed batches")


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info("🔍 TavilyMap: Starting to map documentation from https://python.langchain.com/")
    site_map = tavily_map.invoke("https://python.langchain.com/")
    log_success(f"TavilyMap: Successfully mapped {len(site_map['results'])} URLs from documentation site")
    
    
    # Split URLs into chunks of 20
    url_batches = chunk_urls(site_map["results"], 20)
    log_info(f"URL processing: Splitting {len(site_map['results'])} URLs into {len(url_batches)} batches of 20")

    
    # Extract documents from batches concurrently
    all_docs = await async_extract(url_batches)

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(f"Text splitter: Splitting {len(all_docs)} documents into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Text splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents.")

    # Process documents asynchronously in batches and index them into the vector store
    log_header("VECTOR STORAGE PHASE")
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("DOCUMENTATION INGESTION PIPELINE COMPLETED")
    log_success("🎉 Documentation ingestion pipeline completed successfully")
    log_info("Summary:")
    log_info(f"Total URLs processed: {len(site_map['results'])}")
    log_info(f"Total documents extracted: {len(all_docs)}")
    log_info(f"Total chunks created: {len(splitted_docs)}")
    


    # log_info("🔍 TavilyCrawl: Starting to crawl documentation from https://python.langchain.com/")

    # res = tavily_crawl.invoke(
    #     {
    #         "url": "https://python.langchain.com/",
    #         "max_depth": 1,
    #         "extract_depth": "advanced",
    #     }
    # )
    # all_docs = [
    #     Document(page_content=result["raw_content"], metadata={"source": result["url"]})
    #     for result in res["results"]
    # ]
    # log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site")


if __name__ == "__main__":
    asyncio.run(main())
