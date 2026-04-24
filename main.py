# ================================
# Azure RAG Project (Single + Multi Document)
# ================================

import os
import hashlib
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Match the existing Azure AI Search index schema. These can still be
# overridden in .env for a different index.
os.environ.setdefault("AZURESEARCH_FIELDS_ID", "chunk_id")
os.environ.setdefault("AZURESEARCH_FIELDS_CONTENT", "chunk")
os.environ.setdefault("AZURESEARCH_FIELDS_CONTENT_VECTOR", "text_vector")
os.environ.setdefault("AZURESEARCH_FIELDS_TAG", "title")

# ================================
# IMPORTS
# ================================
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.azuresearch import AzureSearch

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


# ================================
# LOAD ENV VARIABLES
# ================================
BASE_DIR = Path(__file__).resolve().parent

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")


# Validate environment variables
def validate_env_variables():
    """Check if all required environment variables are set"""
    required_vars = {
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
        "AZURE_OPENAI_CHAT_DEPLOYMENT": CHAT_DEPLOYMENT,
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": EMBEDDING_DEPLOYMENT,
        "AZURE_SEARCH_API_KEY": AZURE_SEARCH_API_KEY,
        "AZURE_SEARCH_ENDPOINT": AZURE_SEARCH_ENDPOINT,
        "AZURE_SEARCH_INDEX_NAME": INDEX_NAME
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        print("ERROR: Missing environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease configure these in your .env file")
        exit(1)


def _chunk_id(chunk, chunk_index):
    """
    Create a stable Azure Search document key for each chunk.
    Re-running the pipeline updates the same chunks instead of duplicating them.
    """
    source = chunk.metadata.get("source", "")
    page = chunk.metadata.get("page", "")
    raw = f"{source}|{page}|{chunk_index}|{chunk.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _content_hash(text):
    """Create a stable hash for chunk text to support source fallback matching."""
    normalized = " ".join((text or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ================================
# STEP 1: LOAD DOCUMENTS
# ================================
def load_documents(folder_path="data"):
    """
    Load all PDFs from the data folder
    Works for single and multiple documents
    """
    folder = Path(folder_path)
    if not folder.is_absolute():
        folder = BASE_DIR / folder

    if not folder.exists():
        raise FileNotFoundError(f"Document folder not found: {folder}")

    documents = []

    for file in sorted(folder.iterdir()):
        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
            docs = loader.load()
            documents.extend(docs)

    if not documents:
        raise ValueError(f"No PDF documents found in {folder}")

    print(f"Loaded {len(documents)} pages from documents")
    return documents

# ================================
# STEP 2: SPLIT INTO CHUNKS
# ================================
def split_documents(documents):
    """
    Split documents into smaller chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks, start=1):
        # Keep user-friendly source metadata for answer attribution.
        page = chunk.metadata.get("page")
        chunk.metadata["chunk_number"] = idx
        chunk.metadata["page_number"] = (
            page + 1 if isinstance(page, int) else page
        )
        source = chunk.metadata.get("source", "")
        chunk.metadata["source_file"] = Path(source).name if source else "Unknown"

    print(f"Created {len(chunks)} chunks")
    return chunks

# ================================
# STEP 3: CREATE EMBEDDINGS
# ================================
def create_embeddings():
    """
    Create embedding model from Azure OpenAI
    """
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        deployment=EMBEDDING_DEPLOYMENT,
        openai_api_version=AZURE_OPENAI_API_VERSION
    )
    return embeddings

# ================================
# STEP 4: STORE IN AZURE AI SEARCH
# ================================
def create_vector_store(chunks, embeddings):
    """
    Store embeddings in Azure AI Search
    """
    try:
        # Create vector store with proper authentication
        vector_store = AzureSearch(
            azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
            azure_search_key=AZURE_SEARCH_API_KEY,
            index_name=INDEX_NAME,
            embedding_function=embeddings
        )

        print(f"Uploading {len(chunks)} chunks to Azure AI Search...")
        ids = [_chunk_id(chunk, i) for i, chunk in enumerate(chunks)]
        chunk_source_map = {}
        for i, chunk in enumerate(chunks):
            doc_id = ids[i]
            chunk_source_map[doc_id] = {
                "chunk_id": doc_id,
                "chunk_number": chunk.metadata.get("chunk_number"),
                "page_number": chunk.metadata.get("page_number"),
                "source_file": chunk.metadata.get("source_file", "Unknown"),
                "content_hash": _content_hash(chunk.page_content),
            }
        vector_store.add_documents(chunks, ids=ids)

        print(f"Vector store initialized and indexed {len(ids)} chunks")
        return vector_store, chunk_source_map
    except Exception as e:
        print(f"ERROR creating vector store: {str(e)}")
        print("\nNote: Make sure your Azure Search index is properly configured.")
        print("The index should have:")
        print(f"  - {os.getenv('AZURESEARCH_FIELDS_ID')}: Edm.String key field")
        print(f"  - {os.getenv('AZURESEARCH_FIELDS_CONTENT')}: searchable Edm.String field")
        print(f"  - {os.getenv('AZURESEARCH_FIELDS_CONTENT_VECTOR')}: searchable vector field with your embedding dimensions")
        print(f"  - {os.getenv('AZURESEARCH_FIELDS_TAG')}: searchable Edm.String metadata field")
        raise

# ================================
# STEP 5: CREATE RETRIEVER
# ================================
def create_retriever(vector_store):
    """
    Retriever fetches top K relevant chunks
    """
    retriever = vector_store.as_retriever(
        k=3, # Top 3 chunks
        search_kwargs={
            "select": [
                os.getenv("AZURESEARCH_FIELDS_ID"),
                os.getenv("AZURESEARCH_FIELDS_CONTENT"),
            ]
        },
    )
    return retriever

# ================================
# STEP 6: CREATE LLM
# ================================
def create_llm():
    """
    Azure OpenAI Chat Model
    """
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        deployment_name=CHAT_DEPLOYMENT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        temperature=0
    )
    return llm

# ================================
# STEP 7: CREATE RAG CHAIN
# ================================
def create_rag_chain(retriever, llm, chunk_source_map=None):
    # 1. Define Prompt
    prompt = ChatPromptTemplate.from_template(
        "Answer based only on the context:\n\n{context}\n\nQuestion: {input}"
    )

    id_field = os.getenv("AZURESEARCH_FIELDS_ID", "chunk_id")

    def extract_chunk_id(metadata):
        return (
            metadata.get(id_field)
            or metadata.get("id")
            or metadata.get("chunk_id")
        )

    content_hash_map = {
        meta.get("content_hash"): meta
        for meta in (chunk_source_map or {}).values()
        if isinstance(meta, dict) and meta.get("content_hash")
    }

    def extract_source_file(metadata):
        for key in [
            "source_file",
            "file_name",
            "filename",
            "metadata_storage_name",
            os.getenv("AZURESEARCH_FIELDS_TAG", "title"),
            "title",
            "source",
        ]:
            value = metadata.get(key)
            if value:
                return Path(str(value)).name
        return "Unknown"

    def build_sources(docs):
        sources = []
        for i, doc in enumerate(docs, start=1):
            metadata = doc.metadata or {}
            chunk_id = extract_chunk_id(metadata)
            lookup = (chunk_source_map or {}).get(chunk_id, {})
            if not lookup:
                lookup = content_hash_map.get(_content_hash(doc.page_content), {})

            page = lookup.get("page_number")
            if page is None:
                page = metadata.get("page_number")
            if page is None:
                raw_page = metadata.get("page")
                page = raw_page + 1 if isinstance(raw_page, int) else raw_page

            source_info = {
                "rank": i,
                "chunk_id": chunk_id or "Unknown",
                "chunk_number": lookup.get("chunk_number")
                or metadata.get("chunk_number")
                or "Unknown",
                "page_number": page if page is not None else "Unknown",
                "source_file": lookup.get("source_file")
                or extract_source_file(metadata),
            }
            sources.append(source_info)
        return sources

    def retrieve(inputs):
        docs = retriever.invoke(inputs["input"])
        return {
            "input": inputs["input"],
            "context": "\n\n".join(doc.page_content for doc in docs),
            "source_documents": docs,
            "sources": build_sources(docs),
        }

    prompt_inputs = RunnableLambda(
        lambda inputs: {"input": inputs["input"], "context": inputs["context"]}
    )
    answer_chain = prompt_inputs | prompt | llm | StrOutputParser()
    retrieval_chain = RunnableLambda(retrieve) | RunnablePassthrough.assign(
        answer=answer_chain
    )

    return retrieval_chain

def build_ragas_dataset(session_rows):
    """
    Build a Ragas dataset using the current in-memory chat session rows.
    """
    from datasets import Dataset

    return Dataset.from_list(session_rows)


#===============================
# RAGAS EVALUATION
#===============================

def enrich_rows_with_reference(session_rows, llm):
    """
    Ensure every row has a reference answer so reference-based Ragas metrics
    can run on session data.
    """
    reference_prompt = ChatPromptTemplate.from_template(
        "You are generating a ground-truth style reference answer for evaluation.\n"
        "Use only the provided context. If context is insufficient, say so briefly.\n\n"
        "Question: {question}\n\n"
        "Context:\n{context}\n\n"
        "Reference answer:"
    )
    reference_chain = reference_prompt | llm | StrOutputParser()

    enriched_rows = []
    for row in session_rows:
        question = row.get("user_input", "")
        contexts = row.get("retrieved_contexts", []) or []
        context_text = "\n\n".join(contexts)

        reference = row.get("reference")
        if not reference:
            reference = reference_chain.invoke(
                {"question": question, "context": context_text}
            )

        enriched_rows.append(
            {
                "user_input": question,
                "response": row.get("response", ""),
                "retrieved_contexts": contexts,
                "reference": reference,
            }
        )

    return enriched_rows


def run_ragas_evaluation(session_rows, llm, embeddings):
    """
    Evaluate the RAG pipeline with Ragas metrics.
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    )

    if not session_rows:
        print(
            "No Q/A history found for evaluation. Ask at least one question before "
            "running Ragas evaluation."
        )
        return None

    print("Preparing evaluation dataset...")
    dataset_rows = enrich_rows_with_reference(session_rows, llm)
    dataset = build_ragas_dataset(dataset_rows)
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    print("\nRunning Ragas evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=False,
    )

    print("\nRagas Evaluation Results:")
    try:
        result_dict = dict(result)
        for metric_name, score in result_dict.items():
            if isinstance(score, float):
                print(f"{metric_name} - {score:.4f}")
            else:
                print(f"{metric_name} - {score}")
    except Exception:
        print(result)

    return result

# ================================
# MAIN EXECUTION
# ================================
def main(run_evaluation=False):
    print("Starting Azure RAG Pipeline...\n")

    # Validate environment variables first
    validate_env_variables()

    # Initialize resources
    embeddings = None
    vector_store = None
    chunk_source_map = {}
    llm = None
    session_evaluation_rows = []

    try:
        # Step 1: Load documents
        documents = load_documents()

        # Step 2: Split into chunks
        chunks = split_documents(documents)

        # Step 3: Create embeddings
        embeddings = create_embeddings()

        # Step 4: Store in Azure AI Search
        vector_store, chunk_source_map = create_vector_store(chunks, embeddings)

        # Step 5: Create retriever
        retriever = create_retriever(vector_store)

        # Step 6: Create LLM
        llm = create_llm()

        # Step 7: Create RAG chain
        qa_chain = create_rag_chain(retriever, llm, chunk_source_map=chunk_source_map)

        if run_evaluation:
            run_ragas_evaluation(session_evaluation_rows, llm, embeddings)
            return

        print("\n RAG System Ready! Ask questions (type 'exit' to quit)\n")
        print("Type 'evaluate' to run Ragas evaluation.\n")

        # Step 8: Ask questions
        while True:
            query = input("Ask a question: ")

            if query.lower() == "exit":
                break

            if query.lower() == "evaluate":
                run_ragas_evaluation(session_evaluation_rows, llm, embeddings)
                print("\n" + "="*50 + "\n")
                continue

            response = qa_chain.invoke({"input": query})
            source_documents = response.get("source_documents", [])
            sources = response.get("sources", [])
            session_evaluation_rows.append(
                {
                    "user_input": query,
                    "response": response["answer"],
                    "retrieved_contexts": [
                        doc.page_content for doc in source_documents
                    ],
                    "retrieved_sources": sources,
                }
            )

            print("\n Answer:")
            print(response["answer"])
            answer_text = response["answer"].lower()
            not_in_document = any(
                phrase in answer_text
                for phrase in [
                    "does not contain any information",
                    "context is insufficient",
                    "insufficient context",
                    "not available in the context",
                    "not in the context",
                ]
            )
            if sources or not_in_document:
                print("\n Source Details:")
                if not_in_document:
                    print("- File Name: Not available in the document")
                    print("  Page Number: Not available in the document")
                else:
                    primary_source = next(
                        (
                            s for s in sources
                            if s.get("source_file") != "Unknown"
                            or s.get("page_number") != "Unknown"
                        ),
                        sources[0],
                    )
                    print(f"- File Name: {primary_source['source_file']}")
                    print(f"  Page Number: {primary_source['page_number']}")
            print("\n" + "="*50 + "\n")

    except Exception as e:
        print(f"ERROR during execution: {str(e)}")
        raise
    finally:
        # Clean up resources
        print("Cleaning up resources...")
        try:
            if vector_store:
                # Close any connections in the vector store
                if hasattr(vector_store, 'client') and vector_store.client:
                    vector_store.client.close()
                # Additional cleanup for Azure Search
                if hasattr(vector_store, '_client'):
                    vector_store._client.close()
        except Exception as e:
            print(f"Warning: Error during vector store cleanup: {e}")

        try:
            if llm:
                # Close LLM client if it has a close method
                if hasattr(llm, 'close'):
                    llm.close()
        except Exception as e:
            print(f"Warning: Error during LLM cleanup: {e}")

        try:
            if embeddings:
                # Close embeddings client if it has a close method
                if hasattr(embeddings, 'close'):
                    embeddings.close()
        except Exception as e:
            print(f"Warning: Error during embeddings cleanup: {e}")

        print("Cleanup completed.")

# ================================
# RUN
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Azure RAG pipeline")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run Ragas evaluation after building the RAG pipeline, then exit",
    )
    args = parser.parse_args()

    main(run_evaluation=args.evaluate)
