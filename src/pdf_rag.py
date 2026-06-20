"""
Generalized PDF RAG (Retrieval-Augmented Generation) pipeline.

Works with ANY PDF document — financial reports, research papers, contracts,
manuals, articles, etc. Not limited to any specific document type.
"""

import logging
from typing import Optional

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import AppConfig, RAGProcessingError
from src.utils import cleanup_temp_file, save_temp_file

logger = logging.getLogger("financial_analyst.pdf_rag")

# Module-level cache for the embedding model (expensive to load)
_embeddings_cache: Optional[HuggingFaceEmbeddings] = None


def _get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Get or create a cached embeddings model instance."""
    global _embeddings_cache
    if _embeddings_cache is None:
        logger.info("Loading embedding model: %s (first load may take 1-2 min)", model_name)
        _embeddings_cache = HuggingFaceEmbeddings(model_name=model_name)
        logger.info("Embedding model loaded successfully")
    return _embeddings_cache


def process_pdf(file_bytes: bytes, config: AppConfig) -> FAISS:
    """
    Process a PDF file into a FAISS vector store for retrieval.

    This is a generalized pipeline that works with ANY PDF — financial reports,
    research papers, legal documents, manuals, articles, etc.

    Pipeline: Load → Split → Embed → Store

    Args:
        file_bytes: Raw bytes of the uploaded PDF.
        config: Application configuration.

    Returns:
        A FAISS vector store ready for retrieval.

    Raises:
        RAGProcessingError: If any step of the pipeline fails.
    """
    temp_path = None

    try:
        # Step 1: Save to temp file (PyPDFLoader needs a file path)
        temp_path = save_temp_file(file_bytes, suffix=".pdf")

        # Step 2: Load PDF pages
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        if not documents:
            raise RAGProcessingError(
                "The PDF appears to be empty or could not be read. "
                "Please check that it contains selectable text (not just scanned images)."
            )

        logger.info("PDF loaded: %d pages", len(documents))

        # Step 3: Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise RAGProcessingError("Could not extract any text chunks from the PDF.")

        logger.info("Text split into %d chunks (size=%d, overlap=%d)",
                     len(chunks), config.chunk_size, config.chunk_overlap)

        # Step 4: Create embeddings and vector store
        embeddings = _get_embeddings(config.embedding_model)
        vectorstore = FAISS.from_documents(chunks, embeddings)

        logger.info("FAISS vector store created with %d vectors", len(chunks))
        return vectorstore

    except RAGProcessingError:
        raise
    except Exception as e:
        logger.error("PDF processing failed: %s", e)
        raise RAGProcessingError(
            f"Failed to process the PDF. Please ensure it's a valid, text-based PDF.\n"
            f"Details: {e}"
        ) from e
    finally:
        if temp_path:
            cleanup_temp_file(temp_path)


def query_pdf(question: str, vectorstore: FAISS, llm: ChatGroq, top_k: int = 4) -> str:
    """
    Query a PDF knowledge base using RAG.

    Works with any document type — the LLM answers based purely on the
    retrieved context, not assumptions about the document type.

    Args:
        question: The user's natural-language question.
        vectorstore: FAISS vector store built from the PDF.
        llm: The LLM instance.
        top_k: Number of relevant chunks to retrieve.

    Returns:
        The LLM's answer based on retrieved document context.

    Raises:
        RuntimeError: If the query fails.
    """
    logger.info("PDF query: %s", question[:100])

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
        )
        response = rag_chain.invoke({"query": question})
        answer = response.get("result", "")

        if not answer:
            return (
                "I searched the document but couldn't find a clear answer to your question. "
                "Try rephrasing or asking about a specific section."
            )

        logger.info("PDF query completed successfully")
        return answer

    except Exception as e:
        logger.error("PDF query failed: %s", e)
        raise RuntimeError(
            f"I had trouble searching the document. This might be a temporary issue. "
            f"Please try again.\n\nDetails: {e}"
        ) from e
