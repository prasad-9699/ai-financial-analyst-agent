"""
Generalized PDF RAG (Retrieval-Augmented Generation) pipeline.

Works with ANY PDF document — financial reports, research papers, contracts,
manuals, articles, etc. Not limited to any specific document type.

Includes automatic OCR fallback for scanned PDFs using Tesseract.
"""

import logging
from typing import List, Optional

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
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
        try:
            _embeddings_cache = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            # Fallback: if meta tensor error persists, load with safetensors disabled
            logger.warning("Default loading failed (%s), trying fallback...", e)
            import torch
            _embeddings_cache = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={
                    "device": "cpu",
                    "torch_dtype": torch.float32,
                },
                encode_kwargs={"normalize_embeddings": True},
            )
        logger.info("Embedding model loaded successfully")
    return _embeddings_cache


def _pages_have_text(documents: List[Document]) -> bool:
    """
    Check whether PyPDFLoader extracted meaningful text.

    Returns False if all pages are empty or whitespace-only,
    which typically indicates a scanned (image-only) PDF.
    """
    for doc in documents:
        content = doc.page_content.strip()
        if len(content) > 50:  # At least 50 chars of real text
            return True
    return False


def _extract_text_with_ocr(file_path: str, tesseract_cmd: str = "tesseract") -> List[Document]:
    """
    Extract text from a scanned PDF using Tesseract OCR.

    Pipeline: PDF → page images (pdf2image) → OCR text (pytesseract)

    Args:
        file_path: Path to the PDF file.
        tesseract_cmd: Path to the Tesseract binary.

    Returns:
        List of Document objects with OCR-extracted text.

    Raises:
        RAGProcessingError: If OCR extraction fails.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path

        # Set Tesseract command path if custom
        if tesseract_cmd and tesseract_cmd != "tesseract":
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        logger.info("Starting OCR extraction for scanned PDF: %s", file_path)

        # Convert PDF pages to images
        images = convert_from_path(file_path, dpi=300)
        logger.info("Converted PDF to %d page images for OCR", len(images))

        documents = []
        for page_num, image in enumerate(images, start=1):
            # Run OCR on each page image
            text = pytesseract.image_to_string(image, lang="eng")
            text = text.strip()

            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "page": page_num,
                        "extraction_method": "ocr",
                    },
                ))
                logger.debug("OCR page %d: extracted %d characters", page_num, len(text))
            else:
                logger.warning("OCR page %d: no text extracted", page_num)

        if not documents:
            raise RAGProcessingError(
                "OCR could not extract any text from the scanned PDF. "
                "The document may contain only images without readable text, "
                "or the scan quality may be too low."
            )

        logger.info("OCR completed: extracted text from %d/%d pages", len(documents), len(images))
        return documents

    except ImportError as e:
        raise RAGProcessingError(
            "OCR dependencies are not installed. Please install pytesseract and pdf2image:\n"
            "  pip install pytesseract pdf2image Pillow\n"
            "Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract"
        ) from e
    except RAGProcessingError:
        raise
    except Exception as e:
        logger.error("OCR extraction failed: %s", e)
        raise RAGProcessingError(
            f"OCR processing failed. Ensure Tesseract is installed and on your PATH.\n"
            f"Details: {e}"
        ) from e


def process_pdf(file_bytes: bytes, config: AppConfig) -> FAISS:
    """
    Process a PDF file into a FAISS vector store for retrieval.

    This is a generalized pipeline that works with ANY PDF — financial reports,
    research papers, legal documents, manuals, articles, etc.

    For scanned PDFs (image-only), automatically falls back to Tesseract OCR.

    Pipeline: Load → [OCR fallback if needed] → Split → Embed → Store

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

        # Step 2: Try loading PDF pages with PyPDFLoader (text-based PDFs)
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Step 3: Check if text was actually extracted
        extraction_method = "text"
        if not documents or not _pages_have_text(documents):
            # Scanned PDF detected — fall back to OCR
            logger.warning(
                "PyPDFLoader extracted no readable text (%d pages). "
                "Falling back to Tesseract OCR...",
                len(documents) if documents else 0,
            )
            documents = _extract_text_with_ocr(temp_path, config.tesseract_cmd)
            extraction_method = "ocr"

        logger.info(
            "PDF loaded via %s: %d pages with text",
            extraction_method, len(documents),
        )

        # Step 4: Split into chunks
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

        # Step 5: Create embeddings and vector store
        embeddings = _get_embeddings(config.embedding_model)
        vectorstore = FAISS.from_documents(chunks, embeddings)

        logger.info("FAISS vector store created with %d vectors", len(chunks))
        return vectorstore

    except RAGProcessingError:
        raise
    except Exception as e:
        logger.error("PDF processing failed: %s", e)
        raise RAGProcessingError(
            f"Failed to process the PDF. Please ensure it's a valid PDF file.\n"
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

        # Strip <think>...</think> reasoning blocks from model output
        import re
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

        logger.info("PDF query completed successfully")
        return answer

    except Exception as e:
        logger.error("PDF query failed: %s", e)
        raise RuntimeError(
            f"I had trouble searching the document. This might be a temporary issue. "
            f"Please try again.\n\nDetails: {e}"
        ) from e

