import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

print("=" * 55)
print("  DAY 5 - PDF RAG Pipeline")
print("=" * 55)

# ── Setup LLM ─────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ══════════════════════════════════════════════════════
# STEP 1 — Document Loader (PDF load karna)
# ══════════════════════════════════════════════════════
print("\n[Step 1] Loading PDF...")

pdf_path = "data/annual_report.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"  PDF loaded successfully!")
print(f"  Total pages: {len(documents)}")
print(f"  First 200 chars: {documents[0].page_content[:200]}")

# ══════════════════════════════════════════════════════
# STEP 2 — Text Splitter (Chunks mein todna)
# ══════════════════════════════════════════════════════
print("\n[Step 2] Splitting text into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Har chunk ka size
    chunk_overlap=200,    # Chunks ke beech overlap (context preserve karne ke liye)
)

chunks = text_splitter.split_documents(documents)

print(f"  Total chunks created: {len(chunks)}")
print(f"  Sample chunk: {chunks[0].page_content[:150]}...")

# ══════════════════════════════════════════════════════
# STEP 3 — Embeddings (Text ko numbers mein convert karna)
# ══════════════════════════════════════════════════════
print("\n[Step 3] Creating embeddings (this may take 1-2 min first time)...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("  Embedding model loaded!")

# ══════════════════════════════════════════════════════
# STEP 4 — Vector Store (FAISS mein store karna)
# ══════════════════════════════════════════════════════
print("\n[Step 4] Storing embeddings in FAISS vector store...")

vectorstore = FAISS.from_documents(chunks, embeddings)

print("  FAISS vector store created!")
print(f"  Total vectors stored: {len(chunks)}")

# Optional: Save to disk for reuse later
vectorstore.save_local("faiss_index")
print("  Vector store saved to disk (faiss_index folder)")

# ══════════════════════════════════════════════════════
# STEP 5 — Retriever (Query ke liye relevant chunks dhundna)
# ══════════════════════════════════════════════════════
print("\n[Step 5] Creating retriever...")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Top 3 relevant chunks

print("  Retriever ready!")

# ══════════════════════════════════════════════════════
# STEP 6 — RAG Chain (Retrieval + LLM combine karna)
# ══════════════════════════════════════════════════════
print("\n[Step 6] Building RAG chain...")

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

print("  RAG chain ready!")
print("=" * 55)

# ══════════════════════════════════════════════════════
# TEST — PDF Se Questions Poochna
# ══════════════════════════════════════════════════════
test_questions = [
    "What is this document about?",
    "Summarize the key points in 2-3 sentences",
]

for i, question in enumerate(test_questions, 1):
    print(f"\nQ{i}: {question}")
    print("-" * 45)
    try:
        response = rag_chain.invoke({"query": question})
        print(f"Answer: {response['result']}")
        print(f"\nSources used: {len(response['source_documents'])} chunks")
    except Exception as e:
        print(f"Error: {e}")
    print("=" * 55)

print("\n Day 5 Complete!")
print(" PDF successfully converted into searchable knowledge base!")
print(" RAG pipeline working: Load -> Split -> Embed -> Store -> Retrieve -> Answer")