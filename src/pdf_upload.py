# -------------------- standard library --------------------
import os
from datetime import datetime

# -------------------- LangChain loaders --------------------
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
)

# -------------------- LangChain text splitting --------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------- LangChain embeddings --------------------
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------- LangChain vectorstore (IMPORTANT: community) --------------------
from langchain_community.vectorstores import FAISS

def load_documents(file_path):
    try:
        file_path = _to_path(file_path)
        print("load_document type:", type(file_path), file_path)

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return []

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            loader = PyMuPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".csv":
            loader = CSVLoader(file_path)
        else:
            print(f"Skipping unsupported file type: {file_path}")
            return []

        loaded_docs = loader.load()
        print(f"Loaded docs from {file_path}: {len(loaded_docs)}")

        clean_docs = []
        for i, doc in enumerate(loaded_docs):
            text = getattr(doc, "page_content", "")
            text = "" if text is None else str(text).strip()
            print(f"Doc {i} text length: {len(text)}")
            if text:
                doc.page_content = text
                clean_docs.append(doc)

        print(f"Final readable docs count: {len(clean_docs)}")
        return clean_docs

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []
    

def _to_path(x):
    if isinstance(x, list):
        x = x[0] if x else ""
    return os.path.abspath(os.path.normpath(x))


def upload_and_index(file_path, state, save_dir="stores/pdf"):
    state = state or {}

    if not file_path:
        return "❌ No file received.", state

    try:
        file_path = _to_path(file_path)
        print("upload_and_index type:", type(file_path), file_path)

        os.makedirs(save_dir, exist_ok=True)

        documents = load_documents(file_path)
        print("Readable documents after filtering:", len(documents))

        if not documents:
            return "❌ No readable text found in uploaded file.", state

        chunks = chunk_documents(documents)
        print("Chunks created:", len(chunks))

        if not chunks:
            return "❌ No chunks created from uploaded file.", state

        plan_summary = "\n\n".join(
            doc.page_content for doc in documents[:5]
            if getattr(doc, "page_content", None)
        )[:8000]

        raw_text = "\n\n".join(
            doc.page_content for doc in documents
            if getattr(doc, "page_content", None)
        )
        pdf_raw_preview = raw_text[:15000]

        embedding_kind = "sentence-transformers/all-MiniLM-L6-v2"
        doc_embeddings_local = HuggingFaceEmbeddings(model_name=embedding_kind)

        vs = FAISS.from_documents(documents=chunks, embedding=doc_embeddings_local)

        vs_path = os.path.abspath(os.path.join(save_dir, "faiss_index"))
        vs.save_local(vs_path)

        state["document"] = {
    "vectorstore_path": vs_path,
    "embedding_kind": embedding_kind,
    "plan_summary": plan_summary,
    "raw_preview": pdf_raw_preview,
    "meta": {
        "n_files": 1,
        "n_docs": len(documents),
        "n_chunks": len(chunks),
        "created_at": str(datetime.now()),
        "file_paths": [file_path],
    },
    "uploaded": True,
    "file_paths": [file_path],
}

        return f"✅ Uploaded 1 file; created {len(chunks)} chunks.", state

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error during file processing: {e}", state

def chunk_documents(documents, chunk_size=800, overlap=100):
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print("Chunks after splitting:", len(chunks))
    for i, chunk in enumerate(chunks[:5]):
        print(f"Chunk {i} length: {len(chunk.page_content)}")

    return chunks

# FUTURE EXTENSION
# # Simple Retrieval
# def query_pdf(message, state, llm, k=4):
#     vs = load_document_vectorstore(state)
#     if vs is None:
#         return "⚠️ Please upload and process files first."

#     docs = vs.similarity_search(message, k=k)
#     context = "\n\n".join(d.page_content for d in docs)

#     plan_summary = (state or {}).get("plan_summary", "")
#     profile = (state or {}).get("user_profile", {})

#     prompt = f"""
# You are a fitness assistant answering questions about the uploaded plan.
# Avoid LaTeX. Use profile if relevant.

# User profile:
# - Age: {profile.get("age")}
# - Height: {profile.get("height")}
# - Weight: {profile.get("weight")}
# - Fitness Level: {profile.get("fitness_level")}
# - Gender: {profile.get("gender")}
# - Health issues: {profile.get("health_issues")}

# Plan preview:
# {plan_summary}

# Relevant retrieved parts:
# {context}

# User question:
# {message}
# """
#     return llm.invoke(prompt).content.strip()