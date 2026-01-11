import os
import glob
import logging
import json
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

CHROMA_PATH = "./chroma_db"
DATA_PATH = "./data"
# Path for the descriptive dictionary
MAPPING_EXPORT_PATH = os.path.join(CHROMA_PATH, "embeddings_mapping.json")

logger = logging.getLogger("blossom_agent.database")
_RETRIEVER_INSTANCE = None
_CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_PATH)

ALLOWED_DOCS = [
    "Magic Training — Back Office",
    "Magic Training (Member admin)",
    "Magic Phone Banking Training",
    "Personal Banking Training",
    "Login — Security items",
    "Sign Up (Set Up Online Access)",
    "Detailed_Security_Protocol"
]

def _get_tags(text: str) -> str:
    """Extract tags and return as a comma-separated string."""
    mapping = {
        "password": ["password", "reset", "credential"],
        "lockout": ["lockout", "locked", "attempts", "suspend"],
        "mfa": ["verification", "code", "mfa", "otp", "prompt"],
        "remember_me": ["remember me", "cadence", "session"],
        "username": ["username", "id", "login name"]
    }
    text_lower = text.lower()
    found = [tag for tag, keywords in mapping.items() if any(k in text_lower for k in keywords)]
    return ", ".join(found) if found else ""

def _load_pdf_documents():
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    documents = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        if not any(doc_name in filename for doc_name in ALLOWED_DOCS):
            continue
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            for d in docs:
                d.metadata = {
                    "source": filename,
                    "page": d.metadata.get("page", 0) + 1,
                    "tags": _get_tags(d.page_content)
                }
            documents.extend(docs)
        except Exception as e:
            logger.error(f"Error loading {pdf_path}: {e}")
    return documents

def _export_embeddings_dictionary(chunks):
    """Generates a JSON file mapping each chunk to its source and summary."""
    mapping = []
    for i, chunk in enumerate(chunks):
        mapping.append({
            "chunk_id": i,
            "source": chunk.metadata.get("source"),
            "page": chunk.metadata.get("page"),
            "tags": chunk.metadata.get("tags"),
            "preview": chunk.page_content[:150].replace("\n", " ") + "..."
        })
    
    with open(MAPPING_EXPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)
    logger.info(f"Embeddings dictionary exported to {MAPPING_EXPORT_PATH}")

def run_ingestion(force_rebuild: bool = False) -> bool:
    """
    Persists embeddings in a local folder. 
    If chroma.sqlite3 exists, it skips the process to save costs/time.
    """
    os.makedirs(CHROMA_PATH, exist_ok=True)
    db_exists = os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3"))
    
    if not force_rebuild and db_exists:
        logger.info("Local Chroma DB found. Skipping ingestion to reuse existing embeddings.")
        return True

    logger.info("Initializing ingestion: Creating local embeddings and dictionary...")
    documents = _load_pdf_documents()
    if not documents:
        logger.warning("No documents found to ingest.")
        return False

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Save to local folder using the shared client
    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")),
        client=_CHROMA_CLIENT,
        collection_name="blossom_security_v1"
    )
    
    # Generate the readable JSON dictionary
    _export_embeddings_dictionary(chunks)
    
    logger.info(f"Success: Vector store persisted in {CHROMA_PATH}.")
    return True

def get_active_retriever():
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        vector_db = Chroma(
            client=_CHROMA_CLIENT,
            collection_name="blossom_security_v1",
            embedding_function=OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")),
        )
        _RETRIEVER_INSTANCE = vector_db.as_retriever(search_kwargs={"k": 2})
    return _RETRIEVER_INSTANCE