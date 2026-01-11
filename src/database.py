import os
import glob
import logging
import json
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- PATH CONFIGURATION ---
# Railway uses /app as workdir. os.getcwd() is the safest reference.
BASE_DIR = os.getcwd() 
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
DATA_PATH = os.path.join(BASE_DIR, "data")
MAPPING_EXPORT_PATH = os.path.join(CHROMA_PATH, "embeddings_mapping.json")

# ENSURE DIRECTORIES EXIST IMMEDIATELY
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

logger = logging.getLogger("blossom_agent.database")

# Global variables for singleton instances
_CHROMA_CLIENT = None
_RETRIEVER_INSTANCE = None

def get_chroma_client():
    """Lazy initialization of the Chroma client to prevent start-up errors."""
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        try:
            # Code 14 fix: Ensure path is absolute and exists
            _CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_PATH)
            logger.info(f"ChromaDB PersistentClient initialized at {CHROMA_PATH}")
        except Exception as e:
            logger.error(f"Critical error initializing ChromaDB: {e}")
            raise
    return _CHROMA_CLIENT

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

def run_ingestion(force_rebuild: bool = False) -> bool:
    db_file = os.path.join(CHROMA_PATH, "chroma.sqlite3")
    if not force_rebuild and os.path.exists(db_file):
        logger.info("Local Chroma DB found. Skipping ingestion.")
        return True

    logger.info("Starting ingestion process...")
    documents = _load_pdf_documents()
    if not documents:
        logger.warning("No documents found to ingest.")
        return False

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")),
        client=get_chroma_client(),
        collection_name="blossom_security_v1"
    )
    
    _export_embeddings_dictionary(chunks)
    logger.info(f"Success: Vector store persisted in {CHROMA_PATH}.")
    return True

def get_active_retriever():
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        vector_db = Chroma(
            client=get_chroma_client(),
            collection_name="blossom_security_v1",
            embedding_function=OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")),
        )
        _RETRIEVER_INSTANCE = vector_db.as_retriever(search_kwargs={"k": 3})
    return _RETRIEVER_INSTANCE