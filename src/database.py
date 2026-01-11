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

# ENSURE DIRECTORIES EXIST IMMEDIATELY
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

logger = logging.getLogger("blossom_agent.database")

# Global variables for singleton instances
_RETRIEVER_INSTANCE = None
_CHROMA_CLIENT = None

def get_chroma_client():
    """Crea el cliente solo cuando se solicita, evitando el Error 14 al importar."""
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        # Aseguramos que la carpeta exista con permisos justo antes de abrirla
        os.makedirs(CHROMA_PATH, exist_ok=True)
        try:
            _CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_PATH)
            logger.info(f"ChromaDB initialized successfully at {CHROMA_PATH}")
        except Exception as e:
            logger.error(f"Error opening ChromaDB: {e}")
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