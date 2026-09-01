"""Database and vector store cleanup script.
Purges all organizations, documents, chunks, logs, and resets ChromaDB collections.
"""

import os
import sys
import shutil
import logging

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.session import SessionLocal, Base, engine
from app.db.models import Organization, Document, Chunk, QueryLog
from app.retrieval.service import get_vector_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clean_db")


def reset_entire_database():
    """Wipes all database tables, uploaded files, and vector store data."""
    logger.info("Starting complete database and tenant purge...")

    # 1. Clear SQLite tables
    db = SessionLocal()
    try:
        db.query(QueryLog).delete()
        db.query(Chunk).delete()
        db.query(Document).delete()
        db.query(Organization).delete()
        db.commit()
        logger.info("Purged all SQLite records (organizations, documents, chunks, logs).")
    except Exception as e:
        logger.warning(f"Error purging SQLite tables: {e}")
        db.rollback()
    finally:
        db.close()

    # 2. Reset ChromaDB collection
    try:
        vs = get_vector_service()
        if vs._collection:
            try:
                vs._client.delete_collection(vs.COLLECTION_NAME)
            except Exception:
                pass
            vs._collection = vs._client.get_or_create_collection(
                name=vs.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        logger.info("ChromaDB vector collection reset successfully.")
    except Exception as e:
        logger.warning(f"Error resetting ChromaDB: {e}")

    # 3. Clean uploaded documents folder
    uploads_dir = "./uploaded_documents"
    if os.path.exists(uploads_dir):
        try:
            for item in os.listdir(uploads_dir):
                item_path = os.path.join(uploads_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                elif os.path.isfile(item_path):
                    os.remove(item_path)
            logger.info("Cleaned uploaded_documents directory.")
        except Exception as e:
            logger.warning(f"Error cleaning uploads directory: {e}")

    logger.info("All tenants, documents, and vector data have been completely deleted.")


if __name__ == "__main__":
    reset_entire_database()
