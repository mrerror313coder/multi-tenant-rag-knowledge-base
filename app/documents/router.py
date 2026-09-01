"""Document management API endpoints for tenant document upload, status, and chunking."""

import os
import shutil
import logging
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import Organization, Document, Chunk
from app.schemas.models import DocumentResponse, DocumentStatusResponse, ChunkDetail
from app.auth.middleware import get_current_tenant
from app.retrieval.service import get_vector_service
from services.chunking import RecursiveChunker
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["Documents"])

UPLOAD_BASE_DIR = "./uploaded_documents"
os.makedirs(UPLOAD_BASE_DIR, exist_ok=True)


def process_document_pipeline(document_id: str, org_id: str, file_bytes: bytes, filename: str, content_type: str):
    """Background or synchronous worker processing chunking and vector embedding."""
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id, Document.org_id == org_id).first()
        if not doc:
            return

        doc.status = "chunking"
        db.commit()

        chunker = RecursiveChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks = chunker.extract_and_chunk(file_bytes, filename=filename, content_type=content_type)

        if not chunks:
            doc.status = "failed"
            doc.error_message = "No readable text content could be extracted from document."
            db.commit()
            return

        doc.status = "embedding"
        db.commit()

        # Save chunks in Relational DB
        db_chunks = []
        for c in chunks:
            db_chunk = Chunk(
                document_id=document_id,
                org_id=org_id,
                chunk_index=c.chunk_index,
                text=c.text,
                token_count=c.token_count,
                page_number=c.page_number,
            )
            db_chunks.append(db_chunk)
            db.add(db_chunk)

        # Upsert into ChromaDB Vector Store with mandatory org_id filter
        vector_service = get_vector_service()
        vector_service.add_document_chunks(
            org_id=org_id,
            document_id=document_id,
            filename=filename,
            chunks=chunks,
        )

        doc.status = "ready"
        doc.chunk_count = len(chunks)
        doc.error_message = None
        db.commit()
        logger.info(f"Document {filename} ({document_id}) processed successfully: {len(chunks)} chunks.")

    except Exception as e:
        logger.exception(f"Document processing failed for {filename}: {e}")
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = "failed"
            doc.error_message = str(e)
            db.commit()
    finally:
        db.close()


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tenant: Organization = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Uploads a PDF, Markdown, or TXT document and triggers chunking and embedding pipeline."""
    filename = file.filename or "uploaded_document.txt"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in [".pdf", ".txt", ".md", ".markdown", ".csv", ".json"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported: .pdf, .txt, .md, .csv, .json",
        )

    # Read content
    content_bytes = await file.read()
    file_size = len(content_bytes)

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Save to disk
    tenant_upload_dir = os.path.join(UPLOAD_BASE_DIR, tenant.id)
    os.makedirs(tenant_upload_dir, exist_ok=True)
    saved_path = os.path.join(tenant_upload_dir, filename)

    with open(saved_path, "wb") as f:
        f.write(content_bytes)

    # Create document record
    doc = Document(
        org_id=tenant.id,
        filename=filename,
        file_path=saved_path,
        file_type=ext.replace(".", ""),
        file_size=file_size,
        status="pending",
        chunk_count=0,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Trigger processing
    process_document_pipeline(
        document_id=doc.id,
        org_id=tenant.id,
        file_bytes=content_bytes,
        filename=filename,
        content_type=file.content_type or "",
    )

    db.refresh(doc)
    return doc


@router.get("/", response_model=List[DocumentResponse])
def list_documents(
    tenant: Organization = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Lists all knowledge base documents belonging to the authenticated organization."""
    return db.query(Document).filter(Document.org_id == tenant.id).order_by(Document.created_at.desc()).all()


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
def get_document_status(
    document_id: str,
    tenant: Organization = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Retrieves document indexing and embedding status."""
    doc = db.query(Document).filter(Document.id == document_id, Document.org_id == tenant.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found for this organization.")
    return doc


@router.get("/{document_id}/chunks", response_model=List[ChunkDetail])
def get_document_chunks(
    document_id: str,
    tenant: Organization = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Inspects chunked segments of a document."""
    doc = db.query(Document).filter(Document.id == document_id, Document.org_id == tenant.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found for this organization.")

    chunks = db.query(Chunk).filter(Chunk.document_id == document_id, Chunk.org_id == tenant.id).order_by(Chunk.chunk_index.asc()).all()
    return chunks


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(
    document_id: str,
    tenant: Organization = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Deletes document and removes all associated vectors from ChromaDB."""
    doc = db.query(Document).filter(Document.id == document_id, Document.org_id == tenant.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found for this organization.")

    # Remove vectors
    vector_service = get_vector_service()
    vector_service.delete_document_chunks(document_id=doc.id, org_id=tenant.id)

    # Delete local file
    if os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except Exception as e:
            logger.warning(f"Could not remove local file {doc.file_path}: {e}")

    # Remove from relational DB (cascades chunks)
    db.delete(doc)
    db.commit()

    return {"message": f"Document '{doc.filename}' and all associated vector embeddings deleted successfully."}
