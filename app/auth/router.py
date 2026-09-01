"""Organization onboarding and API key management routes."""

import secrets
import re
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import Organization, Document
from app.schemas.models import OrganizationCreate, OrganizationResponse

router = APIRouter(prefix="/api/organizations", tags=["Organizations"])


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_-]+", "_", text)


@router.post("/", response_model=OrganizationResponse, status_code=status.HTTP_201_CREATED)
def create_organization(
    payload: OrganizationCreate,
    db: Session = Depends(get_db),
):
    """Registers a new organization tenant and generates a secure API key."""
    org_id = payload.org_id or slugify(payload.name)
    if not org_id:
        org_id = f"org_{secrets.token_hex(4)}"

    # Check if org_id exists
    existing = db.query(Organization).filter(Organization.id == org_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Organization with ID '{org_id}' already exists. Choose a unique name/ID.",
        )

    api_key = f"sk_{org_id}_{secrets.token_urlsafe(16)}"
    org = Organization(
        id=org_id,
        name=payload.name,
        api_key=api_key,
    )
    db.add(org)
    db.commit()
    db.refresh(org)

    return OrganizationResponse(
        id=org.id,
        name=org.name,
        api_key=org.api_key,
        created_at=org.created_at,
        document_count=0,
    )


@router.get("/", response_model=List[OrganizationResponse])
def list_organizations(db: Session = Depends(get_db)):
    """Lists all registered organizations for the demo / switcher UI."""
    orgs = db.query(Organization).all()
    results = []
    for org in orgs:
        doc_count = db.query(Document).filter(Document.org_id == org.id).count()
        results.append(
            OrganizationResponse(
                id=org.id,
                name=org.name,
                api_key=org.api_key,
                created_at=org.created_at,
                document_count=doc_count,
            )
        )
    return results


@router.delete("/{org_id}", status_code=status.HTTP_200_OK)
def delete_organization(
    org_id: str,
    db: Session = Depends(get_db),
):
    """Deletes an organization, purging its documents, database chunks, and vector store embeddings."""
    import os
    import shutil
    from app.retrieval.service import get_vector_service

    org = db.query(Organization).filter(Organization.id == org_id).first()
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Organization with ID '{org_id}' not found.",
        )

    # 1. Purge all vector embeddings for this tenant
    vector_service = get_vector_service()
    vector_service.delete_tenant_vectors(org_id=org_id)

    # 2. Clean up uploaded files directory
    tenant_dir = os.path.join("./uploaded_documents", org_id)
    if os.path.exists(tenant_dir):
        try:
            shutil.rmtree(tenant_dir)
        except Exception:
            pass

    # 3. Delete from relational database (cascade deletes documents, chunks, query logs)
    org_name = org.name
    db.delete(org)
    db.commit()

    return {
        "message": f"Organization '{org_name}' ({org_id}) and all associated documents, chunks, and vector embeddings were deleted successfully.",
        "deleted_org_id": org_id,
    }

