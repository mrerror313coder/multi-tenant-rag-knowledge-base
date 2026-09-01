"""Authentication and tenant isolation middleware for FastAPI."""

from typing import Optional
from fastapi import Header, HTTPException, status, Depends, Query
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import Organization


def get_current_tenant(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key", description="Organization API Key"),
    authorization: Optional[str] = Header(None, description="Bearer API Key"),
    api_key_query: Optional[str] = Query(None, alias="api_key", description="Optional query param API Key"),
    db: Session = Depends(get_db),
) -> Organization:
    """Extracts and validates tenant API key, binding request context to authenticated Organization."""
    token = None

    if x_api_key:
        token = x_api_key.strip()
    elif authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "").strip()
    elif api_key_query:
        token = api_key_query.strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Organization API Key. Provide 'X-API-Key' header or 'Bearer <key>'.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    org = db.query(Organization).filter(Organization.api_key == token).first()
    if not org:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Organization API Key. Access denied.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return org
