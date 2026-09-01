"""FastAPI Application entry point for Multi-Tenant RAG Knowledge Base."""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.db.session import engine, Base, SessionLocal
from app.db.models import Organization, Document, Chunk
from app.auth.router import router as auth_router
from app.documents.router import router as documents_router, process_document_pipeline
from app.chat.router import router as chat_router
from app.eval.router import router as eval_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_app")


def seed_default_tenants_and_docs(force: bool = False):
    """Seeds initial demonstration tenants and sample documents only if configured."""
    if not settings.SEED_DEMO_DATA and not force:
        return
    db = SessionLocal()
    try:
        # 1. Acme Corporation
        org_a = db.query(Organization).filter(Organization.id == "org_acme_corp").first()
        if not org_a:
            org_a = Organization(
                id="org_acme_corp",
                name="Acme Corporation",
                api_key="sk_acme_demo_key_1001",
            )
            db.add(org_a)
        else:
            org_a.api_key = "sk_acme_demo_key_1001"
        db.commit()

        doc_a = db.query(Document).filter(Document.id == "doc_acme_handbook").first()
        if not doc_a or force:
            doc_a_text = (
                "# Acme Corporation - Internal Engineering & Employee Handbook\n\n"
                "## Project Phoenix\n"
                "Project Phoenix is Acme Corp's next-generation quantum computing accelerator. "
                "The targeted production release date is October 15, 2026. "
                "The engineering lead is Dr. Elena Rostova. "
                "The allocated Q4 budget is $12,500,000 USD.\n\n"
                "## Remote Work and Travel Expense Policy\n"
                "Acme employees may expense up to $75/day for meals during domestic business travel. "
                "Receipts must be submitted in Expensify within 14 calendar days of trip conclusion. "
                "All flights over 4 hours qualify for premium economy upgrade.\n\n"
                "## Health Benefits\n"
                "Acme Platinum Health covers 100% preventive care, dental cleanings twice a year, and "
                "$2,000 annual mental wellness stipend."
            )
            if not doc_a:
                doc_a = Document(
                    id="doc_acme_handbook",
                    org_id=org_a.id,
                    filename="Acme_Employee_Handbook_2026.md",
                    file_path="./uploaded_documents/org_acme_corp/Acme_Employee_Handbook_2026.md",
                    file_type="md",
                    file_size=len(doc_a_text.encode("utf-8")),
                    status="pending",
                )
                db.add(doc_a)
                db.commit()

            os.makedirs("./uploaded_documents/org_acme_corp", exist_ok=True)
            with open(doc_a.file_path, "w", encoding="utf-8") as f:
                f.write(doc_a_text)

            process_document_pipeline(
                document_id=doc_a.id,
                org_id=org_a.id,
                file_bytes=doc_a_text.encode("utf-8"),
                filename=doc_a.filename,
                content_type="text/markdown",
            )

        # 2. Cyberdyne Systems
        org_b = db.query(Organization).filter(Organization.id == "org_cyberdyne").first()
        if not org_b:
            org_b = Organization(
                id="org_cyberdyne",
                name="Cyberdyne Systems",
                api_key="sk_cyberdyne_demo_key_2002",
            )
            db.add(org_b)
        else:
            org_b.api_key = "sk_cyberdyne_demo_key_2002"
        db.commit()

        doc_b = db.query(Document).filter(Document.id == "doc_cyberdyne_defense").first()
        if not doc_b or force:
            doc_b_text = (
                "# Cyberdyne Systems - Defense Technologies & Operations\n\n"
                "## Project Phoenix\n"
                "Project Phoenix at Cyberdyne is a classified autonomous neural-mesh defense framework. "
                "The deployment launch is scheduled for January 30, 2029. "
                "The defense program director is Miles Dyson. "
                "The allocated defense contract value is $840,000,000 USD.\n\n"
                "## Security Clearances & Access Protocols\n"
                "Level 5 clearance is mandatory for Sector 7 cleanroom access. "
                "Biometric retargeting must be calibrated every 30 days. "
                "Any breach of protocol results in immediate badge revocation and military tribunal review.\n\n"
                "## Overtime & Hazardous Duty Pay\n"
                "Cyberdyne engineers in high-radiation sectors receive 2.5x base hourly compensation and "
                "30 mandatory rest days post-cycle."
            )
            if not doc_b:
                doc_b = Document(
                    id="doc_cyberdyne_defense",
                    org_id=org_b.id,
                    filename="Cyberdyne_Defense_Protocols.md",
                    file_path="./uploaded_documents/org_cyberdyne/Cyberdyne_Defense_Protocols.md",
                    file_type="md",
                    file_size=len(doc_b_text.encode("utf-8")),
                    status="pending",
                )
                db.add(doc_b)
                db.commit()

            os.makedirs("./uploaded_documents/org_cyberdyne", exist_ok=True)
            with open(doc_b.file_path, "w", encoding="utf-8") as f:
                f.write(doc_b_text)

            process_document_pipeline(
                document_id=doc_b.id,
                org_id=org_b.id,
                file_bytes=doc_b_text.encode("utf-8"),
                filename=doc_b.filename,
                content_type="text/markdown",
            )

        logger.info("Demo organizations and documents seeded successfully.")
    except Exception as e:
        logger.warning(f"Error seeding demo tenants: {e}")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    seed_default_tenants_and_docs()
    logger.info("Multi-Tenant RAG Knowledge Base initialized successfully.")
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Multi-Tenant RAG Knowledge Base with Multimodal Vision, Voice Input/Output, and Zero-Leakage Grounded AI",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(eval_router)

# Mount Static Files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "features": ["zero-leakage-vector-isolation", "multimodal-vision-rag", "voice-stt-tts", "sse-streaming"],
        "primary_llm": settings.PRIMARY_LLM_PROVIDER,
    }
