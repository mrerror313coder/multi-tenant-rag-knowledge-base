"""Integration tests for FastAPI REST API endpoints."""

import pytest



def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_organization_onboarding_and_api_key(client):
    response = client.post(
        "/api/organizations/",
        json={"name": "Stark Industries", "org_id": "org_stark"},
    )
    assert response.status_code in [201, 409]
    if response.status_code == 201:
        data = response.json()
        assert data["id"] == "org_stark"
        assert data["api_key"].startswith("sk_org_stark_")


def test_document_upload_and_auth(client):
    # Attempt unauthenticated upload
    response = client.post("/api/documents/upload", files={"file": ("test.txt", b"Secret data")})
    assert response.status_code == 401

    # Upload with valid API key
    response = client.post(
        "/api/documents/upload",
        headers={"X-API-Key": "sk_acme_demo_key_1001"},
        files={"file": ("quantum_research.txt", b"Quantum qubits operate in superposition.")},
    )
    assert response.status_code == 201
    doc = response.json()
    assert doc["filename"] == "quantum_research.txt"
    assert doc["org_id"] == "org_acme_corp"


def test_rag_query_endpoint(client):
    response = client.post(
        "/api/chat/query",
        headers={"X-API-Key": "sk_acme_demo_key_1001"},
        json={"query": "When is Project Phoenix launch date?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["org_id"] == "org_acme_corp"
    assert "citations" in data
    assert data["latency_ms"] >= 0


def test_isolation_audit_endpoint(client):
    response = client.post("/api/eval/run-isolation-check")
    assert response.status_code == 200
    data = response.json()
    assert data["isolation_passed"] is True
    assert data["org_a_leak_count"] == 0
    assert data["org_b_leak_count"] == 0


def test_delete_organization_endpoint(client):
    # Create temp tenant
    client.post(
        "/api/organizations/",
        json={"name": "Temp Org For Deletion", "org_id": "org_temp_del"},
    )
    # Delete temp tenant
    response = client.delete("/api/organizations/org_temp_del")
    assert response.status_code == 200
    data = response.json()
    assert data["deleted_org_id"] == "org_temp_del"

    # Confirm 404 on subsequent delete
    response_404 = client.delete("/api/organizations/org_temp_del")
    assert response_404.status_code == 404

