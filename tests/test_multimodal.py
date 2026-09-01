"""Tests for Multimodal Vision (Image/Screenshot) and Voice Audio Transcription."""

import pytest
import io



def test_multimodal_vision_query(client):
    """Asserts that multimodal queries with screenshot attachments return grounded answers."""
    # Create fake image bytes
    fake_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"

    response = client.post(
        "/api/chat/multimodal-query",
        headers={"X-API-Key": "sk_acme_demo_key_1001"},
        data={"query": "Explain what this diagram shows about Project Phoenix launch date", "top_k": 3},
        files={"image": ("architecture_diagram.png", fake_png, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["org_id"] == "org_acme_corp"
    assert len(data["citations"]) > 0


def test_multimodal_isolation(client):
    """Asserts that multimodal queries executed as Org A NEVER return Org B chunks."""
    fake_png = b"fake-image-bytes"

    response = client.post(
        "/api/chat/multimodal-query",
        headers={"X-API-Key": "sk_acme_demo_key_1001"},
        data={"query": "What is the Project Phoenix budget?", "top_k": 3},
        files={"image": ("screenshot.png", fake_png, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "$840,000,000" not in data["answer"], "Org B budget leaked during multimodal query!"
    assert data["org_id"] == "org_acme_corp"


def test_voice_transcription_endpoint(client):
    """Asserts that uploaded audio recordings return text transcription."""
    fake_audio = b"fake-audio-recording-bytes"

    response = client.post(
        "/api/chat/transcribe",
        headers={"X-API-Key": "sk_acme_demo_key_1001"},
        files={"audio_file": ("voice_memo.webm", fake_audio, "audio/webm")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "transcription" in data
    assert len(data["transcription"]) > 0
