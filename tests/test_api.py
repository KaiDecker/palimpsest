from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from palimpsest.app import create_app
from palimpsest.db import Database


@pytest.fixture
def client():
    path = Path("data") / f"test-{uuid4().hex}.db"
    api = TestClient(create_app(Database(path)))
    yield api
    for suffix in ("", "-wal", "-shm"):
        path.with_name(path.name + suffix).unlink(missing_ok=True)


def test_chat_persists_conversation_memory_and_experience(client):
    api = client
    response = api.post("/api/chat", json={"message": "I prefer concise answers"})
    assert response.status_code == 200
    result = response.json()
    assert result["conversation_id"]
    assert result["experience_id"]
    assert result["extracted_memories"][0]["type"] == "preference"

    conversation = api.get(f"/api/conversations/{result['conversation_id']}").json()
    assert [message["role"] for message in conversation["messages"]] == ["user", "assistant"]
    assert api.get("/api/memories").json()[0]["content"] == "concise answers"

    second = api.post("/api/chat", json={"conversation_id": result["conversation_id"], "message": "How should you answer me?"}).json()
    assert "Relevant memory" in second["response"]
    memories = api.get("/api/memories").json()
    assert memories[0]["evidence_count"] == 1
    api.post("/api/chat", json={"conversation_id": result["conversation_id"], "message": "I prefer concise answers"})
    assert api.get("/api/memories").json()[0]["evidence_count"] == 2
    assert api.get(f"/api/experiences/{result['experience_id']}").status_code == 200


def test_feedback_and_dataset_export(client):
    api = client
    chat = api.post("/api/chat", json={"message": "Remember that I study astronomy"}).json()
    feedback = api.post(f"/api/experiences/{chat['experience_id']}/feedback", json={"rating": 1, "edited_response": "A better answer", "chosen_response": "A", "rejected_response": "B"})
    assert feedback.status_code == 200
    assert feedback.json()["rating"] == 1
    export = api.get("/api/dataset/export")
    assert export.status_code == 200
    assert '"edited_response": "A better answer"' in export.text
    assert "palimpsest-dataset.jsonl" in export.headers["content-disposition"]


def test_manual_profile_and_streaming_endpoint(client):
    api = client
    profile = api.put("/api/profile", json={"key": "communication", "value": "direct", "confidence": 0.9})
    assert profile.status_code == 200
    streamed = api.post("/api/chat/stream", json={"message": "Hello"})
    assert streamed.status_code == 200
    assert "event: token" in streamed.text
    assert api.get("/api/profile").json()[0]["value"] == "direct"


def test_models_and_ab_candidates_are_persisted(client):
    api = client
    assert api.get("/api/models").json()["models"][0]["id"] == "palimpsest-mock-v1"
    chat = api.post("/api/chat", json={"message": "Give me a plan"}).json()
    ab = api.post(f"/api/experiences/{chat['experience_id']}/ab", json={})
    assert ab.status_code == 200
    variants = ab.json()["variants"]
    assert [variant["label"] for variant in variants] == ["A", "B"]
    feedback = api.post(f"/api/experiences/{chat['experience_id']}/feedback", json={"chosen_response": variants[1]["content"], "rejected_response": variants[0]["content"]})
    assert feedback.status_code == 200
    stored = api.get(f"/api/experiences/{chat['experience_id']}").json()
    assert stored["feedback_entries"][0]["chosen_response"] == variants[1]["content"]
    assert api.get(f"/api/experiences/{chat['experience_id']}/ab").json()["variants"][1]["content"] == variants[1]["content"]
