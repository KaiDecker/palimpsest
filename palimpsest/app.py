"""FastAPI application for Palimpsest MVP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import Database, utc_now
from .generation import Generator, GenerationError, MockGenerator, OpenAICompatibleGenerator
from .memory import retrieve, save_extracted


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=20_000)
    model: str | None = None


class FeedbackRequest(BaseModel):
    rating: int | None = Field(default=None, ge=-1, le=1)
    edited_response: str | None = None
    chosen_response: str | None = None
    rejected_response: str | None = None


class ABRequest(BaseModel):
    model: str | None = None


class MemoryRequest(BaseModel):
    type: str = "fact"
    content: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(default=0.8, ge=0, le=1)
    stability: float = Field(default=0.5, ge=0, le=1)


class ProfileRequest(BaseModel):
    key: str = Field(min_length=1, max_length=200)
    value: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(default=0.7, ge=0, le=1)


def _conversation(db: Database, conversation_id: str | None, title: str) -> dict[str, Any]:
    now = utc_now()
    with db.connection() as conn:
        if conversation_id:
            row = conn.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
            return dict(row)
        conversation_id = db.new_id()
        conn.execute("INSERT INTO conversations VALUES(?,?,?,?)", (conversation_id, title[:80], now, now))
        return {"id": conversation_id, "title": title[:80], "created_at": now, "updated_at": now}


def _messages(db: Database, conversation_id: str) -> list[dict[str, Any]]:
    with db.connection() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at", (conversation_id,)).fetchall()]


def _serialize_message(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["metadata"] = Database.parse_json(result.pop("metadata_json", "{}"))
    return result


def create_app(db: Database | None = None, generator: Generator | None = None) -> FastAPI:
    database = db or Database()
    configuration_error: str | None = None
    if generator is not None:
        model = generator
    else:
        try:
            model = OpenAICompatibleGenerator.from_env() or MockGenerator()
        except ValueError as exc:
            # A typo in an environment variable must not make the offline app
            # impossible to start. Keep the error available to diagnostics.
            configuration_error = str(exc)
            model = MockGenerator()
    app = FastAPI(title="Palimpsest", version="0.1.0", description="Local-first personal AI MVP")
    app.state.db = database
    app.state.generator = model
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model.model_name}

    @app.get("/api/models")
    def list_models() -> dict[str, Any]:
        models = [{"id": "palimpsest-mock-v1", "name": "Mock (offline)", "backend": "mock", "available": True}]
        if isinstance(model, OpenAICompatibleGenerator):
            models.extend({"id": name, "name": name, "backend": "openai-compatible", "available": True} for name in OpenAICompatibleGenerator.model_names_from_env(model.model_name))
        return {"default": model.model_name, "models": models}

    @app.get("/api/model/diagnostics")
    def model_diagnostics() -> dict[str, Any]:
        """Return local model connectivity information without failing the API."""
        if configuration_error:
            return {"backend": "openai-compatible", "status": "error", "reachable": False, "endpoint": None, "models_endpoint": None, "model": model.model_name, "models": [], "latency_ms": 0, "error": configuration_error}
        if isinstance(model, OpenAICompatibleGenerator):
            return {"backend": "openai-compatible", **model.diagnose()}
        return {"backend": "mock", "status": "ready", "reachable": True, "endpoint": None, "models_endpoint": None, "model": model.model_name, "models": [model.model_name], "latency_ms": 0, "error": None}

    @app.get("/api/diagnostics")
    def diagnostics_alias() -> dict[str, Any]:
        return model_diagnostics()

    def selected_model(name: str | None) -> Generator:
        if not name or name == model.model_name:
            return model
        if name == "palimpsest-mock-v1":
            return MockGenerator()
        if isinstance(model, OpenAICompatibleGenerator) and name in OpenAICompatibleGenerator.model_names_from_env(model.model_name):
            return OpenAICompatibleGenerator(model.base_url, name, model.api_key, model.timeout)
        raise HTTPException(status_code=400, detail=f"Unknown model: {name}")

    @app.get("/api/conversations")
    def list_conversations() -> list[dict[str, Any]]:
        with database.connection() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM conversations ORDER BY updated_at DESC").fetchall()]

    @app.get("/api/conversations/{conversation_id}")
    def get_conversation(conversation_id: str) -> dict[str, Any]:
        with database.connection() as conn:
            conversation = conn.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"conversation": dict(conversation), "messages": [_serialize_message(row) for row in _messages(database, conversation_id)]}

    def prepare_chat(payload: ChatRequest) -> dict[str, Any]:
        generator = selected_model(payload.model)
        conversation = _conversation(database, payload.conversation_id, payload.message)
        history = _messages(database, conversation["id"])
        memories = retrieve(database, payload.message)
        with database.connection() as conn:
            profile = [dict(row) for row in conn.execute("SELECT * FROM profile ORDER BY updated_at DESC").fetchall()]
        return {"generator": generator, "conversation": conversation, "history": history, "memories": memories, "profile": profile, "prompt": payload.message}

    def persist_chat(prepared: dict[str, Any], response: str) -> dict[str, Any]:
        generator = prepared["generator"]
        conversation = prepared["conversation"]
        history = prepared["history"]
        memories = prepared["memories"]
        prompt = prepared["prompt"]
        now = utc_now()
        user_message_id, assistant_message_id, experience_id = database.new_id(), database.new_id(), database.new_id()
        with database.connection() as conn:
            conn.execute("INSERT INTO messages VALUES(?,?,?,?,?,?)", (user_message_id, conversation["id"], "user", prompt, now, "{}"))
            conn.execute("INSERT INTO messages VALUES(?,?,?,?,?,?)", (assistant_message_id, conversation["id"], "assistant", response, now, database.json_value({"experience_id": experience_id})))
            conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation["id"]))
            conn.execute("INSERT INTO experiences VALUES(?,?,?,?,?,?,?,?,?)", (experience_id, conversation["id"], prompt, response, database.json_value(history), database.json_value(memories), "{}", database.json_value({"base_model": generator.model_name, "adapter": generator.adapter}), now))
        extracted = save_extracted(database, prompt)
        # Explicit preference statements are also represented in the user model.
        for item in extracted:
            if item["type"] == "preference":
                _upsert_profile("preference.communication", item["content"], 0.65, "explicit_heuristic")
        return {"conversation_id": conversation["id"], "experience_id": experience_id, "response": response, "memories": memories, "extracted_memories": extracted, "model": {"base_model": generator.model_name, "adapter": generator.adapter}}

    def run_chat(payload: ChatRequest) -> dict[str, Any]:
        prepared = prepare_chat(payload)
        generator = prepared["generator"]
        try:
            response = generator.generate(prepared["prompt"], [{"role": row["role"], "content": row["content"]} for row in prepared["history"]], prepared["memories"], prepared["profile"])
        except GenerationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return persist_chat(prepared, response)

    @app.post("/api/chat")
    def chat(payload: ChatRequest) -> dict[str, Any]:
        return run_chat(payload)

    @app.post("/api/chat/stream")
    def chat_stream(payload: ChatRequest) -> StreamingResponse:
        prepared = prepare_chat(payload)
        generator = prepared["generator"]

        def events():
            metadata = {"conversation_id": prepared["conversation"]["id"], "memories": prepared["memories"], "model": {"base_model": generator.model_name, "adapter": generator.adapter}}
            yield f"event: metadata\ndata: {json.dumps(metadata, ensure_ascii=False)}\n\n"
            chunks: list[str] = []
            try:
                stream_method = getattr(generator, "generate_stream", None)
                if callable(stream_method):
                    for chunk in stream_method(prepared["prompt"], [{"role": row["role"], "content": row["content"]} for row in prepared["history"]], prepared["memories"], prepared["profile"]):
                        if chunk:
                            chunks.append(chunk)
                            yield f"event: token\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                else:
                    chunk = generator.generate(prepared["prompt"], [{"role": row["role"], "content": row["content"]} for row in prepared["history"]], prepared["memories"], prepared["profile"])
                    chunks.append(chunk)
                    yield f"event: token\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                response = "".join(chunks)
                if not response.strip():
                    raise GenerationError("Local model endpoint returned an empty response")
                result = persist_chat(prepared, response)
                yield f"event: done\ndata: {json.dumps({k: result[k] for k in ('conversation_id', 'experience_id', 'extracted_memories', 'model')}, ensure_ascii=False)}\n\n"
            except GenerationError as exc:
                yield f"event: error\ndata: {json.dumps({'detail': str(exc)}, ensure_ascii=False)}\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/api/experiences/{experience_id}")
    def get_experience(experience_id: str) -> dict[str, Any]:
        with database.connection() as conn:
            row = conn.execute("SELECT * FROM experiences WHERE id=?", (experience_id,)).fetchone()
            feedback = conn.execute("SELECT * FROM feedback WHERE experience_id=? ORDER BY created_at DESC", (experience_id,)).fetchall()
        if row is None:
            raise HTTPException(status_code=404, detail="Experience not found")
        item = dict(row)
        for key in ("context_json", "retrieved_memories_json", "feedback_json", "model_json"):
            item[key.removesuffix("_json")] = database.parse_json(item.pop(key))
        item["feedback_entries"] = [dict(entry) for entry in feedback]
        return item

    @app.post("/api/experiences/{experience_id}/feedback")
    def feedback(experience_id: str, payload: FeedbackRequest) -> dict[str, Any]:
        with database.connection() as conn:
            if conn.execute("SELECT 1 FROM experiences WHERE id=?", (experience_id,)).fetchone() is None:
                raise HTTPException(status_code=404, detail="Experience not found")
            feedback_id, now = database.new_id(), utc_now()
            conn.execute("INSERT INTO feedback VALUES(?,?,?,?,?,?,?)", (feedback_id, experience_id, payload.rating, payload.edited_response, payload.chosen_response, payload.rejected_response, now))
            conn.execute("UPDATE experiences SET feedback_json=? WHERE id=?", (database.json_value(payload.model_dump(exclude_none=True)), experience_id))
        return {"id": feedback_id, "experience_id": experience_id, **payload.model_dump()}

    @app.post("/api/experiences/{experience_id}/ab")
    def generate_ab(experience_id: str, payload: ABRequest = ABRequest()) -> dict[str, Any]:
        with database.connection() as conn:
            experience = conn.execute("SELECT * FROM experiences WHERE id=?", (experience_id,)).fetchone()
        if experience is None:
            raise HTTPException(status_code=404, detail="Experience not found")
        generator = selected_model(payload.model)
        history = database.parse_json(experience["context_json"])
        memories = database.parse_json(experience["retrieved_memories_json"])
        with database.connection() as conn:
            profile = [dict(row) for row in conn.execute("SELECT * FROM profile ORDER BY updated_at DESC").fetchall()]
        try:
            candidate_b = generator.generate_variant(experience["prompt"], history, memories, profile)
        except AttributeError:
            candidate_b = generator.generate(experience["prompt"] + "\nProvide an alternative answer.", history, memories, profile)
        except GenerationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        now = utc_now()
        with database.connection() as conn:
            conn.execute("DELETE FROM variants WHERE experience_id=?", (experience_id,))
            conn.execute("INSERT OR REPLACE INTO variants(id,experience_id,label,content,model_json,created_at) VALUES(?,?,?,?,?,?)", (database.new_id(), experience_id, "A", experience["response"], database.json_value({"base_model": generator.model_name, "adapter": generator.adapter}), now))
            conn.execute("INSERT OR REPLACE INTO variants(id,experience_id,label,content,model_json,created_at) VALUES(?,?,?,?,?,?)", (database.new_id(), experience_id, "B", candidate_b, database.json_value({"base_model": generator.model_name, "adapter": generator.adapter}), now))
        return {"experience_id": experience_id, "variants": [{"label": "A", "content": experience["response"]}, {"label": "B", "content": candidate_b}], "model": {"base_model": generator.model_name, "adapter": generator.adapter}}

    @app.get("/api/experiences/{experience_id}/ab")
    def get_ab(experience_id: str) -> dict[str, Any]:
        with database.connection() as conn:
            rows = [dict(row) for row in conn.execute("SELECT * FROM variants WHERE experience_id=? ORDER BY label", (experience_id,)).fetchall()]
        if not rows:
            raise HTTPException(status_code=404, detail="A/B candidates not found")
        for row in rows:
            row["model"] = database.parse_json(row.pop("model_json"))
        return {"experience_id": experience_id, "variants": rows}

    @app.get("/api/memories")
    def memories(q: str | None = Query(default=None), limit: int = Query(default=50, ge=1, le=200)) -> list[dict[str, Any]]:
        if q:
            return retrieve(database, q, limit)
        with database.connection() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM memories ORDER BY last_updated DESC LIMIT ?", (limit,)).fetchall()]

    @app.post("/api/memories")
    def add_memory(payload: MemoryRequest) -> dict[str, Any]:
        now, memory_id = utc_now(), database.new_id()
        with database.connection() as conn:
            conn.execute("INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?)", (memory_id, payload.type, payload.content, payload.confidence, payload.stability, "user", 1, now, now, None))
        return {"id": memory_id, **payload.model_dump(), "source": "user", "evidence_count": 1, "created_at": now, "last_updated": now, "valid_until": None}

    def _upsert_profile(key: str, value: str, confidence: float, source: str) -> None:
        now = utc_now()
        with database.connection() as conn:
            conn.execute("INSERT INTO profile(key,value,confidence,source,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, confidence=excluded.confidence, source=excluded.source, updated_at=excluded.updated_at", (key, value, confidence, source, now))

    @app.get("/api/profile")
    def get_profile() -> list[dict[str, Any]]:
        with database.connection() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM profile ORDER BY updated_at DESC").fetchall()]

    @app.put("/api/profile")
    def set_profile(payload: ProfileRequest) -> dict[str, Any]:
        _upsert_profile(payload.key, payload.value, payload.confidence, "user")
        with database.connection() as conn:
            return dict(conn.execute("SELECT * FROM profile WHERE key=?", (payload.key,)).fetchone())

    @app.get("/api/dataset/export")
    def export_dataset() -> StreamingResponse:
        with database.connection() as conn:
            rows = conn.execute("SELECT * FROM experiences ORDER BY created_at").fetchall()
            feedback_rows = conn.execute("SELECT * FROM feedback ORDER BY created_at").fetchall()
        feedback_by_experience: dict[str, list[dict[str, Any]]] = {}
        for row in feedback_rows:
            feedback_by_experience.setdefault(row["experience_id"], []).append(dict(row))
        lines = []
        for row in rows:
            item = dict(row)
            item["context"] = database.parse_json(item.pop("context_json"))
            item["retrieved_memories"] = database.parse_json(item.pop("retrieved_memories_json"))
            item["feedback"] = feedback_by_experience.get(item["id"], [])
            item["model"] = database.parse_json(item.pop("model_json"))
            item.pop("feedback_json", None)
            lines.append(json.dumps(item, ensure_ascii=False))
        return StreamingResponse(iter(["\n".join(lines) + ("\n" if lines else "")]), media_type="application/jsonl", headers={"Content-Disposition": "attachment; filename=palimpsest-dataset.jsonl"})

    return app


app = create_app()
