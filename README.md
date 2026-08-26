# Palimpsest

Palimpsest is a local-first personal AI MVP: a small assistant that stores conversations, extracts explicit memories, maintains a user profile, and turns feedback into an exportable longitudinal dataset.

This version intentionally stops before training, RL, autonomous agents, or silent model updates. It has no external model dependency. The default deterministic generator is a replaceable interface, so a local model or API-backed implementation can be injected later.

## Run locally

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
source .venv/bin/activate
pip install -e ".[test]"
uvicorn palimpsest.app:app --reload
```

Open http://127.0.0.1:8000. SQLite data is stored in `data/palimpsest.db` (created on first run). The app can also be started with `python -m palimpsest`.

## API surface

- `POST /api/chat` — create a conversation or continue one.
- `POST /api/chat/stream` — SSE-compatible response for streamed clients.
- `GET /api/conversations` and `GET /api/conversations/{id}` — history.
- `GET/POST /api/memories` — list, lexical search (`?q=`), or add memories.
- `GET/PUT /api/profile` — user model fields.
- `POST /api/experiences/{id}/feedback` — rating, edit, and A/B chosen/rejected fields.
- `GET /api/dataset/export` — JSONL download preserving context, memories, model metadata, and feedback.
- `GET /api/models` — available mock and configured local models.
- `POST /api/experiences/{id}/ab` and `GET /api/experiences/{id}/ab` — generate/read A/B candidates.

Memory extraction is intentionally conservative and offline. Statements such as “I prefer concise answers”, “I study astronomy”, or “remember that …” are candidates; arbitrary chat is not automatically persisted as a fact. Repeated identical memories increase evidence and confidence.

### Optional local model endpoint

The mock generator remains the default. To use an OpenAI-compatible local server (llama.cpp, vLLM, or LM Studio), set `PALIMPSEST_MODEL_ENDPOINT` to its base URL (for example `http://127.0.0.1:1234/v1`) and optionally set `PALIMPSEST_MODEL_NAME`, `PALIMPSEST_MODEL_NAMES` (comma-separated choices), and `PALIMPSEST_API_KEY`. `OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_API_KEY` are also recognized. No provider SDK is required.

## Test

```bash
pytest
```

## 提交约定

提交信息使用中文，并采用简短的类型前缀：

```text
feat：增加记忆冲突检测
fix：修复反馈记录问题
docs：补充本地模型配置说明
test：增加 A/B 偏好接口测试
chore：更新开发工具配置
```

标题描述“发生了什么”，提交正文（如有）说明“为什么这样改”。维护者创建 PR 时使用 [.dev/pr-template.md](.dev/pr-template.md)。

## Project direction

The MVP implements the loop: interaction → structured experience → memory/profile → feedback → preference dataset. Model training, adapters, and RL are deliberately out of scope until real interaction data and personal evaluation exist.
