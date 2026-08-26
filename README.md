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
- `GET /api/model/diagnostics` (或 `/api/diagnostics`) — 检查本地端点连接、模型列表、延迟和错误信息。
- `POST /api/experiences/{id}/ab` and `GET /api/experiences/{id}/ab` — generate/read A/B candidates.

Memory extraction is intentionally conservative and offline. Statements such as “I prefer concise answers”, “I study astronomy”, or “remember that …” are candidates; arbitrary chat is not automatically persisted as a fact. Repeated identical memories increase evidence and confidence.

### 可选：接入本地模型

Mock generator 默认保持启用，因此没有模型服务时仍可离线运行。要接入兼容 OpenAI Chat Completions API 的本地服务（Ollama、llama.cpp、vLLM 或 LM Studio），设置服务的基础 URL：

```powershell
# Ollama（需先 ollama serve；模型名示例）
$env:PALIMPSEST_MODEL_ENDPOINT = "http://127.0.0.1:11434/v1"
$env:PALIMPSEST_MODEL_NAME = "qwen2.5:3b"

# LM Studio / llama.cpp 常见地址
$env:PALIMPSEST_MODEL_ENDPOINT = "http://127.0.0.1:1234/v1"
$env:PALIMPSEST_MODEL_NAME = "local-model"
```

可选配置项：

- `PALIMPSEST_MODEL_NAMES`：逗号分隔的模型选择列表；未设置时只显示 `PALIMPSEST_MODEL_NAME`。
- `PALIMPSEST_API_KEY`：需要鉴权的本地代理使用；也接受 `OPENAI_API_KEY`。
- `PALIMPSEST_MODEL_TIMEOUT`：请求超时秒数，默认 120；诊断请求最多等待 5 秒。
- `OPENAI_BASE_URL`、`OPENAI_MODEL`：兼容已有 OpenAI 风格环境变量。

端点可以填写 `/v1` 基础路径，也可以直接填写 `/chat/completions`；客户端会自动补齐路径。启动后访问 `/api/model/diagnostics`，或查看页面顶部模型旁的状态，即可确认连接和模型列表。`/api/chat/stream` 会转发服务端的 SSE token；如果服务忽略 `stream=true` 而返回普通 JSON，则自动以单块 SSE 兼容返回。

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
