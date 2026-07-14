# TwinSpark

A minimal, clean agent built on **Alibaba Bailian (DashScope)** via its
OpenAI-compatible API. TwinSpark is a slimmed-down rewrite focused on a small,
readable core: configuration, an LLM client, lightweight memory, skills and
tools.

> Status: early scaffolding. The `core`, `memory`, `skills` and `tools`
> subpackages are placeholders — additional modules (`llm.py`, `store.py`,
> `cli.py`, `api.py`, ...) are actively being implemented.

## Installation

```bash
cd twinspark
pip install -e .
```

For development (tests):

```bash
pip install -e ".[dev]"
```

## Configuration

TwinSpark reads settings from environment variables and an optional `.env`
file. Copy the example and set your key:

```bash
cp .env.example .env
# then edit .env and set DASHSCOPE_API_KEY
```

Or export it directly:

```bash
export DASHSCOPE_API_KEY="your-key-here"
```

| Variable | Default | Description |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | *(required)* | DashScope / Bailian API key |
| `TWINSPARK_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI-compatible endpoint |
| `TWINSPARK_MODEL` | `qwen-plus` | Default model name |
| `TWINSPARK_DB_PATH` | `~/.twinspark/state.db` | Local SQLite state database |
| `TWINSPARK_SKILLS_DIR` | `~/.twinspark/skills/` | User skills directory |

## Entry points

TwinSpark offers two ways to run (both implemented in later tasks):

- **CLI** — an interactive terminal session:

  ```bash
  twinspark
  ```

- **API** — an HTTP server:

  ```bash
  uvicorn twinspark.api:app
  ```

## License

MIT
