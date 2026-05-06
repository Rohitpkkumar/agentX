# Local Coding Agent

A local AI coding agent that runs on your machine and uses a remote LLM (Ollama or Groq).
Works like Claude Code — persistent sessions, full file/git/shell access, ReAct loop.

## Requirements

- Python 3.11+
- An LLM provider:
  - **Ollama** running on a server: `OLLAMA_URL=http://server-ip:11434`
  - **Groq** cloud API: `LLM_PROVIDER=groq GROQ_API_KEY=<your-key>`
- `rg` (ripgrep) on PATH for code search — `brew install ripgrep`

## Install

```bash
# Core install — chat, run, git, shell tools
pip install "git+https://github.com/Rohitpkkumar/agentX.git"

# With code indexing (vector search, symbol index)
pip install "git+https://github.com/Rohitpkkumar/agentX.git#egg=local-coding-agent[index]"

# With Groq support
pip install "git+https://github.com/Rohitpkkumar/agentX.git#egg=local-coding-agent[groq]"

# Everything
pip install "git+https://github.com/Rohitpkkumar/agentX.git#egg=local-coding-agent[all]"
```

## Quickstart

```bash
# Point at your LLM (pick one)
export OLLAMA_URL="http://server-ip:11434"       # Ollama on a server
export LLM_PROVIDER=groq GROQ_API_KEY=sk-...    # or Groq cloud

# Optional: choose a model
export CHAT_MODEL="qwen3-coder:30b"             # default for Ollama
export GROQ_MODEL="llama-3.3-70b-versatile"     # default for Groq

# Go to your project and type agentX — that's it
cd /path/to/your/project
agentX
```

## Usage

```bash
agentX                        # start interactive session (like typing `claude`)
agentX "add a login endpoint" # one-shot task then exit
agentX sessions               # list saved sessions
agentX resume <id>            # resume a previous session
```

## Advanced commands (`agent` CLI)

| Command | Description |
|---|---|
| `agent init` | Set up `.agent/` and run first index |
| `agent chat` | Interactive session (same as `agentX`) |
| `agent run "<task>"` | One-shot task |
| `agent sessions` | List saved sessions |
| `agent index` | Force full reindex (requires `[index]`) |
| `agent rollback` | Undo last task via git checkpoint |
| `agent log` | Show recent task traces |
| `agent config` | Get/set trust mode and settings |

## LLM Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `groq` |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `CHAT_MODEL` | `qwen3-coder:30b` | Model name for Ollama |
| `GROQ_API_KEY` | — | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model name for Groq |

## Trust Modes

| Mode | Description |
|---|---|
| `readonly` | No file writes, no destructive shell commands |
| `trusted` | File writes allowed; no network from tools (default) |
| `yolo` | All operations allowed |

```bash
agent config trust_mode yolo    # set trust mode
agent chat --trust readonly     # or pass per-session
```

## What it can do

- Read, write, edit files in your project
- Run shell commands (tests, builds, installs)
- Search code with ripgrep
- Git operations (status, diff, add, commit, log, checkpoint, rollback)
- Run tests and interpret failures
- Persistent sessions — resume any conversation by ID
- Code indexing + vector search (with `[index]` extras)
