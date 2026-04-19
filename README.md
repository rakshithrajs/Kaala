# Kaal – Time-Aware Proactive Assistant (CLI)

Kaal is a proactive, multi-persona assistant prototype.

## Current Architecture

- **Niyati**: routing/orchestration persona
- **Iccha**: goal extraction persona
- **Karya**: planning persona (creates future prompt schedule)
- **Karma**: execution persona (turns due prompts into actions)
- **SQLite task store**: persists planned prompts and execution logs
- **Background scheduler loop**: checks and executes due tasks automatically

## Quick Start

1. Install dependencies:

```bash
uv sync
```

2. Create `.env` with your Google GenAI API key (`GOOGLE_API_KEY`).

3. Run:

```bash
uv run python main.py --model GEMINI-1.5-PRO --poll-interval 15
```

4. Type goals naturally in the chat. Kaal will schedule follow-ups and execute them when due.

## Notes

- Model aliases are loaded from `config/models.json`.
- Scheduled tasks and execution history are stored in `data/kaal.db`.
- The goal classifier module is included and now uses cross-platform file paths.
