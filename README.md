# Crash Causal Event Pipeline

## Setup

1. Create and activate virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

1. Install dependencies with uv:

```bash
uv sync
```

1. Configure environment variables:

```bash
# Gemini API keys (supports rotation across models and keys)
export GOOGLE_API_KEY=...                  # optional; any valid key
export GOOGLE_API_KEY_1=...                # recommended to set multiple
export GOOGLE_API_KEY_2=...
export GOOGLE_API_KEY_3=...
export GOOGLE_API_KEY_4=...

# Neo4j (only needed if using --neo4j)
export NEO4J_URI=bolt://localhost:7687
export NEO4J_PASSWORD=your_password
```

Notes:

- If `GOOGLE_API_KEY` is not set but `GOOGLE_API_KEY_1..N` are, the app will automatically use the first numbered key.
- The client rotates models first, then rotates API keys when per-model limits are hit, to avoid workflow interruption.

## Features

- Structured event extraction from narratives (Gemini)
- Model and API-key rotation on rate limits
- Narrative preprocessing to normalize “Unit X”/“Driver of Unit X” to canonical IDs
- Neo4j causal graph build (with crash, person, vehicle, event, condition, outcome)
- Motif mining and stakeholder analysis (optional)
- Visualizations and dashboard generated from extractions

## CLI flags (main.py)

- `--csv PATH`:
  - Input CSV (default: `data/sample_texas_crash_data.csv`).
- `--limit N`:
  - Max rows to process during extraction (default: 10).
- `--out-dir DIR`:
  - Outputs directory (default: `outputs`).
- `--output PATH`:
  - Extractions JSONL output path (default: `OUT_DIR/extractions.jsonl`).
- `--use-existing-extractions PATH`:
  - Skip extraction and use an existing JSONL.
- `--neo4j`:
  - Build Neo4j graph from extractions (requires `NEO4J_URI`, `NEO4J_PASSWORD`).
- `--clear-neo4j`:
  - Clear database before building (use with `--neo4j`).
- `--analysis`:
  - Run stakeholder analysis after graph build (writes JSON and Cypher).
- `--viz`:
  - Generate visualizations and dashboard from extractions.

## Common command combinations

- Extraction only (limit 10):

```bash
source .venv/bin/activate && uv run python main.py --limit 10
```

- Extraction + visualizations:

```bash
source .venv/bin/activate && uv run python main.py --limit 50 --viz
```

- Use an existing extractions file, regenerate visualizations:

```bash
source .venv/bin/activate && uv run python main.py \
  --use-existing-extractions outputs/extractions.jsonl \
  --viz
```

- Extraction + Neo4j graph (clear DB) + motif mining + stakeholder analysis:

```bash
source .venv/bin/activate && uv run python main.py \
  --limit 100 \
  --neo4j --clear-neo4j --analysis
```

- Full pipeline with custom paths:

```bash
source .venv/bin/activate && uv run python main.py \
  --csv data/processed_crash_data.csv \
  --limit 500 \
  --out-dir outputs \
  --output outputs/extractions.jsonl \
  --neo4j --clear-neo4j --analysis --viz
```

## Visualizations only (module entry)

Regenerate charts and dashboard from an existing JSONL without re-extracting:

```bash
source .venv/bin/activate && uv run python -m utils.visualize \
  --extractions outputs/extractions.jsonl \
  --out-dir outputs/viz
```

## Tips

- If you hit Gemini quota on one model, the client automatically rotates models: `gemini-2.5-flash-lite → gemini-2.5-flash → gemini-2.0-flash → gemini-2.5-pro`, then rotates API keys when needed.
- Ensure `Crash_ID` exists in rows you plan to load into Neo4j (used for normalizing Unit references).
- To start fresh in Neo4j, always include `--clear-neo4j` with `--neo4j`.
