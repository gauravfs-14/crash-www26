# Crash Causal Graph Pipeline

Single-entry workflow to extract events with Gemini, build a comprehensive Neo4j causal graph, and run stakeholder analysis.

## Requirements

- Neo4j running locally (or remote) and accessible
- Python managed via `uv`
- Google Gemini API Key

## Setup

```bash
uv sync
```

Add the following to the `.env` file.

```bash
GOOGLE_API_KEY=
NEO4J_URI=
NEO4J_PASSWORD=
```

## Single-entry CLI (main.py)

```bash
uv run main.py --csv data/sample_texas_crash_data.csv --limit 100 --neo4j --clear-neo4j --analysis --viz --out-dir outputs
```

Run defaults:

```bash
uv run main.py --limit 10 --neo4j --clear-neo4j --analysis --viz
```

### Flags

- `--csv`: Input CSV (default: `data/sample_texas_crash_data.csv`)
- `--limit`: Number of rows to process (default: 10)
- `--out-dir`: Directory for outputs (default: `outputs/`)
- `--output`: Where to write extractions (default: `OUT_DIR/extractions.jsonl`)
- `--use-existing-extractions`: Skip extraction; use an existing JSONL
- `--neo4j`: Build Neo4j graph from extractions
- `--clear-neo4j`: Clear database first (use with `--neo4j`)
- `--analysis`: Run stakeholder analysis after graph build
- `--viz`: Generate visualizations and dashboard (saved under `OUT_DIR/viz/`)

### Examples

- Extract only:

```bash
uv run main.py --limit 20 --viz --out-dir outputs
```

- Build graph from existing extractions and run analysis:

```bash
uv run main.py --use-existing-extractions outputs/extractions.jsonl --neo4j --clear-neo4j --analysis --viz --out-dir outputs
```

## Outputs

- `outputs/extractions.jsonl`: Structured LLM extractions per row
- Neo4j Graph: Nodes (`Person`, `Vehicle`, `CausalEvent`, `Condition`, `Outcome`, `Crash`, `Unit`) and relationships (`CAUSES`, `RESULTS_IN`, `OCCURRED_UNDER`, `AFFECTS`, `INVOLVES`, `CONTAINS`)
- `outputs/stakeholder_report.json`: Multi-stakeholder insights
- `outputs/stakeholder_queries.cypher`: Ready-to-run analysis queries
- `outputs/viz/`: Dashboard (`dashboard.html`) and PNGs (`severity_distribution.png`, `alcohol_mentions.png`, `top_actions.png`, `conditions_heatmap.png`)

## Notes

- Logs: `logs/main.log`, `logs/neo4j.log`, `logs/stakeholder_analysis.log`
- If `GOOGLE_API_KEY` is not set, `main.py` will securely prompt for it using getpass

## Graph Schema (Overview)

- Nodes: `Person`, `Vehicle`, `CausalEvent`, `Condition`, `Outcome`, `Crash`, `Unit`
- Key relationships:
  - `(:Person)-[:CAUSES]->(:CausalEvent)`
  - `(:CausalEvent)-[:RESULTS_IN]->(:Outcome)`
  - `(:CausalEvent)-[:OCCURRED_UNDER]->(:Condition)`
  - `(:CausalEvent)-[:AFFECTS]->(:Person|:Vehicle)`
  - `(:Crash)-[:CONTAINS]->(:CausalEvent)`
  - `(:CausalEvent)-[:INVOLVES]->(:Unit)`

## Example Cypher Queries

```cypher
// Alcohol-related persons
MATCH (p:Person)-[:CAUSES]->(e:CausalEvent)
WHERE e.mentions_alcohol = true
RETURN count(DISTINCT p) AS alcohol_related_persons;

// Severity distribution
MATCH (e:CausalEvent)
RETURN e.severity_impact AS severity, count(e) AS count;

// Condition type distribution
MATCH (e:CausalEvent)-[:OCCURRED_UNDER]->(c:Condition)
RETURN c.type AS condition_type, count(c) AS count;

// Causal chain
MATCH path = (p:Person)-[:CAUSES]->(e:CausalEvent)-[:AFFECTS]->(q:Person)
RETURN path LIMIT 25;
```
