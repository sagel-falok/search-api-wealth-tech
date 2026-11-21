# Nevis WealthTech Search API

A FastAPI + PostgreSQL 18 backend providing unified, typo-tolerant search across **clients** and **client documents**, exposed as both REST endpoints and a Model Context Protocol (**MCP**) server.

## Live Deployment (Google Cloud Run)

- **Service URL:** `https://search-api-m4l2onkmna-nw.a.run.app`
- **Swagger Docs:** [https://search-api-m4l2onkmna-nw.a.run.app/docs](https://search-api-m4l2onkmna-nw.a.run.app/docs)
- **MCP Endpoint:** `https://search-api-m4l2onkmna-nw.a.run.app/mcp`

---

## Tech Stack

- **Runtime & Web Framework:** Python 3.11+ (`uv`), FastAPI, Pydantic v2 (OpenAPI 3.1.0)
- **MCP Server:** `fastapi-mcp` (Streamable HTTP at `/mcp`)
- **Database:** PostgreSQL 18 (`pgvector`, `pg_trgm`, `unaccent`, Full-Text Search) + SQLAlchemy 2.0 Async
- **Embeddings:** `FastEmbed` (`BAAI/bge-base-en-v1.5`, 768 dimensions)
- **LLM Document Enrichment:** OpenAI Structured Outputs (`gpt-6-luna` by default) for automatic summary generation and wealth-management category classification

---

## Quick Start

### Prerequisites

- **Docker & Docker Compose**
- **Python 3.11+** and **[uv](https://docs.astral.sh/uv/)** *(optional, for local environment)*

### 1. Configure Environment (Optional)

To enable LLM-powered document summarization and categorization on `POST /clients/{id}/documents`, copy `.env.example` to `.env` and set `OPENAI_API_KEY` (if omitted, document creation still works and falls back to `summary = null`):

```bash
cp .env.example .env
```

### 2. Start & Seed

```bash
# Build and start PostgreSQL 18 + FastAPI containers
make start

# Seed synthetic categories, clients, and documents
make seed
```

The API will be available at **`http://localhost:8000`** ([Swagger UI](http://localhost:8000/docs) | [ReDoc](http://localhost:8000/redoc) | [MCP](http://localhost:8000/mcp)).

### 3. Commands Reference

| Command | Description |
| :--- | :--- |
| `make start` | Build and start `db` + `app` containers (`docker compose up -d --build`) |
| `make seed` | Seed synthetic categories, clients, and documents |
| `make stop` | Stop running containers (`docker compose down`) |
| `make db-reset` | Wipe the database volume and restart containers from scratch |
| `make install` | Install Python dependencies locally (`uv sync`) |
| `make deploy` | Seed Neon PostgreSQL and deploy to Google Cloud Run (`./deploy.sh`) |
| `make destroy` | Tear down Google Cloud Run service and container images (`./destroy.sh`) |

### 4. Run Tests

```bash
docker compose exec app uv run pytest -v
```

---

## API Endpoints & MCP Integration

| Method | Endpoint | MCP Tool | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/clients` | `create_client` | Create a new client (`201 Created`, `409 Conflict` on duplicate email) |
| `POST` | `/clients/{id}/documents` | `create_client_document` | Attach a document to a client, enrich via LLM, and index for search (`201 Created`, `404 Not Found`) |
| `GET` | `/search?q={query}` | `search` | Unified search across clients and documents (`200 OK`) |

To connect Cursor or Claude Desktop (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "nevis-search-api": {
      "type": "http",
      "url": "https://search-api-m4l2onkmna-nw.a.run.app/mcp"
    }
  }
}
```
*(Use `"url": "http://localhost:8000/mcp"` when running locally.)*

---

## How It Works (Executive Summary)

- **Smart Document Ingestion (`POST /clients/{id}/documents`):** New documents are automatically summarized and classified into one of 16 wealth-management categories (`data/document_categories.json`) using OpenAI Structured Outputs, then split into contextual chunks and embedded with `BAAI/bge-base-en-v1.5`.
- **Identity-First Client Search:** Combines full-text search, accent normalization (`unaccent`), and trigram fuzzy matching (`pg_trgm`) over names, emails, and bios. Direct identity matches (`first_name`, `last_name`, `email`) always rank above secondary mentions in a client's `description`.
- **Hybrid Document Search:** Combines **semantic vector search** (`pgvector`), **lexical full-text search**, **fuzzy trigram matching**, and **cross-entity (`Client + Document`) boosting**. This allows advisors to find documents by concept (even when exact words aren't in the text), exact legal/tax codes, client name + document type, or misspelled queries.

---

## Supported Search Scenarios & Examples

### Client Search Cases

| Query (`GET /search?q=...`) | Scenario Covered | Top Result(s) |
| :--- | :--- | :--- |
| `?q=NevisWealth` | Exact company / email domain match | `@neviswealth.com` clients ranked above bio mentions |
| `?q=Nevis wealth` | Space-split compound domain/company name | `john.doe@neviswealth.com`, `jane.doe@neviswealth.com`, ... |
| `?q=nefis wealth` | Typo tolerance + space-split query | `john.doe@neviswealth.com`, `jane.doe@neviswealth.com`, ... |
| `?q=john` | Direct identity priority over bio mentions | **John Doe** ranked `#1` above **Jane Doe** (*"spouse of John Doe"*) |
| `?q=estate planning` | Natural-language keyword match in bio | `john.doe@neviswealth.com` |
| `?q=chloe` | Diacritic / accent normalization | `Chloé Dubois` (`chloe.dubois@vanguardia-bio.com`) |

### Document Search Cases

| Query (`GET /search?q=...`) | Scenario Covered | Top Result(s) |
| :--- | :--- | :--- |
| `?q=address proof` | Conceptual / category search (words not in document text) | Utility bills & council tax statements |
| `?q=electricity invoice` | Semantic synonym (*"invoice"* matches *"Utility Bill"*) | `Thames Water & EDF Electricity Utility Bill`, `Eversource Electric...` |
| `?q=estate succession and living will` | Conceptual match to trust & estate documents | `The Doe Family Revocable Living Trust Deed` |
| `?q=John Doe utility bill` | Cross-entity query (`Client + Document`) | **John Doe's** utility bill ranked `#1` above other clients' bills |
| `?q=adres proof` / `?q=pasport verification` | Severe typo tolerance on categories and titles | Address proof utility bills / US Passport verification |

### Example Request

```bash
curl -s "http://localhost:8000/search?q=John%20Doe%20utility%20bill"
```

---

## Disclaimer

This project is built as a **demonstration and prototype** to showcase hybrid search and LLM-driven document enrichment in a WealthTech context while keeping the architecture intentionally simple and self-contained.

To make the project easy to run, inspect, and evaluate out of the box, everything—relational storage, full-text search, trigram fuzzy matching, and vector similarity search—runs inside a single PostgreSQL instance, and document enrichment/embedding happens synchronously during API requests. In a real-world production environment handling sensitive financial records at scale, this setup would be extended with asynchronous background ingestion queues, dedicated search/reranking infrastructure, strict multi-tenant access controls (authentication, RBAC, and Row-Level Security), schema migrations, and PII encryption/audit logging.
