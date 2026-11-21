# Nevis WealthTech Search API

A FastAPI + PostgreSQL 18 backend providing intelligent search across **clients** (`first_name`, `last_name`, `email`, `description`) and **client documents**.

---

## Tech Stack

- **Language & Runtime:** Python 3.11+ (`uv` package manager)
- **Web Framework:** FastAPI + Pydantic v2 (OpenAPI 3.1.0 compliant)
- **MCP Server:** `fastapi-mcp` exposing all API endpoints as Model Context Protocol (MCP) tools over Streamable HTTP (`/mcp`)
- **Database:** PostgreSQL 18 with `pgvector` (`pgvector/pgvector:pg18`, `asyncpg` + SQLAlchemy 2.0 Async ORM)
- **Document Enrichment (LLM):** OpenAI `AsyncOpenAI` Structured Outputs (`gpt-6-luna` by default, configurable via `OPENAI_MODEL`) classifying documents against our 16 defined wealth-management categories (`data/document_categories.json`) and generating concise summaries.
- **Search Engine:** Hybrid Lexical + Semantic + Fuzzy Search combining:
  - **Semantic Vector Search:** `pgvector` (`HNSW` cosine index) + `FastEmbed` (`BAAI/bge-base-en-v1.5`, 768 dimensions) over contextualized document chunks (`document_chunks`).
  - **Lexical Full-Text Search (BM25-like):** PostgreSQL `tsvector` / `tsquery` (`ts_rank_cd` with length and saturation normalization `33`) backed by `GENERATED ALWAYS AS ... STORED` columns and `GIN` indexes.
  - **Typo-Insensitive Fuzzy Search:** Trigram Similarity (`pg_trgm` `word_similarity` / `<%`) and Accent Normalization (`unaccent`).

---

## Setup Instructions

### Prerequisites

- **Docker & Docker Compose** (for PostgreSQL 18)
- **Python 3.11+** and **[uv](https://docs.astral.sh/uv/)**

### 1. Install Dependencies

```bash
make install
# or: uv sync
```

### 2. Configure Environment (Optional)

By default, the application connects to `postgresql+asyncpg://postgres:postgres@localhost:5433/search_api` and uses `gpt-6-luna` (`OPENAI_MODEL`) for document enrichment when `OPENAI_API_KEY` is provided. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

```dotenv
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/search_api
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-6-luna
```

> If `OPENAI_API_KEY` is left empty, `POST /clients/{id}/documents` gracefully falls back to leaving `summary` and `category` empty (`null`) while still generating contextual chunks and 768-dimensional embeddings for hybrid search.

### 3. Start PostgreSQL & Seed Synthetic Data

```bash
# Start PostgreSQL container on port 5433
make db-up

# Seed synthetic document categories, clients, and documents
make seed
```

> To reset the database volume from scratch at any time:
> ```bash
> make db-reset
> make seed
> ```

### 4. Run the API Server

```bash
make run
# or start both DB and API together:
make start
```

The API will be available at **`http://localhost:8000`**:
- **Interactive Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI 3.1 Schema:** [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)
- **MCP Server (Streamable HTTP):** [http://localhost:8000/mcp](http://localhost:8000/mcp)

### 5. Run Tests

```bash
uv run pytest -v
```

---

## API Endpoints & MCP Tools

All REST endpoints are also automatically exposed as **Model Context Protocol (MCP)** tools at **`http://localhost:8000/mcp`** via `fastapi-mcp`:

| Method | Endpoint | MCP Tool Name (`operation_id`) | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/clients` | `create_client` | Create a new client (`201 Created`, or `409 Conflict` on duplicate email) |
| `POST` | `/clients/{id}/documents` | `create_client_document` | Attach a new document to an existing client; enriches `summary` & `category` via OpenAI Structured Outputs (`gpt-6-luna`), builds contextual chunks, and indexes 768-d vectors (`201 Created`, or `404 Not Found`) |
| `GET` | `/search?q={query}` | `search` | Search across clients and documents (`200 OK`) |

### Connecting an MCP Client (Cursor / Claude Desktop)

Add the following to your MCP configuration (e.g. `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "nevis-search-api": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

---

## Document Creation Pipeline (`POST /clients/{id}/documents`)

When a new document is created via `POST /clients/{id}/documents`, `repo.create_document` runs an end-to-end enrichment and indexing pipeline:

1. **Load Defined Category Taxonomy (`llm.load_defined_categories`):**
   - Loads and caches the 16 wealth-management categories from [`data/document_categories.json`](data/document_categories.json) (`category` slug, `name`, `summary`, and `examples`).
2. **OpenAI Structured Output Generation (`llm.analyze_document`):**
   - Calls `AsyncOpenAI.beta.chat.completions.parse` (default model `gpt-6-luna`, configurable via `OPENAI_MODEL`) with `response_format=DocumentAnalysisOutput`.
   - Enforces a structured `{ "summary": str, "category": CategorySlug }` output where `category` is strictly constrained to the defined category slugs.
   - Resolves the matching `DocumentCategoryModel` in PostgreSQL (`category_id`).
   - **Graceful Fallback:** If `OPENAI_API_KEY` is not set or the API call fails, `summary` and `category_id` gracefully default to `None` without failing document creation.
3. **Contextual Chunking & Vector Indexing (`embeddings.build_document_chunks` + `embed_passages`):**
   - Builds `Chunk 0` (combining document title, owning client identity, resolved category taxonomy, and generated summary) plus sliding 200-word content windows (`Chunks 1..N`) for long documents.
   - Computes 768-dimensional embeddings using `FastEmbed` (`BAAI/bge-base-en-v1.5`) and inserts `DocumentChunkModel` rows into `document_chunks` (`HNSW` cosine index) within the same transaction.

---

## Client Search Architecture & Logic

Searching across advisor client books requires handling **compound company/domain names**, **missing or extra spaces**, **typos**, **diacritics/accents**, and **natural-language biographical notes**—while ensuring direct identity matches (`first_name`, `last_name`, `email`) always outrank secondary mentions in `description`.

### 1. Two Precomputed `GENERATED ALWAYS AS ... STORED` Columns + `GIN` Indexes

Rather than computing text transformations on every row at query time (which would force a full table scan), `ClientModel` precomputes two stored columns on `INSERT`/`UPDATE` and indexes them with PostgreSQL `GIN` (Generalized Inverted Index) structures:

1. **`search_vector` (`TSVECTOR` + `GIN` index `ix_clients_search_vector`) — Word/Lexeme Level:**
   - Uses PostgreSQL Full-Text Search (`to_tsvector('english', ...)`).
   - Assigns **Weight `'A'`** (highest priority) to `first_name`, `last_name`, `email`, and punctuation-split email tokens (`john.doe@neviswealth.com` $\rightarrow$ `john`, `doe`, `neviswealth`, `com`).
   - Assigns **Weight `'B'`** (secondary priority) to `description`.
   - Handles English stemming and multi-word boolean queries (e.g., `"estates planning"` matches `"estate planning"`).

2. **`identity_search_text` (`Text` + `GIN gin_trgm_ops` index `ix_clients_identity_trgm`) — Character/Trigram Level:**
   - Stores lowercased `first_name`, `last_name`, `email`, plus a separator-stripped copy of `email` and full name (`john.doe@neviswealth.com` $\rightarrow$ `johndoeneviswealthcom`).
   - Indexed with `gin_trgm_ops` (3-character sliding windows), allowing PostgreSQL to execute $O(\log N)$ **Bitmap Index Scans** for both substring queries (`ILIKE '%...%'`) and fuzzy word-similarity queries (`<%` operator).
   - A separate trigram GIN index (`ix_clients_description_trgm`) indexes `lower(coalesce(description, ''))` so biographical fuzzy/substring matches remain index-backed without polluting identity scores.

### 2. Query Normalization Scoped to Client Search

The `/search?q=...` endpoint passes the raw query `q` untouched to `repo.search_clients(session, q)` so downstream document search receives the original query string. Inside `search_clients`:

- **`q_lower`**: Trimmed, lowercased, and accent-normalized (`unaccent`) query.
- **`q_compact`**: Strips spaces and email/name separators (`re.sub(r"[\s._@-]+", "", q_lower)`), turning `"Nevis wealth"` $\rightarrow$ `"neviswealth"` and `"nefis wealth"` $\rightarrow$ `"nefiswealth"`. Crucially, `q_compact` is **only** matched against `identity_search_text` (never against `description`), preventing false positives across adjacent words in long paragraphs.

### 3. Column-Weighted Relevance Ranking

Candidate rows matched by the GIN indexes (`@@`, `ILIKE`, `<%`) are ordered by a composite score:

$$\text{Score} = 4.0 \cdot \mathbb{I}_{\text{exact\_identity}} + 2.0 \cdot \mathbb{I}_{\text{identity\_substr}} + 1.5 \cdot \text{trgm}_{\text{identity}} + 1.0 \cdot \text{ts\_rank\_cd} + 0.6 \cdot \mathbb{I}_{\text{desc\_substr}} + 0.5 \cdot \text{trgm}_{\text{desc}}$$

This guarantees:
- Searching `"john"` ranks **John Doe** (`first_name = "john"`) above **Jane Doe** (whose `description` says *"spouse of John Doe"*).
- Searching `"NevisWealth"` ranks `@neviswealth.com` team members above external clients whose `description` mentions *"early seed backer of NevisWealth"*.

---

## Example Client Search Queries & Responses

| Query (`GET /search?q=...`) | Matching Mechanism | Top Returned Client(s) |
| :--- | :--- | :--- |
| **`?q=NevisWealth`** | Case-insensitive substring (`ILIKE`) + FTS token match on email domain and description | `john.doe@neviswealth.com`, `jane.doe@neviswealth.com`, `j.doe@neviswealth.com`, `marcus.vance@neviswealth.com`, `r.whitaker1958@gmail.com` |
| **`?q=Nevis wealth`** | Space-split query compacted (`q_compact = "neviswealth"`) matches compacted email in `identity_search_text` | `john.doe@neviswealth.com`, `jane.doe@neviswealth.com`, `j.doe@neviswealth.com`, `marcus.vance@neviswealth.com` |
| **`?q=nefis wealth`** | Typo (`f` $\rightarrow$ `v`) + space-split query matched via `pg_trgm` `word_similarity` (`<%`) on `identity_search_text` | `john.doe@neviswealth.com`, `jane.doe@neviswealth.com`, `j.doe@neviswealth.com`, `marcus.vance@neviswealth.com` |
| **`?q=john`** | Exact `first_name` match (`+4.0` boost) + prefix/substring match | `john.doe@neviswealth.com` (ranked #1), followed by `jane.doe@neviswealth.com` (mentioned in bio) |
| **`?q=estate planning`** | English Full-Text Search (`websearch_to_tsquery`) + phrase match on `description` | `john.doe@neviswealth.com` |
| **`?q=chloe`** | `unaccent` normalization matches `Chloé Dubois` | `chloe.dubois@vanguardia-bio.com` |

### Example Request & Response

```bash
curl -s "http://localhost:8000/search?q=nefis%20wealth"
```

```json
[
  {
    "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "first_name": "john",
    "last_name": "doe",
    "email": "john.doe@neviswealth.com",
    "description": "Managing Director at NevisWealth. High-net-worth fintech executive looking for comprehensive estate planning, revocable living trust setup, and multi-jurisdictional tax optimization between London and New York.",
    "social_links": [
      "https://linkedin.com/in/johndoe-nevis",
      "https://x.com/johndoe_wealth"
    ],
    "type": "client"
  }
]
```

---

## Document Search Architecture & Rationale

Advisors searching client document vaults use a mix of **conceptual terms** (*"address proof"*, *"living will"*, *"electricity invoice"*), **exact financial/legal identifiers** (*"Form 1040"*, *"Schedule K-1"*, *"SBLOC"*), **multi-entity queries** (*"John Doe utility bill"*, *"Alistair Sterling address proof"*), and **typo-heavy queries** (*"adres proof"*, *"utlity bill"*, *"pasport verification"*).

No single retrieval mechanism handles all four cases alone. Before building the document search pipeline, we ran an empirical benchmark (documented in [`DOCUMENT_SEARCH_EVALUATION.md`](DOCUMENT_SEARCH_EVALUATION.md)) evaluating `FastEmbed` (`BAAI/bge-base-en-v1.5`, 768-dim) + `pgvector` across all query classes:

1. **Why Contextual Enrichment is Required for Embeddings:**
   - Embedding raw `title + content` alone failed on documents that did not repeat the category name verbatim (e.g., searching `"address proof"` against an `"EDF Energy Statement"` scored `0.4532`, ranking below an unrelated `"US Passport"` at `0.5331`).
   - Enriching the embedding context with the document's **owner**, **category taxonomy (`name`, `summary`, `examples`)**, and **summary** lifted all `address_proof` documents to **`0.6102 – 0.6473`**, cleanly separating them from unrelated documents (`<= 0.5697`).
2. **Why `0.60` is the Empirically Validated Semantic Gate Threshold:**
   - Every valid conceptual match (`"address proof"`, `"electricity invoice"`, `"estate succession and living will"`, `"pension valuation"`, `"fine art insurance"`) scored **`0.61 – 0.78`**.
   - Unrelated and gibberish queries (`"zzzznonexistentclient99999"`, `"mars rover telemetry"`, `"chocolate cake baking recipe"`) plateaued at **`0.43 – 0.50`**.
   - Gating pure vector candidates at `cosine_similarity >= 0.60` eliminates vector noise while preserving 100% of semantic synonym matches.
3. **Why Vector Search Alone Fails on Severe Typos (Requiring `pg_trgm`):**
   - For `"adres proof"`, subword tokenization degraded `BGE-base-en-v1.5` cosine similarity to `0.5380` (below the `0.60` threshold).
   - PostgreSQL `pg_trgm` `word_similarity('adres proof', category_search_text)` scores **`0.6250`** (`<%` match = `True`), immediately rescuing the query.

### Handling the 512-Token Limit: Contextual Chunking & `document_chunks` Table

`BAAI/bge-base-en-v1.5` enforces a strict **512-token context window**. Stuffing taxonomy metadata and multi-page legal/financial text into a single column on `documents` would silently truncate content past token 512.

To solve this, embeddings live in a dedicated **`document_chunks`** table (`1-to-many` from `documents` with `ON DELETE CASCADE` and an `HNSW` `vector_cosine_ops` index):

- **Chunk `0` (Overview & Taxonomy Chunk):**
  - Encodes `Document: {title}`, `Client: {first_name} {last_name} ({email})`, `Category: {name} ({slug}) - {summary}`, `Related Document Types: {examples}`, and `Summary: {summary}` (plus `Content` directly when `<= 160` words).
- **Chunks `1..N` (Overlapping Content Windows for Long Documents):**
  - Splits long `content` into overlapping **200-word windows** (40-word overlap) prefixed with a compact 1-line header (`Document: {title} | Client: {name} | Category: {category_name}`), keeping every chunk under `~350` tokens (`< 512` token limit).
- **Parent-Level Aggregation at Query Time:**
  - `repo.search_documents` groups `document_chunks` by `document_id` using `MAX(1.0 - (embedding <=> :query_vec)) AS semantic_sim`, so a document matches whether the query targets its high-level category (`Chunk 0`) or a specific clause buried in paragraph 12 (`Chunk k`), without returning duplicate documents.

### 4-Pillar Hybrid Document Ranking Formula

`repo.search_documents_with_scores` joins `documents`, `best_chunk`, `clients`, and `document_categories` in a single SQL query:

1. **Semantic Vector Score (`pgvector` `HNSW`):**
   - `GREATEST(0.0, semantic_sim - 0.50) * 4.0 + (0.8 if semantic_sim >= 0.60)`
2. **Lexical Full-Text Score (`ts_rank_cd` with normalization `33`):**
   - Uses PostgreSQL `ts_rank_cd(..., 1 | 32)` (length normalization `1 + log(len)` + saturation `rank / (rank + 1)`) across `documents.search_vector` (`Weight 'A'`: `title`, `'B'`: `summary`, `'C'`: `content`) combined with `document_categories.search_vector`.
3. **Fuzzy Trigram & Substring Score (`pg_trgm`):**
   - `word_similarity` and `<%` operator against `ix_documents_title_summary_trgm` and `ix_document_categories_trgm`.
4. **Cross-Entity (`Client + Document/Category`) Synergy Boost:**
   - When a query matches **both** the owning client (`clients.search_vector` / `identity_search_text`) **and** the document/category (e.g., `"John Doe utility bill"` or `"Alistair Sterling address proof"`), a synergy boost (`+1.5` + weighted client/combined FTS rank) promotes that specific client's document to `#1`.

### Example Document Search Queries & Responses

| Query (`GET /search?q=...`) | Primary Matching Signal(s) | Top Returned Document(s) |
| :--- | :--- | :--- |
| **`?q=address proof`** | Semantic vector (`0.61–0.65`) + Category FTS/trigram (`address_proof`) | `City of Westminster Annual Council Tax Bill`, `Thames Water & EDF Electricity Utility Bill`, `Eversource Electric & Boston Water Utility Statement` |
| **`?q=electricity invoice`** | Semantic vector (`0.68–0.72`, word *"invoice"* not in text) + partial FTS | `Thames Water & EDF Electricity Utility Bill`, `Eversource Electric & Boston Water Utility Statement` |
| **`?q=estate succession and living will`** | Semantic vector (`0.71`) + Category taxonomy (`estate_planning_and_trusts`) | `The Doe Family Revocable Living Trust Deed` |
| **`?q=John Doe utility bill`** | Cross-Entity Synergy (`Client + Document`) + Semantic (`0.78`) + FTS | `Thames Water & EDF Electricity Utility Bill` (**John Doe**, ranked #1 above Arjun Patel's utility bill) |
| **`?q=Arjun Patel utility bill`** | Cross-Entity Synergy (`Client + Document`) + Semantic (`0.75`) + FTS | `Eversource Electric & Boston Water Utility Statement` (**Dr. Arjun Patel**, ranked #1 above John Doe's utility bill) |
| **`?q=adres proof`** | Category trigram (`word_similarity = 0.625`) + Semantic | All 3 `address_proof` utility & council tax documents |
| **`?q=pasport verification`** | Document title/summary trigram (`0.619`) + Semantic (`0.617`) | `United States Passport - Biometric Identity Verification` |

```bash
curl -s "http://localhost:8000/search?q=address%20proof"
```

```json
[
  {
    "id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
    "client_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "category_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "title": "Thames Water & EDF Electricity Utility Bill - October 2026",
    "content": "Account Holder: John Doe. Service Address: 42 Kensington Palace Gardens, London W8 4QP, United Kingdom...",
    "summary": "October 2026 electricity and water utility bill confirming John Doe's residential address at 42 Kensington Palace Gardens, London.",
    "created_at": "2026-11-20T16:00:00Z",
    "type": "document",
    "category": "address_proof"
  }
]
```

---

## What a Production-Level Search Architecture Looks Like at Scale

While PostgreSQL `ts_rank_cd` + `pgvector` + `pg_trgm` is ideal for this service (zero external infrastructure dependencies, transactional consistency, and single-query cross-table joins), scaling to **10M+ multi-page advisor documents** in production involves the following evolution:

1. **True BM25 Engine (`ParadeDB / pg_search` or Elasticsearch/OpenSearch):**
   - Replace `ts_rank_cd` with a Tantivy-backed BM25 index (`pg_search` inside PostgreSQL or a dedicated search cluster) for true corpus-wide Inverse Document Frequency (IDF) statistics, configurable $k_1$ / $b$ parameters, and fast block-max WAND top-$k$ retrieval over millions of chunks.
2. **Asynchronous Ingestion, OCR & LLM Enrichment Pipeline:**
   - `POST /clients/{id}/documents` should enqueue an asynchronous job (Celery / Temporal / Kafka) that runs layout-aware PDF/OCR extraction (e.g., Unstructured / Docling), invokes an LLM with Structured Outputs to classify `category_id` and generate a grounded `summary`, chunks the document by semantic section headers, and computes embeddings out-of-band without blocking the HTTP response.
3. **Two-Stage Retrieval with Cross-Encoder Reranking & Reciprocal Rank Fusion (RRF):**
   - **Stage 1 (Candidate Generation):** Retrieve the top 100 candidates in parallel from BM25, `pgvector` HNSW, and `pg_trgm`, merging rank lists via **Reciprocal Rank Fusion (RRF)**.
   - **Stage 2 (Reranking):** Pass the top 25 candidates through a lightweight Cross-Encoder reranker (e.g., `BAAI/bge-reranker-v2-m3` or Cohere Rerank) to produce calibrated relevance probabilities and snippet highlights.
4. **Authentication, Authorization & Multi-Tenancy (RBAC / Row-Level Security):**
   - Financial advisors must only search clients and documents within their authorized tenant/book of business via OAuth2/OIDC JWT validation, mandatory `tenant_id` / `advisor_id` pre-filtering on HNSW and GIN scans, and PostgreSQL Row-Level Security (RLS).
5. **Database Migrations (Alembic), PII Compliance & Observability:**
   - Manage schema and index evolution via **Alembic** (`CREATE INDEX CONCURRENTLY`), enforce field-level encryption/tokenization for sensitive PII (SSNs, passport numbers, account numbers), record immutable compliance audit logs for every search query, and enforce per-tenant rate limits and `statement_timeout`.
