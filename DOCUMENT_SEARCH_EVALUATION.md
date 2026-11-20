# Document Search Evaluation: `BGE-base-en-v1.5` + `pgvector`

Empirical benchmark evaluating **`FastEmbed` (`BAAI/bge-base-en-v1.5`, 768 dimensions)** with **PostgreSQL 18 `pgvector`** cosine similarity (`1 - (embedding <=> query_vec)`) against the synthetic dataset (`15` documents across `5` clients and `15` categories, plus an unannotated `"EDF Energy Statement"` containing no literal `"address"` or `"proof"` tokens).

---

## 1. Embedding Context Comparison

Three text templates were tested for generating document embeddings:

1. **`raw`**: `title + content`
2. **`enr`**: `title + client_name + category(name, slug, summary) + doc_summary + content`
3. **`ex`**: `enr` + `category.examples` (from `data/document_categories.json`)

### Query: `"address proof"` (Core PDF Requirement)

| Document | Category | `raw` | `enr` | `ex` |
| :--- | :--- | :---: | :---: | :---: |
| City of Westminster Annual Council Tax Bill | `address_proof` | `0.5550` | `0.6178` | **`0.6473`** |
| Eversource Electric & Boston Water Statement | `address_proof` | `0.5134` | `0.5973` | **`0.6463`** |
| EDF Energy Statement *(no literal "address/proof" in text)* | `address_proof` | `0.4532` | `0.5856` | **`0.6412`** |
| Thames Water & EDF Electricity Utility Bill | `address_proof` | `0.5037` | `0.5705` | **`0.6102`** |
| *Top non-matching:* United States Passport | `identity_verification` | `0.5331` | `0.5635` | `0.5697` |

**Finding:** `raw` embeddings fail on `"EDF Energy Statement"` (`0.4532` < `0.5331` for US Passport). Including category taxonomy and examples (`ex`) lifts all `address_proof` documents to **`0.6102 – 0.6473`**, creating a clean margin above the highest non-matching document (`0.5697`).

---

## 2. Cosine Similarity Across Search Scenarios

| Scenario | Query | Top Matched Document (`category`) | `enr` Score | `raw` Score | Next Unrelated Doc |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Semantic Synonym** | `"utility bill"` | Utility Bill - Oct 2026 (`address_proof`) | **`0.7354`** | `0.7547` | `0.4901` |
| **Semantic Synonym** | `"electricity invoice"` | EDF Energy Statement (`address_proof`) | **`0.7172`** | `0.6650` | `0.4850` |
| **Semantic Synonym** | `"estate succession and living will"` | Doe Family Revocable Living Trust (`estate_planning_and_trusts`) | **`0.7057`** | `0.6468` | `0.5245` |
| **Semantic Synonym** | `"pension valuation"` | Meridian Executive SIPP Valuation (`retirement_and_pensions`) | **`0.7282`** | `0.7158` | `0.5677` |
| **Semantic Synonym** | `"fine art insurance"` | Chubb Masterpiece Fine Art Schedule (`insurance_and_risk_management`) | **`0.7335`** | `0.6735` | `0.5874` |
| **Semantic Synonym** | `"private equity capital call"` | Apex AI Ventures Capital Call #4 (`alternative_and_private_investments`) | **`0.7017`** | `0.6771` | `0.6015` |
| **Semantic Synonym** | `"tax return"` | US Form 1040 & UK HMRC Tax Return (`tax_filings_and_reporting`) | **`0.6701`** | `0.6650` | `0.5674` |
| **Semantic Synonym** | `"source of wealth"` | Notarized Estate Inheritance Attestation (`income_and_source_of_wealth`) | **`0.6603`** | `0.5930` | `0.5554` |
| **Multi-Entity** | `"John Doe utility bill"` | John Doe — Thames Water & EDF Utility Bill | **`0.7836`** | `0.7569` | `0.6531` *(Arjun Patel)* |
| **Multi-Entity** | `"Arjun Patel utility bill"` | Arjun Patel — Eversource Utility Statement | **`0.7460`** | `0.7300` | `0.6797` *(John Doe)* |
| **Multi-Entity** | `"Alistair Sterling address proof"` | Alistair Sterling — Westminster Council Tax | **`0.7173`** | `0.6312` | `0.6396` *(John Doe)* |
| **Multi-Entity** | `"Zoe Laurent inheritance"` | Zoë Laurent — Estate Inheritance Attestation | **`0.7402`** | `0.7633` | `0.5355` *(John Doe)* |
| **Typo** | `"utlity bill"` | Utility Bill - Oct 2026 (`address_proof`) | **`0.6574`** | `0.6489` | `0.4820` |
| **Typo** | `"pasport verification"` | US Passport (`identity_verification`) | **`0.6173`** | `0.5635` | `0.5724` |
| **Typo (Failure)** | `"adres proof"` | Estate Inheritance (`0.5453`) vs. Utility Bill (`0.5380`) | **`0.5380`** | `0.4802` | `0.5453` *(False #1)* |
| **Negative / Noise** | `"zzzznonexistentclient99999"` | Eversource Utility Statement (Noise) | `0.4988` | `0.4527` | — |
| **Negative / Noise** | `"mars rover telemetry"` | Lombard Odier Credit Facility (Noise) | `0.4639` | `0.4786` | — |
| **Negative / Noise** | `"chocolate cake baking recipe"` | Vanguardia Founders Equity Grant (Noise) | `0.4310` | `0.3698` | — |

---

## 3. Architecture Decisions Driven by Data

1. **Enriched Embedding Template (`ex`):**
   - Documents must be embedded with their `title`, `summary`, `content`, owner (`first_name`, `last_name`), and assigned category (`name`, `summary`, `examples`). This boosts concept-to-document recall (`"address proof"` $\rightarrow$ `"EDF Energy Statement"` from `0.4532` to `0.6412`) and owner-specific queries (`"John Doe utility bill"` to `0.7836`).
2. **Vector Candidate Gate Threshold (`cosine_similarity >= 0.60`):**
   - Valid semantic matches consistently score **`0.61 – 0.78`**.
   - Irrelevant and gibberish queries plateau at **`0.43 – 0.50`**.
   - A `0.60` minimum similarity threshold eliminates vector noise while preserving semantic recall.
3. **Why Hybrid Search (FTS + `pgvector` + `pg_trgm`) is Required:**
   - **Vector alone fails on severe token typos** (`"adres proof"` drops to `0.5380`, below the `0.60` threshold, and ranks an unrelated document at `0.5453`).
   - **`pg_trgm` (`word_similarity`)** scores `'adres proof'` against `'address proof'` at **`0.714`**, recovering the match when combined with category/document trigram indexes.
   - **PostgreSQL FTS (`ts_rank_cd`)** provides exact token and phrase precision (e.g., `"Form 1040"`, `"Schedule K-1"`, `"SBLOC"`, account numbers) and cross-entity boolean matching across joined `DocumentModel + ClientModel + DocumentCategoryModel` columns.
