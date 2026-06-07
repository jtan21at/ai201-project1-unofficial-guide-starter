# The Unofficial Guide — Project 1

## Domain
This system covers **student-reported off-campus housing experiences**. The knowledge is valuable because official apartment listings do not surface operational realities like maintenance delays, noise patterns, mold risk, and leasing friction. Those details exist in scattered informal posts and are hard to search with regular keyword lookup.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|------------------|
| 1 | Hillview forum thread | Local text file | `documents/housing/01_hillview_forum.txt` |
| 2 | Cedar Court Reddit summary | Local text file | `documents/housing/02_cedar_reddit.txt` |
| 3 | Rivergate review digest | Local text file | `documents/housing/03_rivergate_review.txt` |
| 4 | Oak Crossing Q&A archive | Local text file | `documents/housing/04_oak_crossing_forum.txt` |
| 5 | Maple Square notes | Local markdown file | `documents/housing/05_maple_square_blog.md` |
| 6 | Harbor Point HTML-like review | Local text file | `documents/housing/06_harbor_point_html.txt` |
| 7 | Elm Lofts spreadsheet notes | Local text file | `documents/housing/07_elm_lofts_sheet.txt` |
| 8 | Pine Gardens Discord transcript | Local text file | `documents/housing/08_pine_gardens_discord.txt` |
| 9 | Summit House FAQ | Local text file | `documents/housing/09_summit_house_faq.txt` |
| 10 | Lakeside student thread | Local text file | `documents/housing/10_lakeside_thread.txt` |

---

## Chunking Strategy

**Chunk size:** 600 characters  
**Overlap:** 120 characters

**Why these choices fit your documents:**
My corpus is mostly short review-style documents, so I need chunks large enough to keep related claims together (e.g., pricing + utilities + service complaints) but not so large that unrelated facts dilute retrieval. A 120-character overlap helps preserve context if key claims lie near chunk boundaries. The chunker also prefers sentence/paragraph boundaries to avoid broken phrases.

**Final chunk count:** 10 chunks (1 chunk per source document for this dataset).

---

## Embedding Model

**Model used:** Primary target is `all-MiniLM-L6-v2` via sentence-transformers. In this sandbox run, outbound model download was blocked, so indexing/retrieval used the implemented local `hashing-fallback` embedding backend.

**Production tradeoff reflection:**
For production, I would benchmark multiple embedding models on domain-specific retrieval precision, especially for paraphrased housing questions. I would weigh model quality vs latency/cost, multilingual support (if documents include non-English comments), and whether local hosting or an API-hosted model is more reliable for my deployment constraints.

---

## Grounded Generation

**System prompt grounding instruction:**
The generation step instructs the model to answer **only** from retrieved context and return "I don't have enough evidence in the retrieved documents" when evidence is missing. It also requires source citations in `[source:chunk]` form.

**How source attribution is surfaced in the response:**
Each retrieved chunk is passed with explicit source metadata (`source` + `chunk_index`), and responses include citations tied to those chunk identifiers. The CLI also prints the retrieved chunks and similarity distances so users can inspect grounding evidence.

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Is Hillview's housing lottery fully random for all applicants? | Renewals first, then lottery for remaining units | Correctly states renewals are processed first and lottery is applied afterward for remaining units (`housing/01_hillview_forum.txt`). | Relevant | Accurate |
| 2 | Which housing option is described as the cheapest two-bedroom and about how much does it cost per person? | Cedar Court, about $980/person + utilities | Correctly identifies Cedar Court and ~$980/person, but adds irrelevant extra lines from other chunks. | Partially relevant | Partially accurate |
| 3 | Which complex has very slow non-emergency maintenance and how long do requests take? | Pine Gardens, typically 5–7 days | Correctly identifies Pine Gardens and 5–7 day delay, but includes one unrelated Elm Lofts line. | Partially relevant | Partially accurate |
| 4 | At Summit House, which utilities are included and which are extra? | Water/trash included; electricity/parking extra | Correctly returns utility split from Summit House chunk; includes one extra irrelevant line from a different chunk. | Relevant | Accurate |
| 5 | Which place has repeated mold complaints and what precaution did students recommend? | Lakeside; pre-move inspection + photo documentation | Correctly returns Lakeside mold issue and recommended mitigation steps, with proper source chunk. | Relevant | Accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:** Which complex has very slow non-emergency maintenance and how long do requests take?

**What the system returned:**
It correctly cited Pine Gardens with 5–7 day delays, but added an unrelated Elm Lofts pricing sentence.

**Root cause (tied to a specific pipeline stage):**
The retrieval stage returned a semantically weaker secondary chunk due broad lexical overlap in short documents. The generation stage (offline extractive mode) selected additional lines from top-k chunks without strong reranking/filtering.

**What you would change to fix it:**
Add a reranking/filtering pass (e.g., keyword-aware or cross-encoder reranker) before generation, and require answer-line selection to meet a minimum relevance threshold. I would also test smaller top-k for narrow factual questions.

---

## Spec Reflection

**One way the spec helped you during implementation:**
The spec forced me to choose chunking and retrieval settings before coding, which made implementation decisions concrete. That made it easy to verify whether the output matched design intent (chunk readability, source traceability, and top-k behavior).

**One way your implementation diverged from the spec, and why:**
The planned embedding model was all-MiniLM-L6-v2, but in this sandbox environment model download from Hugging Face was blocked. I added a local hashing embedding fallback so the full pipeline (index, retrieve, answer, evaluate) can still run end-to-end without external model download.

---

## AI Usage

**Instance 1**

- *What I gave the AI:* Chunking and ingestion requirements from planning.md (document types, preprocessing goals, chunk size/overlap intent).
- *What it produced:* A Python ingestion/chunking pipeline with cleaning, sentence-boundary-aware splitting, and CLI integration.
- *What I changed or overrode:* Hardened HTML cleanup, refined chunk metadata, and added fallback-safe behavior for offline environments.

**Instance 2**

- *What I gave the AI:* Grounding and citation requirements for generation/evaluation.
- *What it produced:* Retrieval-grounded answer flow with source-attributed output and query CLI.
- *What I changed or overrode:* Added an offline extractive answer mode and a local embedding fallback to complete milestones without external API/model availability.

---

## Local CLI (Implemented)

This repository includes a starter RAG CLI at `rag_app.py`.

### 1) Index your documents

```bash
python rag_app.py index --documents-dir documents --reindex
```

If the sentence-transformer model cannot be downloaded in your environment, use:

```bash
python rag_app.py index --documents-dir documents --embedding-model hashing-fallback --reindex
```

### 2) Ask a grounded question

Groq-backed mode (requires `.env` with `GROQ_API_KEY`):

```bash
python rag_app.py ask "Is the housing lottery actually random?"
```

Offline extractive mode (no API key needed):

```bash
python rag_app.py ask "Is the housing lottery actually random?" --embedding-model hashing-fallback --offline
```

