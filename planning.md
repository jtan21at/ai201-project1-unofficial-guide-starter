# Project 1 Planning: The Unofficial Guide

## Domain
[What domain did you choose? Why is this knowledge valuable and hard to find through official channels?]

## Documents
[List your specific sources: URLs, subreddit names, forum threads, or file descriptions. Aim for variety — sources that together cover different subtopics or perspectives within your domain.]

## Chunking Strategy
[How will you split documents into chunks? State your chunk size (in tokens or characters), overlap size, and explain why those numbers fit the structure of your documents. A review-heavy corpus warrants different chunking than a long FAQ.

Guiding questions — use these to think it through before deciding:
- Are your documents short reviews (1–3 sentences) or long guides (many paragraphs)? How does that affect the right chunk size?
- If a key fact spans two adjacent chunks, will either chunk be retrievable on its own? What does overlap help with?
- How would you know if your chunks are too small? Too large? What would bad retrieval results look like in each case?

Useful AI prompts:
- "Explain how chunk size affects retrieval quality for short, opinion-based reviews."
- "What are the tradeoffs between chunking by paragraph vs. fixed character count for [my document type]?"
- "If I use 200-character chunks for review text, what kinds of queries might this fail for?"]

## Retrieval Approach
[Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)? How many chunks will you retrieve per query (top-k)? If you were deploying this for real users and cost wasn't a constraint, what tradeoffs would you weigh in choosing a different embedding model — context length, multilingual support, accuracy on domain-specific text, latency?

Guiding questions:
- How many retrieved chunks is enough to give the LLM useful context? What happens if you retrieve too few? Too many?
- Why does semantic search find relevant chunks even when the query doesn't share exact words with the document?

Useful AI prompts:
- "What are different strategies for structuring embeddings for short, opinion-based text?"
- "What does top-k mean in a retrieval system, and what are the tradeoffs of setting it too high vs. too low?"]

## Evaluation Plan
[List your 5 test questions with their expected correct answers. Questions should be specific enough that you can judge whether the system's response is right or wrong — "What are good dining halls?" is too vague; "What do students say about wait times at the [dining hall name] during lunch?" is testable.]

## Anticipated Challenges
[What could go wrong? Consider: noisy or inconsistent documents, missing source attribution, off-topic retrieval, chunks that split key information across boundaries. Name at least two specific risks.]

## AI Tool Plan
[Which parts of the pipeline do you plan to use AI tools (Claude, Copilot, ChatGPT, etc.) to help you implement? For each part, describe what you'll give the AI as input — which sections of this planning.md, which requirements from the instructions — and what you expect it to produce. Be specific: "I'll prompt Claude with my chunking strategy section and ask it to implement the chunk_text() function" is a plan. "I'll use AI to help me code" is not.]

---
 

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:**

**Overlap:**

**Reasoning:**

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**

**Top-k:**

**Production tradeoff reflection:**

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1.

2.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
