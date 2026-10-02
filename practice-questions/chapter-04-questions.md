## Chapter 4: RAG as Adaptation: Prompt vs RAG vs Fine-Tune Decision Framework — Practice Questions

**Q1.** A company must answer queries against internal documents that are updated daily, and has a very limited training/compute budget. Which approach BEST meets these requirements?

A) Fine-tune a foundation model weekly on the updated documents
B) Paste all documents into the prompt using the largest available context window
C) Implement a RAG pipeline with a retriever over a continuously updated document store
D) Use zero-shot prompting with no external knowledge

<details>
<summary>Answer</summary>

**Correct: C**

RAG decouples knowledge freshness from model weights — updating the document store (not the model) keeps answers current with near-zero training cost.

**Why the others are wrong:**
- Weekly fine-tuning is high-cost, slow to update, and doesn't fit a low training budget.
- Pasting all docs in-context doesn't scale, wastes tokens/cost per call, and breaks once documents exceed the context window.
- Zero-shot prompting has no access to internal document content, so it cannot answer accurately.

</details>

---

**Q2.** An application must respond in a strict brand voice and tone consistently across all outputs, and is deployed in a latency-sensitive production environment. Which adaptation strategy is BEST?

A) RAG with brand-voice examples injected into retrieved context
B) Fine-tune the model on brand-voice examples
C) Add few-shot brand-voice examples to every prompt at runtime
D) Increase the sampling temperature to encourage stylistic variety

<details>
<summary>Answer</summary>

**Correct: B**

Fine-tuning bakes behavior/style patterns directly into model weights, giving consistent output without adding runtime retrieval or long few-shot prompts that increase latency and cost.

**Why the others are wrong:**
- RAG is for injecting external knowledge, not for enforcing consistent behavior/style, and adds retrieval latency.
- Few-shot examples on every call increase token cost and latency, and consistency degrades as prompts vary.
- Higher temperature increases randomness, which works against consistent tone.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** A support bot only needs to answer from a static set of 20 FAQ pairs that never change. Which adaptation approach is the simplest and most appropriate?

A) Build a full RAG pipeline with a vector database
B) Include the FAQ content directly in the system prompt
C) Fine-tune a model on the FAQ pairs
D) Deploy a reranker model in front of the LLM

<details>
<summary>Answer</summary>

**Correct: B**

Small, static, in-scope knowledge fits comfortably in the prompt/context window, making direct prompting sufficient without added infrastructure.

**Why the others are wrong:**
- A vector DB and retrieval pipeline is unnecessary overhead for 20 static Q&A pairs.
- Fine-tuning is overkill and costly for a small, unchanging knowledge set.
- A reranker is only useful when there are multiple retrieved candidates to reorder — not applicable here.

</details>

---

**Q4.** Which RAG pipeline component is responsible for splitting long source documents into smaller retrievable units before embedding?

A) Embedding model
B) Reranker
C) Chunking / text splitter
D) Generator LLM

<details>
<summary>Answer</summary>

**Correct: C**

Chunking divides documents into segments sized for effective embedding and retrieval, before any embedding or ranking occurs.

**Why the others are wrong:**
- The embedding model converts chunks into vectors — it doesn't split the document.
- The reranker reorders already-retrieved chunks; it operates after retrieval, not before chunking.
- The generator LLM produces the final answer using retrieved chunks — it has no role in splitting documents.

</details>

---

**Q5.** In a RAG pipeline, which component re-scores and reorders the initially retrieved chunks to improve relevance before they are sent to the generator?

A) Reranker
B) Vector store
C) Embedding model
D) Retriever

<details>
<summary>Answer</summary>

**Correct: A**

The reranker uses a more precise (typically cross-encoder) scoring method to refine the initial top-k results from the retriever before generation.

**Why the others are wrong:**
- The vector store only stores and indexes embeddings; it doesn't re-score results.
- The embedding model produces vector representations used for the initial similarity search, not reranking.
- The retriever performs the initial similarity search that produces the candidate set the reranker later refines.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q6.** A team fine-tunes the embedding model used in their RAG pipeline on domain-specific terminology while keeping the generator model general-purpose. What is the primary goal of this hybrid strategy?

A) Reduce the overall fine-tuning cost of the system
B) Improve retrieval relevance for domain-specific terminology
C) Eliminate the need for a chunking step
D) Reduce generation latency at inference time

<details>
<summary>Answer</summary>

**Correct: B**

Fine-tuning the embedding model improves how well domain-specific terms are represented in vector space, directly improving retrieval precision for that domain.

**Why the others are wrong:**
- Fine-tuning still incurs cost — the goal here is retrieval quality, not cost reduction.
- Chunking is still required regardless of embedding model changes.
- Embedding fine-tuning affects retrieval quality, not generation latency.

</details>

---

**Q7.** A RAG system uses very large chunk sizes (e.g., entire chapters) for indexing. What is the most likely negative effect on retrieval quality?

A) It only increases embedding storage cost with no other effect
B) It dilutes relevance signal, reduces retrieval precision, and may exceed the context window
C) It reduces retrieval accuracy to exactly zero
D) It prevents the reranker from functioning at all

<details>
<summary>Answer</summary>

**Correct: B**

Overly large chunks mix multiple topics into a single vector, weakening the relevance signal for specific queries and risking context-window overflow when passed to the generator.

**Why the others are wrong:**
- Storage cost increases, but the more exam-relevant issue is relevance dilution and context limits.
- Accuracy degrades but doesn't drop to literal zero — this overstates the effect.
- The reranker can still operate on large chunks, just less effectively — it doesn't stop functioning.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** What is the primary cost/latency trade-off distinguishing RAG from fine-tuning as adaptation strategies?

A) RAG requires GPU training while fine-tuning does not
B) Fine-tuning is always cheaper than RAG regardless of scale
C) RAG adds retrieval latency and per-query infrastructure cost, while fine-tuning has high upfront training cost but lower per-query latency
D) RAG cannot be combined with a vector database

<details>
<summary>Answer</summary>

**Correct: C**

RAG shifts cost to runtime (retrieval + vector infra per query), while fine-tuning shifts cost to training time upfront, resulting in faster, lower-overhead inference afterward.

**Why the others are wrong:**
- RAG itself doesn't require GPU training — fine-tuning does; this reverses the relationship.
- Fine-tuning is not always cheaper — at large/dynamic knowledge scale, retraining costs can exceed RAG's ongoing retrieval costs.
- RAG is fundamentally built around vector databases for retrieval — this statement is false.

</details>

---

