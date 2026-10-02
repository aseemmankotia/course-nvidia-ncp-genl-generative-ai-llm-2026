## Prompt vs RAG vs Fine-Tune Decision Framework

| Strategy | Best When | Training Cost | Per-Query Latency | Knowledge Freshness |
|---|---|---|---|---|
| Prompting only | Static, small-scope knowledge fits in context | None | Low | Fixed at prompt time |
| RAG | Dynamic/large external knowledge, low training budget | Low (no retraining) | Higher (retrieval step) | Updated by refreshing index |
| Fine-tuning | Behavior/style/domain patterns, latency-sensitive apps | High (upfront) | Low (no retrieval) | Frozen at training time |
| Hybrid (RAG + FT) | Need both fresh knowledge AND consistent behavior/style/domain retrieval | Moderate-High | Moderate | Updated (knowledge) + consistent (behavior) |

### RAG Pipeline Stages (in order)
1. **Chunking** — split documents into retrievable units
2. **Embedding** — convert chunks/query into vectors
3. **Vector Search (Retrieval)** — similarity search (e.g., cosine) for top-k candidates
4. **Reranking** — cross-encoder rescoring of top-k for precision
5. **Generation** — LLM produces answer grounded in reranked context

### Chunking Trade-offs
- **Too large:** dilutes relevance, risks exceeding context window
- **Too small:** loses surrounding context, fragments meaning, more chunks needed
- Rule of thumb: chunk size should preserve one coherent idea/topic per chunk

### Decision Triggers
- **Use Prompting** → knowledge is small, static, in-scope
- **Use RAG** → knowledge is large, external, frequently changing; budget for training is low
- **Use Fine-Tuning** → need consistent tone/behavior/domain reasoning patterns; latency-sensitive (no retrieval overhead acceptable)
- **Use Hybrid** → need both current knowledge (RAG) and consistent style/domain vocabulary (fine-tuned embedding or generator)

### Cost/Latency Summary
- RAG shifts cost to **runtime** (vector DB, retrieval calls, reranking)
- Fine-tuning shifts cost to **training time** (GPU compute upfront), then runs cheaper/faster per query
- Hybrid approaches balance both, often fine-tuning the embedding model (cheaper than fine-tuning the generator) while keeping RAG for knowledge freshness

### Evaluation Loop
- Evaluate **retriever** separately: precision@k, recall@k
- Evaluate **generator** separately: faithfulness/groundedness to retrieved context, answer relevance
- Common exam trap: evaluating only final answer quality and missing that a poor retriever (not the LLM) is the root cause of bad answers

### Common Exam Traps
- Choosing fine-tuning to fix a **knowledge freshness** problem (wrong — use RAG)
- Choosing RAG to fix a **style/behavior consistency** problem (wrong — use fine-tuning)
- Assuming larger chunks always improve retrieval (wrong — dilutes relevance)
- Forgetting the reranker step and assuming retriever output alone is optimal for generation
