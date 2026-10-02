## Chapter 4: RAG as Adaptation: Prompt vs RAG vs Fine-Tune Decision Framework — Flashcards

| # | Front (Question) | Back (Answer) |
|---|-----------------|---------------|
| 1 | When is prompting alone sufficient instead of RAG? | When knowledge is static, small in scope, and fits within the context window. Exam tip: if knowledge is dynamic, large, or external → prefer RAG instead. |
| 2 | What are the 5 stages of a RAG pipeline in order? | Chunking → Embedding → Vector Search (retrieval) → Reranking → Generation. Exam tip: reranker refines the retriever's top-k results before they reach the generator. |
| 3 | When is fine-tuning preferred over RAG? | When the goal is behavior/style/domain-specific pattern learning, or the app is latency-sensitive and can't afford retrieval overhead. Exam tip: fine-tuning changes model weights, not knowledge freshness. |
| 4 | Main cost driver: RAG vs. fine-tuning? | RAG: ongoing per-query retrieval + vector DB infra cost. Fine-tuning: high upfront training compute, low per-query cost after. Exam tip: RAG suits low training budgets. |
| 5 | What does the retriever do in a RAG pipeline? | Performs similarity search (e.g., cosine similarity) over embedded vectors to fetch the top-k relevant chunks. Exam tip: retriever quality depends on embedding model and chunking strategy. |
| 6 | Risk of chunks that are too large? | Dilutes relevance signal, may exceed the model's context window, and lowers retrieval precision. Exam tip: balance chunk size against context window limits. |
| 7 | Risk of chunks that are too small? | Loses surrounding context needed for coherent answers and increases the number of chunks needed per query. Exam tip: too small = fragmented meaning. |
| 8 | Give a hybrid RAG + fine-tuning strategy example. | Fine-tune the embedding/retriever on domain terminology while RAG supplies dynamic facts, OR fine-tune the generator for style while RAG supplies knowledge. Exam tip: hybrid addresses both freshness and behavior needs. |
| 9 | Key differentiator: prompt engineering vs. RAG? | Prompting relies on knowledge already in the model/context; RAG injects external retrieved knowledge at query time. Exam tip: RAG solves knowledge-cutoff and scale limits prompting can't. |
| 10 | What model type is a reranker typically based on? | A cross-encoder that jointly scores query-chunk pairs — more accurate but slower than the bi-encoder used for initial retrieval. Exam tip: reranker trades latency for improved relevance. |
| 11 | What metrics evaluate RAG retrieval quality? | Precision/recall@k for retrieval, and faithfulness/groundedness of the generated answer to retrieved context. Exam tip: evaluate retriever and generator separately. |
| 12 | When is fine-tuning worth its high upfront cost? | When production is latency-sensitive (no retrieval step wanted) or the task needs consistent behavior/style not reliably achievable via prompting/RAG. Exam tip: fine-tuning bakes patterns into weights, removing runtime retrieval cost. |
