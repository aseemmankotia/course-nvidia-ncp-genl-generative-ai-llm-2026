## Chapter 12: Full Practice Exam Simulation and 120-Minute Test-Taking Strategy — Flashcards

| # | Front (Question) | Back (Answer) |
|---|-----------------|---------------|
| 1 | NCP-GENL exam pacing: 65 questions in 120 minutes — target time per question? | ~1.5-2 minutes/question, leaving 10-15 minutes buffer at the end for flagged-question review. |
| 2 | Approx. GPU memory for a 70B-param model at FP16 vs INT4? | FP16 (~2 bytes/param): ~140GB. INT4 (~0.5 bytes/param): ~35GB. Exam tip: match precision to available VRAM. |
| 3 | RAG vs Fine-tuning — when to choose which? | RAG: knowledge changes frequently, need up-to-date facts, low retraining budget. Fine-tuning: stable task/behavior/format change that prompting can't fix reliably. |
| 4 | Tensor parallelism vs Pipeline parallelism — key difference? | Tensor: splits individual layer/matrix ops across GPUs (high intra-layer communication). Pipeline: splits sequential layers/stages across GPUs (stage-to-stage communication). |
| 5 | What serving feature maximizes throughput for variable-length concurrent LLM requests? | In-flight (continuous) batching — new requests join the batch as others finish generating, unlike static batching which waits for the longest sequence. |
| 6 | LoRA — what does it freeze and what does it train? | Freezes original base model weights; trains small low-rank adapter matrices injected into layers. Drastically reduces trainable params vs full fine-tuning. |
| 7 | What grows the KV cache size during LLM inference? | Sequence length and batch size (and number of layers/heads). Longer context + more concurrent requests = larger KV cache memory footprint. |
| 8 | Data parallelism vs Model (tensor/pipeline) parallelism — when is each required? | Data parallelism: model fits on one GPU, replicate model, split data. Model parallelism: model itself doesn't fit on one GPU, must split the model across GPUs. |
| 9 | Elimination strategy tip: how to quickly cut wrong answers on scenario questions? | Eliminate options that use deprecated/mismatched services first, then options that solve only part of the stated requirement, then the most expensive/complex option if a simpler one satisfies constraints. |
| 10 | Post-training quantization methods for LLMs (e.g., AWQ, GPTQ) vs FP8 — key distinction? | AWQ/GPTQ: aggressive weight-only quantization (commonly INT4) for memory reduction with some accuracy loss. FP8: lighter-weight precision reduction often used for both weights and activations with less accuracy impact than INT4. |
| 11 | Final 15 minutes of the exam — what should you do? | Review all flagged questions first, re-verify any answer you were uncertain about, then use any remaining time for a general pass over unanswered items. |
| 12 | Classic cross-domain decision pair tested repeatedly: prompt engineering vs fine-tuning vs RAG — first question to ask? | Is the problem missing KNOWLEDGE (→ RAG), missing BEHAVIOR/FORMAT consistency (→ fine-tune), or solvable with better INSTRUCTIONS/EXAMPLES (→ prompt engineering)? |
