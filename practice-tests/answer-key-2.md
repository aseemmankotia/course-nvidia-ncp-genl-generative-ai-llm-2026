# Answer Key — Practice Test 2

**Q1. Correct: A**

GQA groups query heads to share a smaller set of K/V projections, cutting KV cache memory roughly by the grouping factor while preserving quality much closer to full MHA than MQA does.

---

**Q2. Correct: D**

RoPE scaling methods (NTK-aware interpolation, YaRN) adjust the rotation base/frequencies so position encodings remain well-conditioned at longer lengths, and brief fine-tuning adapts the model to the extended range cheaply compared to full retraining.

---

**Q3. Correct: A**

FlashAttention is an IO-aware algorithm designed specifically to reduce HBM read/write traffic by computing attention in tiles within fast on-chip SRAM, directly addressing memory-bandwidth-bound attention rather than compute-bound bottlenecks.

---

**Q4. Correct: B**

Capacity factor directly sets the per-expert token buffer size in a training batch; raising it (e.g., from 1.25 to 2.0) gives experts more headroom to accept tokens before overflow forces drops, directly reducing dropped-token quality loss.

---

**Q5. Correct: C**

The unigram model treats tokenization probabilistically over multiple candidate segmentations and picks the most likely one under a learned vocabulary, which tends to produce more linguistically coherent subwords for complex morphology than BPE's deterministic greedy merge process.

---

**Q6. Correct: A**

Chinchilla's analysis showed that for a fixed compute budget, model size and training token count should scale together roughly proportionally; earlier large models allocated too much compute to parameters and too little to training tokens, making them undertrained relative to compute-optimal allocation.

---

**Q7. Correct: C**

GQA groups query heads into a small number of shared KV head clusters (e.g., 8 groups instead of 1 shared KV head in MQA), cutting KV cache size nearly as much as MQA while preserving more representational diversity in keys/values, which recovers most of the quality lost versus full MHA. This is the documented middle ground used in models like Llama 2/3 70B for exactly this memory-vs-quality tradeoff.

---

**Q8. Correct: B**

Self-consistency samples multiple independent CoT reasoning paths at nonzero temperature, then aggregates (typically majority vote) over the final answers. Because reasoning errors tend to be inconsistent across samples while the correct answer path recurs, marginalizing over samples reduces the impact of any single faulty chain and improves accuracy at the cost of extra inference compute.

---

**Q9. Correct: B**

ReAct relies on the Observation step surfacing tool results (including errors) back into the context so the model can reason about what went wrong and adjust its next Action. If the error/observation isn't clearly returned and the prompt doesn't instruct the model to use it for correction, the model has no signal to change behavior and will repeat the same malformed call.

---

**Q10. Correct: B**

Constrained/structured decoding enforces validity at the token-sampling level by masking out tokens that would violate the grammar or schema at each generation step, guaranteeing syntactic and structural compliance regardless of prompt phrasing — this is the only listed option that provides a hard guarantee rather than a probabilistic improvement.

---

**Q11. Correct: C**

RAG is designed exactly for this scenario: frequently changing factual data, need for verifiable source citations, and no budget/tolerance for retraining. Indexing updates hourly keeps the retrieval corpus current without touching model weights, and retrieved passages can be directly cited, all while maintaining low latency since retrieval + generation is faster than any training cycle.

---

**Q12. Correct: B**

ROUGE-L measures n-gram/longest-common-subsequence overlap with reference text, which correlates loosely with fluency and coverage but does not directly measure factual accuracy or task usefulness. A prompt can game overlap metrics (e.g., by closely mirroring reference phrasing) while still hallucinating or omitting key facts. Best practice for prompt iteration is to combine automated metrics with human or LLM-as-judge evaluation targeting faithfulness/usefulness directly.

---

**Q13. Correct: C**

Research on in-context learning has documented that LLMs are sensitive to the order of few-shot exemplars due to positional/recency biases in attention, causing significant variance in output quality purely from reordering. Practitioners mitigate this by empirically testing several orderings on a validation set, ensembling predictions across orderings, or using calibration techniques, rather than assuming order is irrelevant.

---

**Q14. Correct: A**

RAG is designed exactly for this scenario: fast-changing, factual, structured data that must stay current without retraining. Grounding generation in retrieved live data eliminates stale-knowledge hallucination and updates instantly as the database changes, with no training cost.

---

**Q15. Correct: C**

MinHash+LSH approximates Jaccard similarity between document n-gram sets efficiently at scale, allowing near-duplicate (not just exact) documents to be clustered and filtered — the standard fuzzy deduplication approach used in large-scale corpus curation (e.g., NeMo Curator's fuzzy dedup module).

---

**Q16. Correct: C**

With a tiny labeled dataset and limited compute, LoRA/QLoRA updates only a small number of low-rank adapter parameters, drastically reducing overfitting risk and memory/compute cost compared to full fine-tuning, while still adapting behavior to the instruction format.

---

**Q17. Correct: A**

LoRA rank sets the dimensionality of the low-rank update matrices (A and B); a higher rank means more trainable parameters and greater capacity to fit complex adaptations, but also more memory usage and greater overfitting risk when training data is limited.

---

**Q18. Correct: C**

DPO reformulates preference alignment as a classification-style loss directly on chosen/rejected pairs, avoiding the need for a separate reward model and the instability/complexity of PPO-based RL, making it well-suited to teams without RLHF infrastructure.

---

**Q19. Correct: B**

P-tuning trains a small set of continuous virtual token embeddings (via a prompt encoder such as an LSTM/MLP) that are prepended to the input sequence, leaving the base model frozen, whereas LoRA freezes the base weights but injects trainable low-rank matrices into the weight layers (e.g., attention projections) themselves.

---

**Q20. Correct: B**

NeMo Curator's quality filtering stage applies heuristic rules and trained classifiers (including toxicity/safety and quality classifiers) to score and remove undesirable documents from the corpus before it moves on to deduplication and downstream blending steps.

---

**Q21. Correct: A**

DPO reformulates the RLHF objective as a single supervised loss over preference pairs, directly optimizing the policy against chosen/rejected comparisons without training a separate reward model or running PPO rollouts — making it far more compute- and data-efficient for small preference datasets like this one.

---

**Q22. Correct: A**

MinHash signatures combined with LSH bucketing is the standard scalable technique for near-duplicate detection in large text corpora, and it is the fuzzy-dedup module NeMo Curator uses — it approximates Jaccard similarity between shingled documents without needing full pairwise comparison.

---

**Q23. Correct: D**

Tensor parallelism requires frequent, latency-sensitive all-reduce operations best suited to the high-bandwidth, low-latency NVLink within a node. Pipeline parallelism has lower communication volume (only activations/gradients at stage boundaries) and tolerates the higher latency of InfiniBand across nodes, giving an efficient 3D-parallel mapping.

---

**Q24. Correct: B**

ZeRO Stage 2 shards both optimizer states and gradients across data-parallel ranks, delivering substantial memory savings while keeping full parameters replicated locally, so no parameter all-gather is needed during the forward pass.

---

**Q25. Correct: A**

Transformer Engine applies FP8 with per-tensor dynamic scaling specifically to the compute-heavy GEMM operations in linear layers, since these tolerate reduced precision well, while numerically sensitive operations (layer norm, softmax) and the master copy of weights are kept in BF16/FP32 to maintain training stability and convergence.

---

**Q26. Correct: B**

With only a 20% acceptance rate, most speculated tokens are rejected and must be recomputed by the target model anyway. If the draft model itself is not cheap/fast enough relative to this low acceptance rate, the added draft-model inference cost plus rejection-driven recomputation can exceed the latency of plain autoregressive decoding, producing a net slowdown.

---

**Q27. Correct: C**

AWQ identifies and protects the small subset of weight channels that correspond to large-magnitude activations by scaling them before quantization, preserving accuracy at 4-bit precision with a lightweight calibration process, making it well suited to models with pronounced activation outliers.

---

**Q28. Correct: D**

Gradient checkpointing trades compute for memory: activations are recomputed via an extra forward pass through the checkpointed segments during backpropagation, which typically adds roughly 20-30% additional training time in exchange for substantial memory savings.

---

**Q29. Correct: C**

Sequence parallelism (as implemented in Megatron-LM/NeMo) splits the non-tensor-parallel regions of the transformer (layernorm, dropout) along the sequence dimension, reusing the same communication group and collectives already used by tensor parallelism, so it reduces activation memory with negligible extra communication cost — exactly the scenario described.

---

**Q30. Correct: A**

Speculative decoding lets the draft model generate several candidate tokens per step, which the larger target model verifies in a single forward pass; accepted tokens are emitted without loss of accuracy (verification guarantees identical output distribution), directly cutting the number of sequential decode steps and thus per-request latency.

---

**Q31. Correct: B**

Standard 3D parallelism combines tensor parallelism (intra-node) and pipeline parallelism (inter-node) to split the model, then uses ZeRO stage 1 to shard only optimizer states across the residual data-parallel dimension. This adds memory savings without the added all-gather/reduce-scatter communication that ZeRO-3 style full parameter sharding would introduce on top of an already partitioned model.

---

**Q32. Correct: D**

Dynamic batching with a preferred_batch_size matched to real traffic and a small max_queue_delay_microseconds fills batches for throughput gains without exceeding the millisecond-scale latency budget left after the SLA.

---

**Q33. Correct: C**

Canary rollout exposes only a small fraction of live users to v2, lets real-traffic quality metrics (like hallucination rate) be measured against a defined threshold, and supports automated, near-instant rollback with zero downtime.

---

**Q34. Correct: A**

NIM's multi-LoRA support serves one shared base model on GPU and hot-swaps lightweight adapter weights per request, minimizing memory footprint and enabling fast, dynamic switching across many tenants.

---

**Q35. Correct: D**

Dynamic batching groups independent requests into a batch for a single execution to raise throughput; instance groups configure how many parallel copies of the model run to serve concurrent requests independently — the two are complementary but distinct mechanisms.

---

**Q36. Correct: C**

A slow, steady degradation in answer quality with no infra or code change is the textbook signature of data or concept drift — the corpus or query patterns have shifted away from what the retriever/model were tuned on, warranting re-indexing and a retraining/re-tuning evaluation per standard LLM lifecycle practice.

---

**Q37. Correct: D**

LLM inference load is GPU-bound, so CPU utilization is a poor autoscaling signal; scaling on GPU utilization or Triton's queue depth (via a custom metrics adapter or KEDA) accurately reflects real inference pressure and triggers timely scale-out during bursts.

---

**Q38. Correct: B**

When request arrival gaps (150-300ms) exceed the queue delay, requests are forced to wait the full max_queue_delay before being sent as a partial batch, adding unnecessary latency for sparse traffic. Reducing the delay and adding small preferred batch sizes (e.g., 1,2,4) lets isolated requests flush quickly while still allowing batching to kick in when several requests arrive close together during bursts.

---

**Q39. Correct: D**

Infra metrics like latency and GPU usage do not capture output quality regressions such as factual drift on specific domains. Continuously scoring live canary outputs against a quality benchmark and gating rollout progression on both infra AND quality signals is the only approach that would catch a quality-specific regression before wider exposure.

---

**Q40. Correct: C**

Combining dialogue/topical rails (conversation-level intent control) with output rails (catching drift in generated content) addresses both indirect requests and responses that drift into prohibited territory during generation.

---

**Q41. Correct: B**

A hybrid approach scales via automated judging while a calibration subsample validates reliability by measuring agreement with human raters — the standard practice for trustworthy large-scale LLM-as-judge evaluation.

---

**Q42. Correct: B**

Attribution/grounding scoring (e.g., NLI entailment or citation checking between generated claims and retrieved context) directly measures whether output is supported by sources, which is the core mechanism for detecting RAG hallucinations.

---

**Q43. Correct: D**

Input rails intercept and evaluate the user message before it is passed into the core LLM prompt, making them the correct place to catch prompt-injection and jailbreak patterns before generation occurs.

---

**Q44. Correct: D**

Allocational bias refers to disparities in how resources, opportunities, or outcomes (e.g., loan approvals, hiring recommendations) are distributed across groups as a result of model decisions, distinct from representational bias.

---

**Q45. Correct: B**

Counterfactual/stratified perturbation testing (swapping only the sensitive attribute while holding content constant) isolates the effect of that attribute and is the standard technique for detecting demographic disparate treatment.

---

