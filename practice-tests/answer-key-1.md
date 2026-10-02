# Answer Key — Practice Test 1

**Q1. Correct: A**

GQA groups query heads into clusters that share KV projections, cutting KV-cache memory and bandwidth substantially versus full MHA while retaining most heads' representational diversity, giving a much smaller quality drop than MQA's single shared KV pair.

---

**Q2. Correct: A**

MoE architectures activate only a small subset of experts per token via a learned router, so total parameter count (capacity) can scale into the hundreds of billions while active FLOPs per token stay near a much smaller dense model's cost.

---

**Q3. Correct: D**

RoPE's frequency base (theta) controls how position information is encoded across dimensions; NTK-aware or linear scaling adjusts this base so unseen longer positions map to a comparable rotational range as training, extending usable context with little or no fine-tuning.

---

**Q4. Correct: A**

FlashAttention is an exact attention algorithm that fuses the QK^T, softmax, and weighted-value steps using SRAM tiling and recomputation, avoiding writing/reading the full N×N score matrix to HBM, which cuts memory I/O and speeds up training/inference without changing results.

---

**Q5. Correct: D**

Retraining the subword vocabulary on the actual target-language corpus lets BPE/SentencePiece learn merges that match that language's morphology, shortening average token length per sentence and directly reducing sequence length and compute cost.

---

**Q6. Correct: B**

Encoder-decoder models explicitly separate the roles of encoding the full source sequence bidirectionally and autoregressively decoding a distinct target sequence via cross-attention, matching the asymmetric input/output structure of summarization better than architectures without this separation.

---

**Q7. Correct: C**

GQA groups multiple query heads to share a smaller set of key/value heads, drastically shrinking the KV cache size stored during autoregressive decoding compared to standard MHA, which improves inference throughput and memory efficiency while retaining most of the quality of full MHA.

---

**Q8. Correct: B**

RAG decouples knowledge freshness from model weights: updating the vector index is near-instant and cheap compared to retraining, and retrieved passages naturally provide citable sources for grounding the answer.

---

**Q9. Correct: B**

With abundant labeled data, a fixed consistent output format, and a hard low-latency/no-external-call constraint, fine-tuning bakes the behavior directly into the weights, eliminating retrieval latency and long reasoning chains while maximizing consistency.

---

**Q10. Correct: C**

Chain-of-thought (CoT) prompting explicitly elicits intermediate reasoning steps before the final answer, which improves accuracy on multi-step arithmetic and logic tasks compared to direct answer generation.

---

**Q11. Correct: A**

ReAct (Reason + Act) explicitly structures the prompt so the model alternates between a reasoning trace, an action (e.g., a tool/API call), and an observation of the result, which is exactly the loop needed to query an API and adapt based on its response.

---

**Q12. Correct: A**

Constrained (grammar-guided) decoding restricts the token sampling process itself so that only tokens valid under the target schema/grammar can be generated, mathematically guaranteeing structural validity rather than merely encouraging it through instructions.

---

**Q13. Correct: C**

Self-consistency generates several diverse chain-of-thought reasoning paths (typically via temperature sampling) for the same prompt and aggregates the final answers by majority vote, which reduces the impact of any single flawed reasoning chain and improves overall accuracy.

---

**Q14. Correct: D**

RAG decouples knowledge updates from model training: reindexing the vector store after a document change propagates the update within minutes, requires no GPU retraining budget, and naturally provides source attribution since responses are grounded in retrieved passages -- exactly matching the freshness, cost, and traceability constraints in the scenario.

---

**Q15. Correct: C**

QLoRA quantizes the frozen base model weights to 4-bit precision and trains only small LoRA adapter matrices in higher precision, dramatically reducing memory footprint so a 70B model can be fine-tuned on a single 80GB GPU, and it works well with small labeled datasets.

---

**Q16. Correct: D**

DPO directly optimizes the policy model on preference pairs (chosen vs. rejected responses) using a closed-form loss derived from the RLHF objective, eliminating the need for a separate reward model or an RL training loop like PPO.

---

**Q17. Correct: B**

The LoRA rank determines the dimensionality of the low-rank decomposition matrices injected into the frozen weight layers; a higher rank increases the number of trainable parameters and the adapter's capacity to represent complex adaptations, at the cost of more memory and compute.

---

**Q18. Correct: A**

Near-duplicate removal reduces the risk of the model overfitting or memorizing frequently repeated passages, which improves generalization, training efficiency, and reduces the chance of verbatim data leakage during generation.

---

**Q19. Correct: A**

Continued pretraining (domain-adaptive pretraining) uses the standard language modeling objective on unlabeled domain text, allowing the model to absorb domain vocabulary, style, and knowledge without requiring labeled instruction data.

---

**Q20. Correct: D**

NeMo Curator is the NVIDIA library specifically designed for scalable data curation pipelines, including deduplication (exact and fuzzy), quality classification, filtering, and PII redaction, to prepare high-quality datasets prior to pretraining or fine-tuning.

---

**Q21. Correct: A**

LoRA is designed exactly for this scenario: small labeled dataset, limited compute, and a need to adapt behavior (instruction following, terminology) without touching the full weight set. It trains far fewer parameters, converges quickly on small data, and avoids catastrophic forgetting.

---

**Q22. Correct: C**

QLoRA quantizes the frozen base model weights to 4-bit NormalFloat (NF4), a data-type optimized for normally distributed weights, and adds double quantization (quantizing the quantization constants themselves) plus paged optimizers to manage GPU memory spikes, enabling fine-tuning of large models on a single GPU.

---

**Q23. Correct: D**

Tensor parallelism requires frequent, low-latency all-reduce communication, so it should be confined to the high-bandwidth NVLink domain within a node. Pipeline parallelism only exchanges activations at stage boundaries, making it tolerant of the lower-bandwidth inter-node InfiniBand links. This 3D-parallelism pattern (TP intra-node, PP inter-node, DP for scaling) is the standard NVIDIA-recommended topology-aware strategy.

---

**Q24. Correct: B**

AWQ (Activation-aware Weight Quantization) observes activation magnitudes to determine which weight channels are most salient and scales/protects those channels, avoiding the costly per-layer Hessian-based reconstruction optimization that GPTQ performs.

---

**Q25. Correct: C**

Gradient checkpointing saves memory by discarding intermediate activations during the forward pass and recomputing them on demand during backward propagation. This recomputation adds extra forward-pass compute, typically increasing wall-clock training time by roughly 20-30% in exchange for the memory savings.

---

**Q26. Correct: A**

ZeRO-3 extends sharding to all three components: optimizer states, gradients, and parameters themselves, with parameters all-gathered just-in-time for each layer's computation. This provides the maximum memory reduction of the standard ZeRO stages.

---

**Q27. Correct: D**

Continuous (in-flight) batching, as implemented in TensorRT-LLM, processes generation iteration-by-iteration rather than request-by-request, immediately freeing a finished sequence's slot for a new incoming request. This directly addresses idle-GPU time caused by variable-length sequences in a static batch.

---

**Q28. Correct: D**

Speculative decoding uses a smaller draft model to generate several candidate tokens quickly, which the larger target model then verifies in a single parallel forward pass, accepting correct tokens and only falling back to standard decoding on mismatches. This reduces the number of sequential large-model forward passes needed per output token, lowering latency while preserving the target model's exact output distribution.

---

**Q29. Correct: A**

Continuous/in-flight batching lets new requests join an already-running batch as soon as GPU slots free up instead of waiting for the whole batch to finish, while paged KV-cache allocates memory in fixed-size blocks (similar to PagedAttention), eliminating fragmentation and letting memory be reused dynamically across variable-length sequences.

---

**Q30. Correct: B**

AWQ identifies salient weight channels by observing activation magnitudes and applies per-channel scaling to protect those channels during quantization, achieving strong INT4 accuracy without any backpropagation or layer-wise reconstruction pass.

---

**Q31. Correct: B**

Only ZeRO Stage 3 shards parameters, gradients, and optimizer states together across all GPUs, reducing per-GPU memory enough to fit a 70B model whose combined footprint exceeds a single 80GB GPU even in mixed precision.

---

**Q32. Correct: A**

Canary rollouts combined with automated metric-based rollback provide progressive exposure and closed-loop safety without requiring a human to watch dashboards, directly satisfying the 'automatic rollback based on live metrics' requirement.

---

**Q33. Correct: C**

Dynamic batching aggregates concurrent requests into larger GPU batches, improving GPU utilization and throughput, which is exactly the symptom described (low utilization despite high request volume).

---

**Q34. Correct: B**

instance_group defines how many copies of the model are instantiated and on which device type (e.g., KIND_GPU), directly controlling concurrent execution capacity per GPU.

---

**Q35. Correct: C**

CPU utilization is a poor proxy for GPU-bound LLM workloads; scaling on queue depth or GPU utilization with pre-warmed capacity addresses both the wrong signal and the cold-start latency that causes SLA breaches during spikes.

---

**Q36. Correct: A**

Larger batches let the GPU process more requests per kernel launch, raising aggregate throughput, but each individual request waits longer for the batch to fill and complete, increasing per-request latency, until the GPU saturates.

---

**Q37. Correct: D**

A gradual rise in low-confidence or out-of-distribution outputs with no infra changes is a classic signature of data or concept drift; the correct lifecycle response is drift monitoring/detection followed by retraining or fine-tuning on fresh data.

---

**Q38. Correct: B**

A canary rollout exposes the new model version to a controlled, small subset of real production traffic and lets teams observe live metrics (latency, error rate, quality signals) before progressively increasing exposure, directly satisfying 'validate on real traffic with minimized risk'.

---

**Q39. Correct: B**

High queue times combined with low GPU compute utilization indicate requests are waiting for an available model instance rather than saturating compute. Raising instance_group count lets Triton run multiple concurrent execution contexts for that one model on the GPU, directly reducing queuing without touching the other two models' configs.

---

**Q40. Correct: C**

Hallucination detection specifically evaluates whether generated claims are grounded in and supported by retrieved/source context, which is exactly the failure described (confident but unsupported factual claims).

---

**Q41. Correct: D**

LLM-as-judge is designed for open-ended generation tasks lacking a single correct answer, using a rubric to approximate nuanced human-like judgment at scale, which fits creative summarization evaluation.

---

**Q42. Correct: C**

NeMo Guardrails uses Colang-defined flows and topical/dialogue rails to detect specific user intents (like competitor pricing) and block or redirect the conversation, which is the intended mechanism for topic restriction.

---

**Q43. Correct: D**

Hallucination is about factual/grounding accuracy of generated content, whereas bias is about systematic unfairness or skew across demographic or ideological dimensions — these are distinct failure modes requiring different evaluation methods.

---

**Q44. Correct: B**

Privacy compliance requires systematic PII detection/redaction integrated at multiple lifecycle stages (data ingestion, logging, output) plus governance policies controlling data handling — this is the standard responsible-AI approach to PII risk.

---

**Q45. Correct: C**

Automated benchmarks like accuracy/F1 often fail to capture subjective qualities like helpfulness and fluency; structured human evaluation with rubrics is specifically designed to catch this gap between benchmark performance and real-world user experience.

---

