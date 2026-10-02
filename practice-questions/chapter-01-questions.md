## Chapter 1: Transformer Internals: Attention, Positional Encoding, and Efficiency Variants — Practice Questions

**Q1.** You are deploying a 70B parameter LLM for real-time chat inference on GPUs with limited memory bandwidth. The model must support long context windows without exceeding KV cache memory during autoregressive generation. Which attention mechanism should you select?

A) Grouped-query attention (GQA) with a reduced number of shared KV heads
B) Standard multi-head attention (MHA) with full KV heads
C) Encoder-decoder cross-attention
D) Sparse Mixture-of-Experts attention

<details>
<summary>Answer</summary>

**Correct: A**

GQA groups multiple query heads per KV head, shrinking KV cache substantially (e.g., 4-8x) versus MHA while retaining most of MHA's quality, ideal for memory-constrained long-context inference.

**Why the others are wrong:**
- MHA keeps a full KV head per query head, maximizing KV cache size and memory bandwidth pressure — the opposite of what's needed.
- Encoder-decoder cross-attention is an architectural pattern for seq2seq tasks, not a KV-cache-reduction technique for decoder-only chat inference.
- MoE is a sparse activation technique for feed-forward layers, unrelated to attention KV cache size.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** Your team is training a transformer on sequences of 32K tokens and observes the training step is memory-bound rather than compute-bound, with the O(N^2) attention matrix causing repeated HBM reads/writes. Which technique directly addresses this bottleneck without changing the attention math's output?

A) Switch to multi-query attention
B) Apply FlashAttention's fused, IO-aware kernel
C) Reduce the number of attention heads
D) Replace attention with a mixture-of-experts layer

<details>
<summary>Answer</summary>

**Correct: B**

FlashAttention tiles computation to keep Q/K/V blocks in fast SRAM, avoiding materialization of the full N×N score matrix in HBM. It is mathematically exact and converts the operation from memory-bound toward compute-bound.

**Why the others are wrong:**
- MQA changes attention's parameter/cache footprint but doesn't address IO patterns of the score matrix during training compute.
- Reducing heads changes model capacity/output, violating 'without changing the attention math's output.'
- MoE replaces FFN sublayers, not attention; irrelevant to the attention memory bottleneck.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** A company wants to scale a model's total parameter count 6x to improve capacity, but their inference budget only allows a 1.5x increase in FLOPs per token. Which architecture choice best satisfies this constraint?

A) Use a Mixture-of-Experts (MoE) layer with top-2 gating over many experts
B) Increase the number of standard dense transformer layers
C) Switch from decoder-only to encoder-decoder
D) Increase the number of attention heads proportionally

<details>
<summary>Answer</summary>

**Correct: A**

MoE stores many experts (large total parameter count) but activates only top-k (e.g., top-2) per token via sparse gating, so active compute grows far more slowly than total parameters — matching large capacity gain with modest compute increase.

**Why the others are wrong:**
- Dense layers scale FLOPs linearly with parameters, violating the compute budget.
- Encoder-decoder changes architecture family, not parameter/compute scaling ratio.
- More heads increases attention compute roughly proportionally, not decoupled from parameter growth.

</details>

---

**Q4.** An LLM must handle inference sequences significantly longer than its training context length, without any fine-tuning, while maintaining stable performance at long distances. Which positional encoding approach is best suited?

A) Absolute learned positional embeddings
B) Sinusoidal absolute positional encoding truncated at max length
C) One-hot positional encoding concatenated to embeddings
D) ALiBi (Attention with Linear Biases)

<details>
<summary>Answer</summary>

**Correct: D**

ALiBi adds a static, distance-proportional linear bias directly to attention scores instead of learning position embeddings, and empirically extrapolates well to lengths far beyond training without retraining.

**Why the others are wrong:**
- Learned absolute embeddings are fixed to a table sized to max training length and cannot represent unseen longer positions.
- Sinusoidal absolute encoding degrades significantly at test-time lengths not seen during training; ALiBi outperforms it for extrapolation and is the standard exam answer.
- One-hot concatenation is not a real production positional encoding scheme.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** You need the absolute minimum KV cache footprint for a high-throughput inference server and can tolerate a modest quality reduction. Which attention variant should you choose?

A) Multi-head attention (MHA)
B) Grouped-query attention (GQA)
C) Cross-attention
D) Multi-query attention (MQA)

<details>
<summary>Answer</summary>

**Correct: D**

MQA shares a single KV head across all query heads, producing the smallest possible KV cache and highest inference throughput, at the cost of some quality versus GQA/MHA.

**Why the others are wrong:**
- MHA has the largest KV cache (one KV head per query head).
- GQA is a middle ground — smaller cache than MHA but larger than MQA since it still uses multiple grouped KV heads.
- Cross-attention is an encoder-decoder mechanism, not a KV-cache optimization technique.

</details>

---

**Q6.** A decoder-only model has 32 query heads and uses GQA with a group size of 4 (4 query heads share each KV head). Compared to standard MHA with 32 KV heads, what is the resulting KV cache memory per token?

A) The same size as MHA
B) Reduced to 1/4 of MHA's size
C) Reduced to 1/8 of MHA's size
D) Reduced to 1/32 of MHA's size (equivalent to MQA)

<details>
<summary>Answer</summary>

**Correct: B**

A group size of 4 means 32/4 = 8 KV heads total, so the KV cache is 8/32 = 1/4 the size of full MHA's 32 KV heads.

**Why the others are wrong:**
- Ignores that GQA reduces the number of distinct KV heads.
- 1/8 would require a group size of 8 (4 KV heads), not 4.
- 1/32 (single KV head) describes MQA, not GQA with group size 4.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** FlashAttention achieves its speed and memory improvements primarily by which mechanism?

A) Approximating the softmax function to skip low-probability tokens
B) Tiling Q, K, V into blocks processed in fast on-chip SRAM to avoid materializing the full attention matrix in HBM
C) Reducing the number of attention heads at inference time
D) Quantizing attention weights to INT8 during the forward pass

<details>
<summary>Answer</summary>

**Correct: B**

FlashAttention's core innovation is an IO-aware, exact algorithm that tiles inputs into SRAM-sized blocks and uses online softmax recomputation, avoiding writing the O(N^2) attention matrix to slower HBM — a memory-bandwidth optimization, not an approximation.

**Why the others are wrong:**
- FlashAttention is mathematically exact, not an approximation of softmax.
- It doesn't change head count; it's an implementation-level kernel fusion.
- Quantization is a separate optimization technique, not what FlashAttention does by default.

</details>

---

**Q8.** Which statement correctly describes a key architectural trade-off between encoder-decoder and decoder-only transformer models?

A) Encoder-decoder models cannot use attention at all, while decoder-only models rely exclusively on attention
B) Decoder-only models always require twice the parameters of encoder-decoder models for equivalent performance
C) Encoder-decoder models use bidirectional self-attention over the input plus cross-attention in the decoder, making them well suited to seq2seq tasks, while decoder-only models use unified causal attention, simplifying scaling for open-ended generation
D) Encoder-decoder architectures cannot be pretrained with self-supervised objectives

<details>
<summary>Answer</summary>

**Correct: C**

This reflects why models like T5/BART (encoder-decoder) excel at translation/summarization via bidirectional encoding and cross-attention, while decoder-only models like GPT/Llama use one causal-attention stack, which scales more simply and dominates modern general-purpose LLMs.

**Why the others are wrong:**
- Both architectures fundamentally rely on attention.
- There is no fixed 2x parameter rule; needs depend on task and design, not architecture family alone.
- Encoder-decoder models (e.g., T5's span corruption) are commonly pretrained with self-supervised objectives.

</details>

---

