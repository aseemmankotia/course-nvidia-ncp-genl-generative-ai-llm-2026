## Chapter 7: Distributed Training: Data, Tensor, Pipeline, and Sequence Parallelism — Practice Questions

**Q1.** A team is training a dense transformer whose feed-forward layer weight matrices are too large to fit in a single GPU's memory, even with a micro-batch size of 1. All 8 GPUs are in one DGX node connected via NVLink. Which parallelism strategy should be applied FIRST to fix this?

A) Data parallelism, to reduce the effective batch size per GPU
B) Pipeline parallelism, to place each layer on a different GPU
C) Tensor parallelism, to shard the layer's weight matrices across GPUs
D) Increase gradient checkpointing frequency

<details>
<summary>Answer</summary>

**Correct: C**

Tensor parallelism splits individual weight matrices (e.g., attention and FFN projections) across GPUs, directly solving a single-layer-too-large problem. It requires the high-bandwidth, low-latency NVLink links found within a node.

**Why the others are wrong:**
- Data parallelism replicates the full model on every GPU, so it doesn't help when a single layer doesn't fit.
- Pipeline parallelism shards layers across GPUs but each GPU still needs to hold a full layer's weights, so an oversized single layer remains a problem.
- Gradient checkpointing reduces activation memory, not parameter memory of a single oversized layer.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** A model is being fine-tuned with a 64K-token context length. Activation memory dominates GPU memory usage even though the model's weights fit comfortably. Which technique specifically targets this activation memory bottleneck for long sequences when combined with tensor parallelism?

A) Sequence parallelism
B) Pipeline parallelism
C) ZeRO stage 3
D) FP8 mixed precision

<details>
<summary>Answer</summary>

**Correct: A**

Sequence parallelism shards the activations of LayerNorm/dropout regions (previously replicated in tensor-parallel-only setups) along the sequence dimension, cutting activation memory that scales with sequence length.

**Why the others are wrong:**
- Pipeline parallelism reduces per-GPU layer count, not per-token activation memory within a layer.
- ZeRO stage 3 shards optimizer states, gradients, and parameters — not activations.
- FP8 reduces numerical precision to save compute/memory on tensors generally but is not the targeted technique for the sequence-length-driven activation blowup.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** A 96-layer model is being trained across 4 nodes connected by a comparatively slow Ethernet fabric (no NVLink between nodes). The team wants to scale training across nodes without incurring heavy all-reduce traffic per layer. Which strategy best fits this constraint?

A) Tensor parallelism across all 4 nodes
B) Sequence parallelism across all 4 nodes
C) ZeRO stage 3 with no other parallelism
D) Pipeline parallelism, splitting layers across nodes

<details>
<summary>Answer</summary>

**Correct: D**

Pipeline parallelism only requires sending activations between adjacent stages at layer boundaries, which is far less communication-intensive than the frequent all-reduces tensor or sequence parallelism require, making it suitable for lower-bandwidth inter-node links.

**Why the others are wrong:**
- Tensor parallelism requires two all-reduces per layer per forward/backward pass, which is prohibitively expensive over slow inter-node links.
- Sequence parallelism is paired with tensor parallelism and inherits the same intra-layer communication overhead.
- ZeRO stage 3 alone still requires all-gather/reduce-scatter communication for every parameter shard, which is also bandwidth-heavy across nodes without a complementary strategy to reduce cross-node traffic.

</details>

---

**Q4.** A team already uses ZeRO stage 2 and is still running out of memory when training a 70B parameter model on GPUs with limited memory. Moving to ZeRO stage 3 will most likely have which effect?

A) It will only affect optimizer states, providing no additional memory savings
B) It will double GPU memory usage due to redundant parameter copies
C) It will reduce memory further by also partitioning parameters, at the cost of additional communication for parameter gathering
D) It eliminates the need for data parallelism entirely

<details>
<summary>Answer</summary>

**Correct: C**

ZeRO stage 3 adds parameter partitioning on top of stage 2's optimizer state and gradient partitioning, giving memory reduction that scales roughly linearly with the number of data-parallel GPUs, but requires extra all-gather communication to reconstruct full parameters during forward/backward.

**Why the others are wrong:**
- Stage 3 partitions optimizer states, gradients, AND parameters — it goes beyond stage 2, not equal to it.
- Stage 3 reduces, not doubles, memory usage since parameters are sharded rather than replicated.
- ZeRO is implemented as an enhancement to data-parallel training, not a replacement for it.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** In standard mixed-precision Adam training without any ZeRO sharding, approximately how many bytes of memory are consumed per model parameter (fp16 weights + fp16 gradients + fp32 master weights + fp32 momentum + fp32 variance)?

A) 4 bytes
B) 8 bytes
C) 16 bytes
D) 32 bytes

<details>
<summary>Answer</summary>

**Correct: C**

The standard breakdown is 2 bytes (fp16 weight) + 2 bytes (fp16 gradient) + 4 bytes (fp32 master weight) + 4 bytes (fp32 momentum) + 4 bytes (fp32 variance) = 16 bytes per parameter, the baseline figure ZeRO stages reduce.

**Why the others are wrong:**
- 4 bytes only covers a single fp32 value, far less than the full optimizer state footprint.
- 8 bytes undercounts the required fp32 master weight and Adam moment buffers.
- 32 bytes overstates the requirement; it would imply double-counting some buffers.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q6.** Why is BF16 generally preferred over FP16 for pretraining large transformer models on modern GPUs?

A) BF16 has more mantissa bits, giving higher precision than FP16
B) BF16 is only supported on CPUs, making it more stable
C) BF16 shares FP32's 8-bit exponent range, avoiding overflow/underflow without requiring loss scaling
D) BF16 uses 32 bits per value, giving it more range than FP16

<details>
<summary>Answer</summary>

**Correct: C**

BF16 keeps the same 8-bit exponent as FP32, giving it the same dynamic range, which prevents the gradient overflow/underflow issues common in FP16 (max ~65504) and removes the need for dynamic loss scaling.

**Why the others are wrong:**
- BF16 actually has fewer mantissa bits (7) than FP16 (10), so it has lower precision, not higher.
- BF16 is a 16-bit format supported on GPU tensor cores (Ampere and later), not a CPU-only format.
- BF16 is 16 bits total, the same as FP16 — its wider range comes from exponent bit allocation, not total bit width.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** In a GPipe-style pipeline-parallel schedule with 4 pipeline stages, increasing the number of micro-batches per global batch from 4 to 32 has which effect on training efficiency?

A) It decreases the pipeline bubble fraction, improving GPU utilization
B) It increases the pipeline bubble fraction, making training slower
C) It has no effect on bubble overhead, only on memory
D) It eliminates the need for activation recomputation

<details>
<summary>Answer</summary>

**Correct: A**

Bubble fraction is approximately (p-1)/m, where p is the number of pipeline stages and m is the number of micro-batches. Increasing m from 4 to 32 shrinks the bubble fraction significantly, improving pipeline utilization.

**Why the others are wrong:**
- This is the opposite effect; more micro-batches reduce, not increase, the bubble fraction.
- Bubble overhead is directly a function of micro-batch count, so it is affected, not memory-only.
- Micro-batch count and activation recomputation address different problems; increasing micro-batches does not remove the need for checkpointing.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** A team enables gradient checkpointing (activation recomputation) on a very deep transformer to fit a larger batch size in memory. What trade-off should they expect?

A) Reduced activation memory, at the cost of roughly 30-40% additional compute from recomputing forward activations during the backward pass
B) Reduced compute time but higher memory usage
C) No change in memory or compute, only in numerical precision
D) Elimination of the need for any parallelism strategy

<details>
<summary>Answer</summary>

**Correct: A**

Gradient checkpointing discards intermediate activations and recomputes them during the backward pass, cutting activation memory substantially (often to roughly sqrt(n) of layers stored) in exchange for about 30-40% extra forward compute.

**Why the others are wrong:**
- It reduces memory, but compute time increases due to recomputation, not decreases.
- It directly changes memory usage and compute time; precision is unaffected.
- Checkpointing addresses activation memory only — it does not replace tensor, pipeline, or data parallelism needed for model or batch scaling.

</details>

---

