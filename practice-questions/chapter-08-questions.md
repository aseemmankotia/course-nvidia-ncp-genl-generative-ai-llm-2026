## Chapter 8: Inference Acceleration: TensorRT-LLM, Quantization, and Speculative Decoding — Practice Questions

**Q1.** A team needs to deploy a 70B-parameter LLM on limited GPU memory while keeping accuracy loss minimal for a chat application. Which quantization approach BEST meets this requirement?

A) AWQ (Activation-aware Weight Quantization) to INT4
B) FP8 weight-only quantization on H100 GPUs
C) Naive INT8 post-training quantization with no calibration
D) Keep FP16 and reduce batch size to fit memory

<details>
<summary>Answer</summary>

**Correct: A**

AWQ protects salient weight channels identified via activation statistics, giving near-FP16 accuracy at INT4 memory footprint, ideal for large models on constrained memory.

**Why the others are wrong:**
- FP8 weight-only still needs newer hardware support and gives less memory reduction than INT4 AWQ
- Naive INT8 without calibration causes significant accuracy degradation, especially on outlier activations
- Reducing batch size lowers throughput and doesn't solve the memory-per-weight problem

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** Which quantization method relies on layer-by-layer weight reconstruction using a small calibration dataset and second-order (Hessian) error minimization?

A) AWQ
B) FP8 dynamic quantization
C) GPTQ
D) Bitsandbytes INT8 mixed-precision

<details>
<summary>Answer</summary>

**Correct: C**

GPTQ uses one-shot layer-wise quantization minimizing reconstruction error via approximate Hessian information, typically at INT4/INT3.

**Why the others are wrong:**
- AWQ scales weights based on activation magnitude, not Hessian-based reconstruction
- FP8 dynamic quantization scales per-tensor/per-channel at runtime, no calibration dataset needed
- Bitsandbytes INT8 mixed-precision uses outlier decomposition, not Hessian-based layer reconstruction

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** A production inference service must maximize GPU throughput for a chat workload with highly variable prompt and response lengths. Which serving strategy BEST achieves this?

A) Static batching with fixed batch size and padding to max sequence length
B) Single-request serving with no batching to guarantee lowest latency
C) Continuous (in-flight) batching that admits and evicts requests per iteration
D) Batch size 1 with maximum KV-cache pre-allocation per request

<details>
<summary>Answer</summary>

**Correct: C**

Continuous batching adds new requests as soon as GPU slots free up (per-token/per-iteration scheduling), avoiding the padding waste and head-of-line blocking seen with static batching, maximizing throughput on variable-length workloads.

**Why the others are wrong:**
- Static batching wastes compute on padding for short sequences and blocks until the whole batch finishes
- No batching guarantees low latency for one request but drastically underutilizes GPU throughput
- Pre-allocating max KV-cache per request wastes memory and reduces achievable concurrency

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** In TensorRT-LLM, what is the primary purpose of building an 'engine' from a model checkpoint before deployment?

A) To apply hardware-specific graph optimizations, kernel fusion, and precision calibration compiled for a target GPU
B) To convert the model to ONNX format only, for cross-framework portability
C) To retrain the model weights using a new dataset for domain adaptation
D) To compress the model checkpoint file size only, without changing execution graph

<details>
<summary>Answer</summary>

**Correct: A**

The TensorRT-LLM build step performs graph-level optimizations (layer/kernel fusion, precision calibration, memory planning) and compiles an engine tuned for the specific target GPU architecture.

**Why the others are wrong:**
- ONNX conversion is a separate/optional intermediate step, not the engine build's main purpose
- Engine building does not retrain weights; it optimizes execution graph for existing weights
- File size reduction is a side effect of quantization, not the primary purpose of engine building

</details>

---

**Q5.** Speculative decoding uses a smaller 'draft' model to propose multiple tokens which the larger target model then verifies in parallel. What is the main latency benefit of this technique?

A) It eliminates the need for a KV-cache in the target model
B) It permanently reduces the target model's parameter count
C) It reduces the number of large-model forward passes needed per accepted token sequence
D) It allows the target model to skip attention computation entirely

<details>
<summary>Answer</summary>

**Correct: C**

By verifying several draft-proposed tokens in a single target-model forward pass, speculative decoding amortizes the expensive large-model call across multiple tokens, cutting average per-token latency when acceptance rate is reasonably high.

**Why the others are wrong:**
- The target model still requires a KV-cache to verify tokens correctly
- The target model's parameters are unchanged; only the number of invocations is reduced
- Attention computation still occurs in the target model during verification

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q6.** A team observes that GPU memory is being exhausted by long-context requests even though compute utilization is low. Which optimization directly addresses this bottleneck?

A) Switch from FP16 to FP32 precision for higher numerical stability
B) Increase static batch size to improve GPU utilization
C) Implement paged KV-cache management to allocate memory in non-contiguous blocks on demand
D) Disable speculative decoding to save compute

<details>
<summary>Answer</summary>

**Correct: C**

Paged KV-cache (block-based, non-contiguous allocation similar to virtual memory paging) reduces fragmentation and allows memory to be allocated only as needed per sequence, directly relieving memory pressure from long contexts.

**Why the others are wrong:**
- FP32 increases memory usage, worsening the problem
- Increasing static batch size increases memory demand further and doesn't fix fragmentation
- Speculative decoding affects compute/latency, not KV-cache memory management

</details>

---

**Q7.** A company wants maximum inference throughput and is willing to accept a small accuracy drop, running on NVIDIA H100 GPUs. Which precision format is BEST suited to exploit H100 hardware acceleration while balancing accuracy?

A) FP8
B) FP32
C) INT4 with no calibration
D) BF16 with double batch padding

<details>
<summary>Answer</summary>

**Correct: A**

H100 GPUs include native FP8 Tensor Core support, delivering near-FP16 accuracy with roughly 2x throughput and reduced memory versus FP16/BF16, making it the best throughput/accuracy trade-off on this hardware.

**Why the others are wrong:**
- FP32 is the slowest and most memory-intensive, opposite of the throughput goal
- INT4 without calibration risks significant accuracy loss beyond 'small drop' tolerance
- BF16 does not leverage FP8 Tensor Core acceleration and padding adds no throughput benefit

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** Which combination of actions BEST reduces end-to-end latency for a real-time chatbot without significantly degrading response quality?

A) Use FP32 precision AND static batching with maximum sequence padding
B) Disable KV-cache reuse AND increase draft model size to match the target model
C) Use INT8 with no calibration AND process requests strictly one at a time
D) Use GPTQ INT4 quantization AND enable continuous batching with paged KV-cache

<details>
<summary>Answer</summary>

**Correct: D**

GPTQ INT4 reduces memory/compute per token while continuous batching with paged KV-cache maximizes GPU utilization and avoids padding/fragmentation waste, together minimizing latency at acceptable accuracy.

**Why the others are wrong:**
- FP32 and static padding increase compute and wasted cycles, raising latency
- Disabling KV-cache reuse forces redundant recomputation; a draft model as large as the target defeats speculative decoding's speed benefit
- Uncalibrated INT8 harms accuracy and serial processing severely limits throughput/latency under load

</details>

---

