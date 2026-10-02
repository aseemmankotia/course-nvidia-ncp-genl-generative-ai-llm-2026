## Chapter 12: Full Practice Exam Simulation and 120-Minute Test-Taking Strategy — Practice Questions

**Q1.** A company needs its LLM-based assistant to answer questions about internal policy documents that are updated daily. Accuracy on the most recent content is critical, and retraining infrastructure is limited. Which approach BEST meets these requirements?

A) Fine-tune the base model on the updated documents every week
B) Implement RAG with a vector store that is re-indexed as documents change
C) Paste the entire document set into the prompt context on every request
D) Rely on few-shot prompting alone with static examples

<details>
<summary>Answer</summary>

**Correct: B**

RAG separates knowledge from model weights, so re-indexing updated documents keeps answers current without retraining, and only relevant chunks are retrieved per query.

**Why the others are wrong:**
- Weekly fine-tuning is costly, slow to reflect daily changes, and risks catastrophic forgetting of general capabilities.
- Pasting entire document sets wastes context window, increases latency/cost, and may exceed the model's context limit.
- Static few-shot examples cannot reflect content that changes daily.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** A team must deploy a 70B-parameter model for inference on a single GPU with 48GB of VRAM, and some accuracy loss is acceptable. Which option fits within the memory budget?

A) FP32 full precision
B) FP16 half precision
C) Train a new model from scratch at the same size
D) INT4 quantization (e.g., AWQ or GPTQ)

<details>
<summary>Answer</summary>

**Correct: D**

70B params at ~0.5 bytes/param (INT4) is roughly 35GB, which fits on a 48GB GPU with headroom for KV cache.

**Why the others are wrong:**
- FP32 requires ~280GB, far beyond a single 48GB GPU.
- FP16 requires ~140GB, still far beyond a single 48GB GPU.
- Training from scratch is not a deployment/memory-fitting decision and is far more resource-intensive than quantizing an existing model.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** A 175B-parameter model does not fit in the memory of a single GPU even with activation checkpointing, but data parallelism alone is being used across 64 GPUs. What must be added to make training feasible?

A) Only increase the global batch size
B) Tensor and/or pipeline (model) parallelism
C) Gradient accumulation without any other change
D) Mixed precision training alone

<details>
<summary>Answer</summary>

**Correct: B**

When a single model replica cannot fit on one GPU, the model itself must be split across GPUs via tensor and/or pipeline parallelism; data parallelism alone assumes each GPU holds a full model copy.

**Why the others are wrong:**
- Increasing batch size does not address the memory-fit problem of a single model replica.
- Gradient accumulation reduces effective batch memory pressure but does not split a model that doesn't fit on one GPU.
- Mixed precision reduces memory use but is typically insufficient alone for a model of this scale.

</details>

---

**Q4.** An LLM inference service using Triton/TensorRT-LLM must maximize throughput under variable-length, concurrent user requests. Which serving feature should be enabled?

A) In-flight (continuous) batching for autoregressive generation
B) Static batching with a fixed batch size
C) Disabling batching so each request is processed alone
D) A single serial request queue with no concurrency

<details>
<summary>Answer</summary>

**Correct: A**

In-flight/continuous batching allows new requests to join a batch as earlier sequences finish generating, maximizing GPU utilization for variable-length LLM outputs.

**Why the others are wrong:**
- Static batching forces all sequences in a batch to wait for the longest one to finish, wasting GPU cycles.
- Disabling batching drastically reduces throughput since GPUs are optimized for batched matrix operations.
- A single serial queue eliminates concurrency benefits entirely and increases latency under load.

</details>

---

**Q5.** A team has limited GPU memory for fine-tuning and wants to adapt a base model to a new task while keeping most of the original weights frozen. Which technique should they configure?

A) Full fine-tuning of all parameters
B) LoRA (Low-Rank Adaptation) as a PEFT method
C) Training an entirely new model from random initialization
D) Switching the entire model to FP32 precision

<details>
<summary>Answer</summary>

**Correct: B**

LoRA freezes the base weights and trains small low-rank adapter matrices, drastically reducing trainable parameters and memory footprint compared to full fine-tuning.

**Why the others are wrong:**
- Full fine-tuning updates all parameters, requiring far more GPU memory and optimizer state.
- Training from scratch discards pretrained knowledge and requires vastly more compute and data.
- Increasing precision to FP32 increases memory usage rather than reducing it, and doesn't address parameter-efficiency.

</details>

---

**Q6.** A team has extensively refined prompts with system instructions and multiple few-shot examples, but the model still inconsistently fails to produce a required structured JSON output for a narrow, stable task. What is the BEST next step?

A) Continue adding more few-shot examples to every prompt
B) Add a RAG pipeline to retrieve relevant reference documents
C) Fine-tune the model on labeled examples of the exact target format
D) Switch to a larger general-purpose model using the same prompts

<details>
<summary>Answer</summary>

**Correct: C**

For a stable, narrow formatting requirement that prompting has already failed to solve reliably, fine-tuning on labeled target-format examples directly teaches the model the desired output behavior.

**Why the others are wrong:**
- More few-shot examples have already been tried per the scenario and consume context without guaranteeing consistency.
- RAG addresses knowledge grounding, not output formatting consistency, and doesn't apply to this problem.
- A larger model with identical prompting does not guarantee format compliance and adds cost without addressing the root cause.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** Which statement BEST describes the difference between tensor parallelism and pipeline parallelism?

A) Tensor parallelism splits the training data across GPUs; pipeline parallelism splits the model weights
B) The two techniques are functionally identical and interchangeable
C) Pipeline parallelism applies only during inference; tensor parallelism applies only during training
D) Tensor parallelism splits individual layers/matrix operations across GPUs; pipeline parallelism splits different layers/stages of the model across GPUs

<details>
<summary>Answer</summary>

**Correct: D**

Tensor parallelism shards individual matrix multiplications/layers across devices, while pipeline parallelism assigns different sequential layers/stages to different devices.

**Why the others are wrong:**
- Splitting training data across devices describes data parallelism, not tensor parallelism.
- They are distinct strategies with different communication patterns and are often combined, not identical.
- Both techniques are used in training and inference contexts, not restricted to one phase.

</details>

---

**Q8.** During a 120-minute exam with 65 questions, a candidate has spent 6 minutes on one difficult scenario question without reaching certainty. Per recommended pacing strategy, what should they do?

A) Flag the question, select the best current guess, move on, and revisit if time remains
B) Keep working until fully certain of the correct answer
C) Skip the question entirely without selecting any answer
D) Submit the exam immediately to avoid running short on time

<details>
<summary>Answer</summary>

**Correct: A**

With roughly 1.5-2 minutes budgeted per question, flagging a best-guess answer and moving on preserves time for the remaining questions while allowing a final review pass if time remains.

**Why the others are wrong:**
- Continuing indefinitely risks running out of time for other answerable questions.
- Leaving a question fully blank forfeits any chance of credit if time runs out before revisiting it.
- Submitting early wastes remaining time that could be used to review flagged questions.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

