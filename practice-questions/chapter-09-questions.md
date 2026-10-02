## Chapter 9: Serving LLMs: NVIDIA NIM, Triton Inference Server, and Model Repositories — Practice Questions

**Q1.** A team must deploy Llama 3 70B to production within two days, expose an OpenAI-compatible API, and avoid writing custom backend or batching code. Which approach BEST meets these requirements?

A) Deploy the NVIDIA NIM microservice for the model from the NGC catalog
B) Write a custom Triton Python backend with manual pre/post-processing scripts
C) Serve the raw HuggingFace checkpoint directly with a Flask wrapper
D) Build a custom TensorRT-LLM engine from scratch and hand-write the config.pbtxt

<details>
<summary>Answer</summary>

**Correct: A**

NIM ships as a prepackaged, GPU-optimized inference container with a standardized OpenAI-compatible API, letting teams deploy in minutes without engine-building or custom serving code.

**Why the others are wrong:**
- Custom Triton backends require significant engineering time and are for cases needing fine-grained control, not fast standardized deployment
- A Flask wrapper lacks batching, optimized kernels, and production-grade scaling
- Hand-building a TensorRT-LLM engine and config.pbtxt is exactly the slow, high-control path NIM is meant to avoid

</details>

---

**Q2.** An ML platform team needs a custom ensemble pipeline: a preprocessing Python model, a TensorRT-optimized LLM, and a custom reranking postprocessor, all chained with model-specific control over instance placement per GPU. Which deployment choice is correct?

A) NVIDIA NIM, since it always supports arbitrary ensemble pipelines out of the box
B) Custom Triton Inference Server deployment using an ensemble model configuration
C) A single NIM container with three replicas load-balanced by Kubernetes
D) Deploy each stage as a separate NIM container with no shared config

<details>
<summary>Answer</summary>

**Correct: B**

Custom Triton deployments give fine-grained control over multi-model ensembles, custom backends, and per-model instance_group placement — exactly what NIM's standardized packaging does not expose.

**Why the others are wrong:**
- NIM is a standardized, prepackaged single-model microservice; it does not expose ensemble pipeline configuration
- Replicating NIM containers doesn't solve chaining pre/postprocessing into one pipeline
- Running independent NIM containers loses the low-latency in-process ensemble chaining Triton provides

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** In a Triton model repository, which file is required at the model's top-level directory (not inside version subfolders) to define the model's inputs, outputs, and backend/platform?

A) model.json
B) config.pbtxt
C) manifest.yaml
D) model_config.txt

<details>
<summary>Answer</summary>

**Correct: B**

config.pbtxt is Triton's protobuf-text configuration file specifying name, platform/backend, max_batch_size, input/output tensors, and instance_group.

**Why the others are wrong:**
- model.json is not a Triton configuration format
- manifest.yaml is not used by Triton's model repository schema
- model_config.txt is not the recognized filename; Triton specifically looks for config.pbtxt

</details>

---

**Q4.** A company wants to run three different fine-tuned LLM variants on a single A100 GPU with hard resource isolation between tenants so one tenant's load spikes cannot starve another's latency SLA. Which configuration is most appropriate?

A) Run all three models as separate Triton instance_group entries with KIND_GPU count=1 and rely on CUDA streams only
B) Partition the GPU using MIG (Multi-Instance GPU) and assign one MIG slice per model/tenant
C) Deploy all three as NIM containers on the same GPU with no scheduling configuration
D) Increase max_batch_size for all three models to force fair round-robin scheduling

<details>
<summary>Answer</summary>

**Correct: B**

MIG provides hardware-level partitioning with dedicated compute and memory per slice, giving true isolation for multi-tenant serving — unlike software-level batching or stream sharing.

**Why the others are wrong:**
- CUDA streams share the same GPU resources and can still allow one tenant to starve another under load
- Deploying NIM containers without any GPU partitioning or scheduling still shares the full GPU with no isolation
- Increasing max_batch_size affects throughput/latency tradeoffs, not resource isolation between tenants

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** What is the primary purpose of NVIDIA NIM microservices?

A) To provide a low-level CUDA kernel compiler for custom model architectures
B) To offer prepackaged, GPU-optimized inference containers with standardized APIs for rapid LLM deployment
C) To replace Kubernetes as the container orchestration layer
D) To serve exclusively as a training framework for fine-tuning LLMs

<details>
<summary>Answer</summary>

**Correct: B**

NIM packages optimized inference engines (often built on Triton/TensorRT-LLM) into ready-to-deploy containers exposing standardized APIs, minimizing setup effort.

**Why the others are wrong:**
- NIM is not a kernel compiler; it consumes optimized engines, it doesn't compile them from scratch by the user
- NIM runs on top of Kubernetes, it does not replace it
- NIM is for inference/serving, not for training or fine-tuning workflows

</details>

---

**Q6.** In a Triton model repository, how must model version directories be named for Triton to recognize and serve them?

A) Numeric integer folder names such as 1, 2, 3
B) Alphabetical strings such as 'v1', 'v2', 'latest'
C) Timestamped folder names such as 2026-01-01
D) A single 'current' folder containing all versions

<details>
<summary>Answer</summary>

**Correct: A**

Triton requires version subdirectories to be positive integers; by default it serves the highest-numbered (latest) version unless a version policy overrides this.

**Why the others are wrong:**
- Non-numeric strings like 'v1' or 'latest' are not recognized as valid version identifiers by Triton
- Timestamp-based folder names are not a supported Triton version scheme
- A single 'current' folder does not match Triton's required per-version directory structure

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** A team's LLM endpoint has high per-request GPU utilization but users complain about inconsistent response times under bursty traffic. They want to increase throughput while controlling worst-case latency. Which Triton configuration adjustment addresses this directly?

A) Reduce max_batch_size to 1 to guarantee the fastest possible single-request latency
B) Disable batching entirely and scale only by adding more Kubernetes replicas
C) Increase the number of model versions loaded simultaneously in the repository
D) Enable dynamic_batching and tune max_queue_delay_microseconds to balance batching gains against latency limits

<details>
<summary>Answer</summary>

**Correct: D**

Dynamic batching groups concurrent requests to improve GPU throughput, and max_queue_delay_microseconds bounds how long requests wait to be batched, directly trading off latency vs throughput.

**Why the others are wrong:**
- Setting max_batch_size to 1 eliminates throughput gains from batching entirely, hurting overall capacity
- Disabling batching and only scaling replicas increases cost without addressing per-request batching efficiency
- Loading multiple model versions simultaneously affects memory usage and rollout strategy, not latency/throughput tuning

</details>

---

**Q8.** A platform team is orchestrating multiple Triton Inference Server pods on Kubernetes for LLM serving and needs pods to autoscale based on actual inference queue depth and GPU utilization rather than CPU usage. Which approach is correct?

A) Use the default Kubernetes Horizontal Pod Autoscaler with CPU utilization metrics only
B) Manually scale replica counts on a fixed weekly schedule regardless of load
C) Disable autoscaling and provision one dedicated GPU node per possible traffic peak
D) Export Triton's Prometheus metrics and configure HPA with custom/external metrics based on GPU and queue statistics

<details>
<summary>Answer</summary>

**Correct: D**

Triton exposes Prometheus-compatible metrics (queue time, GPU utilization) on its metrics port; feeding these into HPA via a custom metrics adapter enables scaling decisions relevant to inference workloads rather than generic CPU load.

**Why the others are wrong:**
- CPU-based HPA is a poor proxy for GPU-bound LLM inference workloads and won't reflect real bottlenecks
- Fixed schedules don't respond to actual traffic spikes and waste or starve resources
- Over-provisioning for peak traffic is costly and defeats the purpose of elastic Kubernetes orchestration

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

