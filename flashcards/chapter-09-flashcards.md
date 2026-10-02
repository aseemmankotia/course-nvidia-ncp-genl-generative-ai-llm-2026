## Chapter 9: Serving LLMs: NVIDIA NIM, Triton Inference Server, and Model Repositories — Flashcards

| # | Front (Question) | Back (Answer) |
|---|-----------------|---------------|
| 1 | What is NVIDIA NIM? | A prepackaged, GPU-optimized inference microservice container (often built on Triton/TensorRT-LLM) offering a standardized API for fast, low-effort LLM deployment. Exam tip: choose NIM for speed and standardization, not custom control. |
| 2 | NIM vs custom Triton — when to choose custom Triton? | When you need fine-grained control: custom ensembles, custom pre/post-processing backends, non-standard batching logic, or unsupported model architectures. Exam tip: 'fine-grained control' = custom Triton keyword. |
| 3 | Required Triton model config file name and location? | config.pbtxt, placed at the model's top-level directory (not inside version subfolders). Defines platform/backend, max_batch_size, inputs, outputs, instance_group. |
| 4 | Triton model repository version folder naming rule? | Must be positive integers (1, 2, 3...). Triton serves the highest-numbered version by default unless a version policy specifies otherwise. |
| 5 | Triton default server ports? | HTTP: 8000, gRPC: 8001, Metrics (Prometheus): 8002. Exam tip: metrics endpoint is key for Kubernetes HPA integration. |
| 6 | What does instance_group control in config.pbtxt? | Number and type (KIND_GPU/KIND_CPU) of model instances to run concurrently, enabling multi-instance serving on one or multiple GPUs. |
| 7 | Purpose of dynamic_batching + max_queue_delay_microseconds? | Groups concurrent inference requests into batches for higher throughput; max_queue_delay_microseconds bounds how long a request waits to be batched, controlling latency vs throughput tradeoff. |
| 8 | How does MIG enable multi-tenant GPU serving? | Multi-Instance GPU hardware-partitions a single GPU into isolated slices with dedicated compute/memory, giving hard resource isolation between tenants — unlike software-only sharing via CUDA streams. |
| 9 | Triton model control modes? | NONE (loads all models at startup, default), POLL (periodically checks repository for changes), EXPLICIT (models loaded/unloaded via API calls). Exam tip: EXPLICIT is used for controlled production rollouts. |
| 10 | Latency vs throughput vs cost — key serving lever? | Larger batch sizes and longer queue delays raise throughput and GPU utilization but increase per-request latency; smaller batches lower latency but raise per-request cost. Exam tests picking config for stated SLA. |
| 11 | Role of NVIDIA GPU Operator in Kubernetes LLM serving? | Automates deployment/management of NVIDIA drivers, device plugin, and container toolkit on K8s nodes, enabling GPU scheduling for Triton/NIM pods. |
| 12 | Ensemble model in Triton — what is it for? | Chains multiple models/backends (e.g., preprocessing → LLM → postprocessing) into a single pipeline defined in config.pbtxt, executed server-side without extra network hops. Only available via custom Triton, not standard NIM packaging. |
