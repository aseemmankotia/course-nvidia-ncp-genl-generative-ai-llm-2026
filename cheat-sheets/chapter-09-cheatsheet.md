## Chapter 9 Cheat Sheet: NIM, Triton, Model Repositories

### NIM vs Custom Triton — Decision Table
| Criterion | NVIDIA NIM | Custom Triton |
|---|---|---|
| Setup speed | Minutes, prepackaged container | Hours/days, manual config |
| API | Standardized (OpenAI-compatible) | Custom-defined |
| Control | Low — fixed pipeline | High — full backend/ensemble control |
| Best for | Standard model deployment fast | Custom pre/post-processing, ensembles, unsupported architectures |
| Underlying engine | Often TensorRT-LLM/Triton internally | You configure the engine/backend directly |

### Triton Model Repository Layout
```
model_repository/
└── model_name/
    ├── config.pbtxt        # required config
    ├── 1/                  # version dir (integer only)
    │   └── model.plan (or .onnx, .pt, etc.)
    └── 2/
        └── model.plan
```
- Version dirs: **integers only** (1, 2, 3...)
- Default policy: serves **latest (highest) version**
- config.pbtxt sits at the **model_name/** level, not inside version folders

### config.pbtxt Key Fields
| Field | Purpose |
|---|---|
| `name` | Model identifier |
| `platform`/`backend` | e.g., tensorrt_llm, python, onnxruntime |
| `max_batch_size` | Upper bound for dynamic batching |
| `input`/`output` | Tensor names, dtypes, dims |
| `instance_group` | count + KIND_GPU/KIND_CPU placement |
| `dynamic_batching` | Enables request batching |
| `max_queue_delay_microseconds` | Latency bound while waiting to batch |

### Triton Ports
| Port | Protocol |
|---|---|
| 8000 | HTTP |
| 8001 | gRPC |
| 8002 | Metrics (Prometheus) |

### Model Control Modes
| Mode | Behavior |
|---|---|
| NONE (default) | Loads all models at startup |
| POLL | Periodically scans repo for changes |
| EXPLICIT | Load/unload via API — used for controlled rollouts |

### Multi-Tenant / Multi-Model Serving
- **CUDA streams / concurrent instance_group**: software-level sharing, no hard isolation
- **MIG (Multi-Instance GPU)**: hardware-partitioned slices, hard isolation — use for strict tenant SLAs

### Latency vs Throughput vs Cost
- ↑ batch size / ↑ queue delay → ↑ throughput, ↑ latency, ↓ cost per request
- ↓ batch size / ↓ queue delay → ↓ latency, ↓ throughput, ↑ cost per request
- Tune `max_batch_size` + `max_queue_delay_microseconds` together to hit SLA targets

### Kubernetes Orchestration
- **NVIDIA GPU Operator**: manages drivers, device plugin, container toolkit on nodes
- **Triton Helm chart**: standard deployment pattern for Triton on K8s
- **HPA with custom metrics**: scale on Triton's exported Prometheus metrics (queue time, GPU util), not just CPU

### Common Exam Traps
- Choosing NIM when the scenario explicitly needs custom ensembles/backends (wrong — use custom Triton)
- Naming version folders with strings like 'v1' or 'latest' (invalid — must be integers)
- Using CPU-based HPA for GPU-bound inference (should use GPU/queue metrics)
- Confusing CUDA stream sharing with MIG hard isolation for multi-tenant SLAs
