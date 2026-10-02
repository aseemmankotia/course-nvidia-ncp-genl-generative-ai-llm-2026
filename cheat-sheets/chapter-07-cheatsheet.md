## Chapter 7 Cheat Sheet: Distributed Training Parallelism

### Parallelism Strategy Selection
| Strategy | Shards | Communication | Best For | Interconnect Need |
|---|---|---|---|---|
| Data Parallel | Batch (full model replicated) | All-reduce gradients (once/step) | Scaling batch size across many GPUs/nodes | Moderate (can cross nodes) |
| Tensor Parallel | Weight matrices within a layer | ~2 all-reduces/layer/pass | Single layer too large for one GPU | High-bandwidth NVLink, intra-node |
| Pipeline Parallel | Layers across devices | Activations at stage boundaries only | Very deep models, multi-node scaling | Lower bandwidth OK |
| Sequence Parallel | Activations along sequence dim | Paired with TP all-reduces | Long context, reduces activation memory | Same as TP (intra-node) |

### ZeRO Stages (mixed-precision Adam baseline = 16 bytes/param)
| Stage | Shards | Approx. Memory Reduction |
|---|---|---|
| Stage 1 (Pos) | Optimizer states | ~4x |
| Stage 2 (Pos+g) | + Gradients | ~8x |
| Stage 3 (Pos+g+p) | + Parameters | ~linear with GPU count |

Trade-off: More sharding = more memory saved but more communication (all-gather/reduce-scatter) overhead.

### Mixed Precision Comparison
| Format | Exponent/Mantissa Bits | Range | Precision | Loss Scaling? |
|---|---|---|---|---|
| FP16 | 5 / 10 | ±65504 | Higher | Required |
| BF16 | 8 / 7 | Same as FP32 | Lower | Not required |
| FP8 E4M3 | 4 / 3 | ±448 | Lowest | Dynamic per-tensor scaling required |
| FP8 E5M2 | 5 / 2 | ±57344 | Lowest | Dynamic per-tensor scaling required |

FP8 requires Hopper (H100) or Blackwell (B100/B200/GB200) Transformer Engine support.

### Pipeline Bubble Formula
Bubble fraction ≈ (p − 1) / m
- p = pipeline stages, m = micro-batches per batch
- More micro-batches → smaller bubble → better utilization

### Gradient Checkpointing
- Saves activation memory (~sqrt(n) layers stored) by recomputing forward pass during backward
- Costs ~30-40% extra compute time

### Exam Traps
- Confusing TP (intra-layer) with PP (inter-layer) — TP needs NVLink, PP tolerates slower links.
- Assuming ZeRO stage 3 always best — it adds communication overhead, only use when stage 1/2 insufficient.
- Assuming FP16 and BF16 have same range — they don't; only exponent bit width determines range, not precision.
- Forgetting sequence parallelism is paired with tensor parallelism, not a standalone replacement for it.
