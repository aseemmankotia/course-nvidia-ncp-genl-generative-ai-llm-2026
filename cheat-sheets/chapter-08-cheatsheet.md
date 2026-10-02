## Chapter 8 Cheat Sheet: Inference Acceleration

### Quantization Method Comparison
| Method | Precision | Calibration Needed | Accuracy Impact | Memory Savings | Best Use |
|---|---|---|---|---|---|
| INT8 (naive) | 8-bit | No | High risk if uncalibrated | ~2x | Quick test only |
| INT8 (calibrated) | 8-bit | Yes | Low-moderate | ~2x | General deployment |
| FP8 | 8-bit float | Minimal | Very low (near FP16) | ~2x | H100/Hopper+ throughput |
| AWQ | INT4 | Small activation set | Low | ~4x | Large models, limited memory |
| GPTQ | INT4/INT3 | Calibration dataset | Low-moderate | ~4-5x | Max compression w/ controlled error |

### Batching Strategies
| Strategy | Mechanism | Throughput | Latency Variance |
|---|---|---|---|
| Static batching | Fixed batch, padded to max length | Lower (padding waste) | High (blocks on longest seq) |
| Continuous (in-flight) batching | Per-iteration admit/evict | Higher | Lower, smoother |

### KV-Cache Management
- **Paged KV-cache**: non-contiguous block allocation (virtual-memory style) → reduces fragmentation, supports variable/long context.
- **Contiguous pre-allocation**: simpler but wastes memory on short sequences and limits concurrency.

### Speculative Decoding
- Draft model proposes N tokens → Target model verifies in one forward pass.
- Latency win scales with draft acceptance rate; low acceptance rate reduces benefit.
- Requires KV-cache in both draft and target models.

### TensorRT-LLM Engine Build
- Steps: checkpoint → graph optimization (kernel/layer fusion) → precision calibration → engine compiled for target GPU.
- Engine is hardware-specific (not portable across GPU architectures).
- Does NOT retrain weights.

### Choosing Optimization by SLA Target
| Target | Primary Lever |
|---|---|
| Max throughput | Continuous batching + FP8/INT4 quantization |
| Min per-request latency | Speculative decoding + paged KV-cache |
| Min memory footprint | AWQ/GPTQ INT4 |
| Max accuracy retention | FP8 (if H100+) or calibrated INT8 |

### Exam Gotchas
1. FP8 requires Hopper (H100) or newer — not available on all GPUs.
2. Static batching throughput looks fine on paper but fails badly with variable-length real traffic.
3. Uncalibrated INT8 is NOT automatically safe — outlier activations cause accuracy loss.
4. Speculative decoding does not shrink the target model; it reduces number of target forward passes.
5. Paged KV-cache solves memory fragmentation, not compute bottlenecks.
