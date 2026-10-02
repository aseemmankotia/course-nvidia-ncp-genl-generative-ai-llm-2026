## Chapter 12 Cheat Sheet — Full Practice Exam Strategy

### Pacing Formula
| Metric | Value |
|---|---|
| Exam length | 120 minutes |
| Question count | ~60-70 |
| Target pace | 1.5-2 min/question |
| Reserve buffer | 10-15 min for flagged review |
| Rule | If stuck >2 min: flag, best-guess, move on |

### Cross-Domain Decision Framework
| Symptom | Solution |
|---|---|
| Knowledge outdated/changes often | RAG |
| Output format/behavior inconsistent despite good prompts | Fine-tune (or LoRA/PEFT) |
| Task solvable with better instructions/examples | Prompt engineering |
| Model too large for GPU, some accuracy loss OK | Quantization (INT4/INT8/FP8) |
| Model doesn't fit on one GPU (training) | Tensor/Pipeline parallelism |
| Model fits on one GPU, need speed via more GPUs | Data parallelism |
| Variable-length concurrent inference requests | In-flight/continuous batching |

### Parallelism Quick Reference
| Type | Splits | Use When |
|---|---|---|
| Data | Training data batches | Model fits on 1 GPU |
| Tensor | Individual layer matrices | Layer too large for 1 GPU |
| Pipeline | Sequential layer groups | Whole model too large for 1 GPU |

### Quantization Quick Reference
| Precision | Bytes/param | ~70B model size |
|---|---|---|
| FP32 | 4 | ~280GB |
| FP16/BF16 | 2 | ~140GB |
| FP8 | 1 | ~70GB |
| INT4 (AWQ/GPTQ) | 0.5 | ~35GB |

### Elimination Strategy Checklist
1. Cut options naming deprecated/mismatched services.
2. Cut options solving only part of the stated requirement.
3. Cut the most complex/expensive option if a simpler valid one exists.
4. When two options seem correct, re-read constraints (latency, memory, cost, data freshness) for the deciding factor.

### Final 15-Minute Protocol
1. Review all flagged questions first.
2. Re-check any question involving numeric limits or decision-pair concepts (RAG vs fine-tune, tensor vs pipeline, quantization tradeoffs).
3. Never leave a question fully blank — a best guess beats zero credit.

### Post-Exam Review (self-practice runs)
- Log every missed question by domain and root cause (misread scenario, wrong decision framework, unknown limit/number).
- Re-drill the specific decision pair or number, not the whole domain.
