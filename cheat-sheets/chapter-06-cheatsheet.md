## Chapter 6 Cheat Sheet: Customization Strategies

### Fine-Tuning Technique Comparison
| Technique | Base Weights | Trainable Params | Memory | Best For |
|---|---|---|---|---|
| Full fine-tuning | Updated | All | Highest (weights+grads+optim states) | Large domain shift, large data budget |
| LoRA | Frozen | Low-rank matrices (rank r=4-64) | Low | General PEFT baseline, moderate data |
| QLoRA | Frozen + 4-bit NF4 quantized | LoRA adapters (higher precision) | Lowest | Large models on limited GPU (e.g., 70B on 80GB) |
| Adapters | Frozen | Bottleneck FF layers inserted between blocks | Low | Modular task-specific tuning |
| P-tuning / Prompt tuning | Frozen | Virtual token embeddings only | Lowest | Small datasets, simple tasks |

### Continued Pretraining vs PEFT
| Aspect | Continued Pretraining | PEFT Fine-Tuning |
|---|---|---|
| Data | Large unlabeled domain corpus | Small labeled task dataset |
| Objective | Causal LM (next-token) | Task-specific (instruction/SFT) |
| Use case | Vocabulary/style domain adaptation | Task/behavior adaptation |
| Order | Typically performed first | Follows continued pretraining |

### RLHF vs DPO
| Aspect | RLHF | DPO |
|---|---|---|
| Reward model | Explicit, separately trained | Not required |
| Optimization | RL (PPO) rollout-based | Closed-form loss (Bradley-Terry derived) |
| Complexity/Stability | High complexity, reward-hacking risk | Simpler, more stable |
| Data | Preference pairs (chosen/rejected) | Preference pairs (chosen/rejected) |
| Pipeline stages | SFT → Reward Model → PPO | SFT → Direct preference loss |

### NVIDIA NeMo Customization Workflow
| Component | Purpose |
|---|---|
| NeMo Framework (Launcher) | Orchestrates pretraining, continued pretraining, SFT, PEFT recipes |
| PEFT schemes (`model.peft.peft_scheme`) | lora, ptuning, adapter, ia3 |
| NeMo Aligner | RLHF (reward model + PPO), DPO, SteerLM |
| NeMo Curator | Data curation/dedup/filtering for pretraining corpora |
| NeMo Guardrails | Runtime safety/topic rails (not training-time) |

### Decision Rules
- Small labeled dataset + limited GPU → **LoRA/QLoRA**
- Very small dataset, simplest task → **P-tuning**
- Domain vocabulary/style shift, large unlabeled corpus → **Continued pretraining** first
- Have preference pairs, want simplicity/stability → **DPO**
- Have preference pairs + reward model infra + tolerance for complexity → **RLHF**
- Fine-tuning huge model (65B-70B) on single GPU → **QLoRA** (4-bit NF4 base + LoRA adapters)

### Key Numbers
- LoRA rank r: typically 4–64
- QLoRA quantization: 4-bit NF4 + double quantization
- 70B model full-precision weights alone: ~140GB (fp16/bf16) vs ~35GB at 4-bit
