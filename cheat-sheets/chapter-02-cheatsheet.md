## Chapter 2 Cheat Sheet: Scaling Laws, Tokenization, Context-Window Mechanics

### Chinchilla Compute-Optimal Scaling
| Concept | Rule of Thumb |
|---|---|
| Tokens : Parameters ratio | ~20 : 1 at compute-optimal point (Hoffmann et al., 2022) |
| Compute budget | Roughly split between scaling params and scaling tokens equally |
| 'Inference-optimal' training | Overtrain beyond 20:1 (e.g., 100–200:1) to shrink model size and cut long-term inference cost — NOT the same as compute-optimal |

**Exam trap:** LLaMA-style models trained on 1T+ tokens for 7B params are intentionally *beyond* Chinchilla-optimal, not an example of it.

### Tokenization: BPE vs SentencePiece
| Feature | BPE (classic, e.g., GPT-2) | SentencePiece |
|---|---|---|
| Pre-tokenization | Assumes whitespace splitting first | None — operates on raw byte/char stream |
| Whitespace handling | Treated as boundary | Treated as a normal symbol |
| Best for | Whitespace-delimited languages (English) | CJK / multilingual, no word boundaries |
| Modes | Merge-based only | BPE or Unigram LM mode |

### Vocabulary Size Trade-offs
| Larger Vocab | Smaller Vocab |
|---|---|
| Bigger embedding table (params ≈ vocab × hidden_dim) | Smaller embedding table |
| Shorter sequences per text | Longer sequences per text |
| More rare/undertrained tokens | Fewer distinct rare tokens, more subword splitting |

### Context-Window Extension Techniques
| Technique | Fine-tuning needed? | Mechanism |
|---|---|---|
| Position Interpolation (PI) | Yes | Linearly compresses position indices into original trained range |
| NTK-aware RoPE scaling | Often no ('training-free') | Non-uniform frequency scaling — less compression on high freq, more on low freq |
| Increase RoPE base theta | Sometimes minimal | Slows frequency decay (e.g., 10000 → 500000) for longer effective range |

### Dense vs MoE Model Selection
| Factor | Dense | MoE |
|---|---|---|
| Active params/token | 100% of model | Subset (e.g., 2 of 8 experts) |
| Inference cost | Scales with total size | Scales with active experts only |
| VRAM footprint | Lower total memory | Higher — must hold all experts |
| Routing complexity | None | Requires load-balanced routing |
| Example | Standard dense LLM | Mixtral 8x7B (~47B total, ~13B active) |

### Context Length Cost/Latency
- KV cache memory: grows **linearly** with context length.
- Raw attention compute: grows **quadratically** without optimization (FlashAttention, sliding window mitigate this).
- Longer context ⇒ higher prefill latency + more memory pressure ⇒ direct cost impact at serving time.
