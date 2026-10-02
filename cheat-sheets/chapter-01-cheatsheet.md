# Chapter 1 Cheat Sheet: Transformer Internals

## Core Attention Math
| Concept | Formula/Fact |
|---|---|
| Scaled dot-product attention | softmax(QK^T / sqrt(d_k)) V |
| Why scale by sqrt(d_k) | Prevents large dot products saturating softmax gradients |
| Multi-head attention | head_dim = d_model / num_heads; heads run in parallel, outputs concatenated + projected |

## Attention Variants — KV Cache Comparison
| Variant | KV Heads | KV Cache Size | Quality | Notable Users |
|---|---|---|---|---|
| MHA | = num query heads | Largest | Highest | Original Transformer |
| GQA | Grouped (subset) | Medium (e.g., 1/4–1/8 of MHA) | Near-MHA | Llama 2/3, Mistral |
| MQA | 1 (shared by all) | Smallest | Lowest of the three | PaLM, Falcon |

**KV cache formula:** cache size ∝ num_layers × num_kv_heads × head_dim × 2 (K and V) × bytes/param. GQA group size g → KV heads = num_query_heads / g.

## FlashAttention
- Exact algorithm (no approximation), IO-aware.
- Tiles Q/K/V into SRAM-sized blocks + online softmax to avoid writing O(N^2) matrix to HBM.
- Memory: O(N) vs standard O(N^2). FLOPs remain O(N^2).
- Fixes memory-bandwidth bottleneck, not compute bottleneck.

## Mixture-of-Experts (MoE)
| Concept | Fact |
|---|---|
| Routing | Gating network scores experts; top-k (commonly 1 or 2) activated per token |
| Sparse activation | Total params scale with expert count; active FLOPs scale with k, not total experts |
| Example | Mixtral 8x7B: top-2 of 8 experts, ~13B active / ~47B total params |

## Positional Encodings
| Type | Mechanism | Extrapolation | Used In |
|---|---|---|---|
| Absolute (sinusoidal/learned) | Fixed/learned per-position vector added to embedding | Poor | Original Transformer, GPT-2 |
| RoPE | Rotates Q/K vectors by position-dependent angle | Needs scaling (NTK/YaRN) | Llama, GPT-NeoX, PaLM |
| ALiBi | Linear distance-based bias added to attention scores | Strong (native) | BLOOM, MPT |

## Encoder-Decoder vs Decoder-Only
| Aspect | Encoder-Decoder | Decoder-Only |
|---|---|---|
| Attention | Bidirectional encoder + causal decoder + cross-attention | Unified causal attention only |
| Best fit | Translation, summarization (seq2seq) | Open-ended generation, general LLMs |
| Examples | T5, BART | GPT, Llama, Mistral |

## Exam Traps
- MoE sparsity ≠ attention sparsity/memory savings (different mechanisms).
- FlashAttention is exact — not an approximation.
- RoPE alone does NOT extrapolate well without additional scaling; ALiBi does natively.
- GQA KV cache reduction = num_query_heads / group_size, not group_size itself.
