## Chapter 2: Scaling Laws, Tokenization, and Context-Window Mechanics — Practice Questions

**Q1.** A team has a fixed training compute budget and wants to train a Chinchilla compute-optimal model. Per the scaling laws findings, what should guide their parameter-to-token allocation?

A) Maximize parameter count and train on as few tokens as possible
B) Fix parameters at a popular size (e.g., 175B) regardless of the compute budget
C) Always prioritize more training tokens over parameter count regardless of budget
D) Scale model parameters and training tokens roughly in tandem, using approximately 20 training tokens per parameter

<details>
<summary>Answer</summary>

**Correct: D**

Hoffmann et al. (2022) found that for a fixed compute budget, parameters and training tokens should be scaled roughly equally, converging on a ~20:1 tokens-to-parameters ratio at the compute-optimal point.

**Why the others are wrong:**
- Maximizing parameters while starving on tokens produces an undertrained model — the opposite of the Chinchilla finding.
- Ignoring the compute budget entirely defeats the purpose of compute-optimal allocation.
- Over-indexing purely on tokens without scaling parameters wastes compute just as badly as the reverse.

</details>

---

**Q2.** An engineering team must extend a deployed model's context window from 4K to 16K tokens without performing any additional fine-tuning, accepting some quality degradation at the longest positions. Which RoPE-based technique is MOST appropriate?

A) Position Interpolation (PI) with a fine-tuning pass
B) NTK-aware RoPE scaling (dynamic/'training-free' variant)
C) Retrain the model from scratch with a longer native context
D) Truncate all inputs to 4K and apply a sliding window with no positional change

<details>
<summary>Answer</summary>

**Correct: B**

NTK-aware scaling non-uniformly adjusts RoPE rotation frequencies, allowing reasonable zero-shot ('training-free') context extension, unlike PI which typically requires fine-tuning to recover quality.

**Why the others are wrong:**
- PI reliably recovers quality only after additional fine-tuning, which violates the stated constraint.
- Retraining from scratch is far more resource-intensive than the goal requires.
- A sliding window with no positional adjustment does not actually extend the effective context length.

</details>

---

**Q3.** A company is building a multilingual LLM that must handle Chinese, Japanese, and English text without relying on whitespace-based pre-tokenization. Which tokenization approach BEST fits this requirement?

A) SentencePiece (unigram or BPE mode) operating directly on raw Unicode/byte streams
B) A whitespace-based BPE tokenizer trained only on English corpora
C) A rule-based English stemmer manually extended to other languages
D) A fixed word-level dictionary of the top 50,000 English words

<details>
<summary>Answer</summary>

**Correct: A**

SentencePiece treats input text (including whitespace) as a raw stream and does not require a whitespace-splitting pre-tokenization step, making it well-suited to CJK languages that lack clear word boundaries.

**Why the others are wrong:**
- Whitespace-based BPE trained only on English fails on CJK text which lacks whitespace word boundaries.
- Manual rule-based stemmers do not generalize to unrelated language families like Chinese/Japanese.
- A fixed English word dictionary cannot represent non-English scripts at all.

</details>

---

**Q4.** A workload needs very high model capacity to boost quality across diverse tasks, but inference latency and cost per token must remain close to a much smaller dense model. Which architecture family BEST satisfies this trade-off?

A) A larger dense model with proportionally increased active parameters
B) A smaller dense model with a reduced vocabulary size
C) An ensemble of several independent dense models averaged at inference
D) A Mixture-of-Experts (MoE) model where only a subset of experts activate per token

<details>
<summary>Answer</summary>

**Correct: D**

MoE architectures route each token through only a subset of experts, so total capacity (and total parameter count) can be very large while per-token compute stays close to that of a much smaller dense model.

**Why the others are wrong:**
- A larger dense model activates all its parameters on every token, directly increasing per-token cost.
- Reducing vocabulary size does not increase model capacity and is unrelated to this trade-off.
- An ensemble multiplies inference cost by the number of member models, the opposite of the goal.

</details>

---

**Q5.** A team is choosing between a 32K and a 128K token vocabulary for a new model. Increasing vocabulary size to 128K will primarily cause which effect?

A) A smaller embedding table and faster inference
B) A larger embedding table (more parameters) but shorter average sequence lengths for the same text
C) No change in parameter count since embeddings are shared across layers
D) Elimination of all rare or out-of-vocabulary tokens permanently

<details>
<summary>Answer</summary>

**Correct: B**

Embedding table size scales as vocab_size × hidden_dim, so a larger vocabulary directly increases parameter count; more common words/subwords become single tokens, shortening sequences for the same text.

**Why the others are wrong:**
- The effect is reversed — a larger vocabulary increases, not decreases, embedding table size.
- Embedding tables scale directly with vocabulary size, so parameter count does change substantially.
- Rare or novel tokens can still occur and remain undertrained even with a very large vocabulary.

</details>

---

**Q6.** Which RoPE configuration change is commonly used to extend a model's usable context length by making rotation frequencies decay more slowly across dimensions, without altering the interpolation formula itself?

A) Decreasing the RoPE base frequency (theta) from 10000 to a smaller value
B) Removing rotary embeddings entirely and using absolute positional embeddings
C) Reducing the number of attention heads
D) Increasing the RoPE base frequency (theta), e.g., from 10000 to a much larger value such as 500000

<details>
<summary>Answer</summary>

**Correct: D**

Increasing theta slows the rotation rate across frequency bands, letting the model represent longer relative distances before wrapping — a technique used in models such as Llama 3 and Code Llama to support longer context.

**Why the others are wrong:**
- Decreasing theta speeds up rotation decay, worsening long-range extrapolation rather than helping it.
- Dropping RoPE for absolute embeddings removes the relative-position properties that make RoPE effective for extension.
- Attention head count is unrelated to positional encoding frequency and does not affect context extension.

</details>

---

**Q7.** A model has 7 billion parameters. Following Chinchilla compute-optimal guidance (~20 tokens per parameter), approximately how many training tokens should it see to be compute-optimal?

A) ~140 billion tokens
B) ~7 billion tokens
C) ~70 billion tokens
D) ~1.4 trillion tokens

<details>
<summary>Answer</summary>

**Correct: A**

7 billion parameters × ~20 tokens/parameter ≈ 140 billion tokens, matching the Chinchilla compute-optimal ratio.

**Why the others are wrong:**
- This reflects roughly a 1:1 ratio, far below the compute-optimal point and would leave the model badly undertrained.
- This is only a ~10:1 ratio, still below the ~20:1 compute-optimal target.
- This corresponds to a ~200:1 ratio, which is intentionally beyond compute-optimal (an 'inference-optimized' overtraining regime used by some LLaMA-style models), not the Chinchilla-optimal point itself.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** What is the key structural difference between classic Byte-Pair Encoding (BPE) as used in GPT-2 and SentencePiece as used in many multilingual/LLM tokenizers?

A) BPE can only produce single-character tokens, while SentencePiece cannot
B) SentencePiece requires a pre-tokenization step that splits on whitespace, while BPE does not
C) They produce mathematically identical vocabularies for any given corpus
D) BPE typically assumes whitespace-delimited pre-tokenization, while SentencePiece treats input as a raw character/byte stream and handles whitespace as a regular symbol

<details>
<summary>Answer</summary>

**Correct: D**

Classic BPE implementations pre-split text on whitespace before merging byte/character pairs, whereas SentencePiece operates directly on the raw input stream, treating spaces as ordinary symbols — this is why it handles languages without whitespace boundaries more naturally.

**Why the others are wrong:**
- BPE produces multi-character subword and whole-word tokens after merges, not just single characters.
- This describes the relationship backwards — it is BPE, not SentencePiece, that traditionally relies on whitespace pre-tokenization.
- Vocabularies differ significantly because the two algorithms use different segmentation strategies and whitespace handling.

</details>

---

