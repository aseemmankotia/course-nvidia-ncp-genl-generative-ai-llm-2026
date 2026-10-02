## Chapter 6: Customization Strategies: LoRA/QLoRA, RLHF vs DPO, and NVIDIA NeMo — Practice Questions

**Q1.** A team must fine-tune a 70B parameter model for domain-specific instruction following using only 5,000 labeled examples, and the entire job must run on a single 80GB GPU. Which approach BEST meets these constraints?

A) Full fine-tuning of all 70B parameters
B) Continued pretraining followed by full fine-tuning
C) QLoRA (4-bit quantized base weights + LoRA adapters)
D) RLHF with a trained reward model and PPO

<details>
<summary>Answer</summary>

**Correct: C**

QLoRA quantizes the frozen base weights to 4-bit (NF4), reducing memory from ~140GB to ~35GB for a 70B model, and trains only small LoRA adapters, allowing the whole job to fit on a single 80GB GPU.

**Why the others are wrong:**
- Full fine-tuning requires storing full-precision weights, gradients, and optimizer states for all 70B parameters — far more than 80GB.
- Continued pretraining requires a large unlabeled corpus, not the small labeled instruction dataset described, and doesn't solve the memory constraint alone.
- RLHF is for preference alignment, not initial domain/instruction adaptation, and is far more compute-intensive than the scenario requires.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** A team has a large set of human preference pairs (chosen vs. rejected responses) and wants the simplest, most stable training pipeline for aligning the model, without building separate reward model infrastructure. Which technique should they use?

A) DPO (Direct Preference Optimization)
B) RLHF with PPO
C) Full fine-tuning on the chosen responses only
D) P-tuning on the chosen responses

<details>
<summary>Answer</summary>

**Correct: A**

DPO directly optimizes the policy on preference pairs using a closed-form loss, eliminating the need for a separate reward model and RL loop — simpler and more stable than RLHF.

**Why the others are wrong:**
- RLHF requires training a reward model plus a PPO reinforcement learning loop, adding complexity and instability risk (reward hacking).
- Fine-tuning only on chosen responses discards the rejected-pair signal and is effectively SFT, not true preference optimization.
- P-tuning is a PEFT method for task adaptation, not a mechanism for preference-pair optimization.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** An enterprise has 50GB of unlabeled legal documents and wants the model to internalize legal vocabulary and writing style before any instruction tuning occurs. Which customization step should be performed first?

A) LoRA fine-tuning on legal QA pairs
B) Continued pretraining (domain-adaptive pretraining) on the legal corpus
C) DPO using legal preference pairs
D) P-tuning with virtual tokens

<details>
<summary>Answer</summary>

**Correct: B**

Continued pretraining on the unlabeled domain corpus using the original causal LM objective adapts vocabulary and style before downstream task tuning.

**Why the others are wrong:**
- LoRA on QA pairs requires labeled instruction data, which isn't available yet, and doesn't address raw vocabulary/style adaptation.
- DPO requires preference pairs, which don't exist in this unlabeled corpus.
- P-tuning also requires labeled task-specific data, not raw unlabeled text.

</details>

---

**Q4.** Which NVIDIA NeMo component is purpose-built for RLHF (reward model training + PPO) and DPO pipelines, distinct from general PEFT/SFT customization recipes?

A) NeMo Curator
B) NeMo Guardrails
C) NeMo Framework Launcher
D) NeMo Aligner

<details>
<summary>Answer</summary>

**Correct: D**

NeMo Aligner is the dedicated library for preference alignment: reward model training, PPO-based RLHF, and DPO.

**Why the others are wrong:**
- NeMo Curator handles data curation, filtering, and deduplication for pretraining corpora, not alignment training.
- NeMo Guardrails manages runtime safety/topic rails, unrelated to training-time alignment.
- The Framework Launcher orchestrates general training jobs including SFT/PEFT, but alignment-specific reward+PPO/DPO pipelines live in Aligner.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** Which PEFT method prepends trainable virtual token embeddings to the input sequence while keeping all base model weights frozen?

A) P-tuning
B) LoRA
C) Bottleneck adapters
D) QLoRA

<details>
<summary>Answer</summary>

**Correct: A**

P-tuning trains a small set of virtual token embeddings prepended to the input, requiring very few trainable parameters with the base model fully frozen.

**Why the others are wrong:**
- LoRA injects trainable low-rank matrices into existing weight layers rather than adding virtual tokens.
- Adapters insert small bottleneck feed-forward layers between transformer blocks, not virtual tokens.
- QLoRA is LoRA combined with 4-bit quantized base weights, not a virtual-token method.

</details>

---

**Q6.** In a NeMo PEFT training YAML configuration, which parameter selects the customization technique (e.g., lora, ptuning, adapter)?

A) trainer.precision
B) model.optim.name
C) model.peft.peft_scheme
D) exp_manager.name

<details>
<summary>Answer</summary>

**Correct: C**

model.peft.peft_scheme is the config key that selects which PEFT method (lora, ptuning, adapter, ia3) is applied during customization.

**Why the others are wrong:**
- trainer.precision sets numeric precision (bf16/fp16/32), not the PEFT method.
- model.optim.name selects the optimizer (e.g., adamw), unrelated to PEFT scheme selection.
- exp_manager.name controls experiment naming/logging/checkpoint paths, not the customization technique.

</details>

---

**Q7.** QLoRA reduces base model memory footprint primarily through which mechanism?

A) Pruning unused attention heads from the base model
B) Truncating training sequence length
C) Quantizing frozen base weights to 4-bit (NF4) with double quantization
D) Distilling the base model into a smaller architecture

<details>
<summary>Answer</summary>

**Correct: C**

QLoRA's core innovation is 4-bit NF4 quantization of the frozen base weights (plus double quantization and paged optimizers), which drastically cuts memory while LoRA adapters remain trainable in higher precision.

**Why the others are wrong:**
- Pruning attention heads is an unrelated compression technique, not part of QLoRA.
- Sequence length reduction affects activation memory but is not QLoRA's defining mechanism.
- Distillation produces an entirely smaller model architecture, a different technique from QLoRA.

</details>

---

**Q8.** What is the main conceptual difference between RLHF and DPO for preference-based LLM alignment?

A) RLHF requires labeled classification data while DPO requires unlabeled text
B) RLHF trains a separate reward model and optimizes the policy via reinforcement learning (e.g., PPO); DPO directly optimizes the policy on preference pairs using a closed-form loss, without a reward model or RL loop
C) DPO only works with encoder-only models while RLHF only works with decoder-only models
D) RLHF is used during pretraining while DPO is used only during fine-tuning

<details>
<summary>Answer</summary>

**Correct: B**

RLHF's pipeline trains an explicit reward model then applies RL (PPO) to optimize the policy against it; DPO bypasses both the reward model and RL loop with a direct closed-form preference loss.

**Why the others are wrong:**
- Both methods use the same preference pair data (chosen/rejected), not different labeled/unlabeled formats.
- Neither method is restricted by encoder-only vs decoder-only architecture.
- Both RLHF and DPO are applied at the alignment/fine-tuning stage, not pretraining.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

