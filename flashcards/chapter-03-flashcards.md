## Chapter 3: Advanced Prompting: Chain-of-Thought, ReAct, and Constrained Decoding — Flashcards

| # | Front (Question) | Back (Answer) |
|---|-----------------|---------------|
| 1 | Zero-shot vs few-shot prompting — key difference? | Zero-shot: instruction only, no examples. Few-shot: 2+ input/output examples included. Exam tip: few-shot improves format adherence but increases prompt length and cost. |
| 2 | What does chain-of-thought (CoT) prompting add? | Instructs the model to produce intermediate reasoning steps before the final answer. Exam tip: improves multi-step reasoning/math accuracy but increases output tokens and latency. |
| 3 | How does self-consistency work? | Sample multiple CoT paths at temperature>0 (commonly ~5-10 samples), then take a majority vote on the final answers. Exam tip: trades higher cost/latency for higher accuracy on reasoning tasks. |
| 4 | ReAct step sequence? | Thought -> Action -> Observation, repeated, ending in Final Answer. Exam tip: Action = tool/API call; Observation = the tool's returned result fed back into context. |
| 5 | When is ReAct preferred over plain CoT? | When the task needs external, real-time, or computed information (search, calculator, database, API) that the model can't produce from parametric knowledge alone. |
| 6 | What is constrained decoding? | A decoding-time mechanism that masks invalid tokens so output must conform to a grammar/schema (JSON schema, regex, enum). Exam tip: guarantees validity, unlike prompt-only instructions. |
| 7 | Constrained decoding vs prompt-based format instructions — exam distinction? | Constrained decoding enforces validity via token-level masking (guaranteed). Prompting only raises the probability of correct format (not guaranteed). Use constrained decoding when zero tolerance for malformed output is required. |
| 8 | One-shot prompting definition? | Exactly one demonstration example included before the actual query. Useful when context window is limited but format guidance is still needed. |
| 9 | Typical self-consistency sample count (k)? | Commonly 5-10 sampled reasoning paths; accuracy gains diminish while cost scales roughly linearly with k. Exam tests understanding of this trade-off, not a fixed exact value. |
| 10 | Temperature requirement for self-consistency? | Temperature must be > 0 (e.g., ~0.7) to get diverse reasoning paths. Greedy decoding (temperature=0) produces identical outputs every run, defeating self-consistency's purpose. |
| 11 | Prompt evaluation / iterative refinement loop? | Define eval metric/rubric -> run prompt on a test set -> measure accuracy and failure modes -> refine (add examples, add CoT, or add constraints) -> re-test. Exam tests picking the correct refinement for an observed failure mode. |
| 12 | Grammar/schema-constrained decoding tools referenced in NCP-GENL context? | Libraries such as NVIDIA NeMo Guardrails, Outlines, and lm-format-enforcer implement grammar-constrained decoding for structured output. Exam tip: this is a decoding-time technique, not a prompting technique. |
