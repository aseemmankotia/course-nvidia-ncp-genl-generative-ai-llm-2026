## Chapter 11: Evaluating LLMs and Mitigating Hallucination, Bias, and Toxicity — Flashcards

| # | Front (Question) | Back (Answer) |
|---|-----------------|---------------|
| 1 | BLEU vs ROUGE — what does each emphasize? | BLEU = precision-oriented n-gram overlap (favors translation). ROUGE = recall-oriented n-gram overlap (favors summarization). Exam tip: BLEU for MT, ROUGE-L for summarization. |
| 2 | When is exact match (EM) the right metric? | Short-answer/extractive QA tasks with a single correct span. Not suitable for open-ended generation or summarization. |
| 3 | Self-consistency hallucination mitigation — how does it work? | Sample multiple reasoning chains at nonzero temperature, take the majority/most-frequent final answer. Reduces one-off reasoning hallucinations without needing retrieval. |
| 4 | Retrieval grounding for hallucination mitigation | Augment generation with retrieved supporting documents (RAG); fact-checking rail compares answer to retrieved context via embedding similarity or entailment check. |
| 5 | LLM-as-judge — three common biases to know | Position bias (favors first answer shown), verbosity bias (favors longer responses), self-enhancement bias (favors same-model-family outputs). Mitigate with randomized order and multiple judges. |
| 6 | NeMo Guardrails — the 5 rail types | Input rails, Output rails, Dialog rails, Retrieval rails, Execution rails. Exam tip: PII masking = output rail; topic control = dialog/topical rail; tool-call restriction = execution rail. |
| 7 | Topical rail vs Safety rail — what's the difference? | Topical rail = keeps conversation within allowed subject scope (e.g., no competitor talk). Safety rail = blocks jailbreaks, toxicity, and harmful/unsafe content regardless of topic. |
| 8 | Fact-checking rail — what does it verify? | Checks that generated claims are supported by retrieved grounding context, flagging unsupported (hallucinated) statements before output is returned. |
| 9 | Cohen's Kappa — inter-rater reliability scale (Landis & Koch) | 0.81–1.00 = almost perfect, 0.61–0.80 = substantial, 0.41–0.60 = moderate agreement. Exam tip: used to validate human evaluation design consistency across raters. |
| 10 | Human eval vs LLM-as-judge — key trade-off | Human eval: higher fidelity, slower, expensive, needs inter-rater reliability checks. LLM-as-judge: fast, cheap, scalable, but introduces position/verbosity/self-enhancement bias. |
| 11 | Bias/toxicity mitigation techniques for LLMs | RLHF/preference tuning, red-teaming with adversarial prompts, output toxicity classifiers, safety rails in guardrails config, curated/debiased training or fine-tuning data. |
| 12 | Data governance in LLM lifecycle — key controls | PII redaction/masking before training or logging, data retention policies, access controls on prompts/outputs, audit logging, and compliance mapping (e.g., GDPR data subject rights) across ingestion, fine-tuning, and inference stages. |
