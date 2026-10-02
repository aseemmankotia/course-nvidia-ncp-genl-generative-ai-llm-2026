## Ch11 Cheat Sheet — Evaluation & Responsible AI

### Metrics: When to Use
| Metric | Best For | Notes |
|---|---|---|
| Exact Match | Extractive QA | Binary, strict |
| ROUGE-L/ROUGE-N | Summarization | Recall-oriented n-gram overlap |
| BLEU | Translation | Precision-oriented n-gram overlap |
| Accuracy | Classification | Discrete labels |
| LLM-as-judge | Open-ended generation, faithfulness | Scalable, biased (position/verbosity/self-enhancement) |
| Human eval | Gold standard | Slow/costly, needs inter-rater reliability (Cohen's Kappa) |

### LLM-as-Judge vs Human Eval
| Factor | LLM-as-Judge | Human Eval |
|---|---|---|
| Speed | Fast | Slow |
| Cost | Low | High |
| Bias risk | Position, verbosity, self-enhancement | Fatigue, subjectivity |
| Reliability check | Multiple judges + swap order | Cohen's Kappa (>0.8 = almost perfect agreement) |

### Hallucination Detection & Mitigation
- **Retrieval grounding (RAG):** compare generated claim to retrieved context via embedding similarity/entailment.
- **Self-consistency:** sample N reasoning chains, majority vote — no retrieval needed.
- **Fact-checking rail:** automated grounding check inside NeMo Guardrails pipeline before output release.

### NeMo Guardrails — Rail Types
| Rail | Purpose | Example |
|---|---|---|
| Input | Sanitize/validate user input | Block prompt injection attempts |
| Output | Filter final response | Mask PII, block toxic text |
| Dialog/Topical | Constrain conversation scope | Refuse competitor discussion |
| Retrieval | Validate retrieved context quality | Filter irrelevant docs |
| Execution | Govern tool/API calls | Restrict function-calling actions |

### Bias & Toxicity Mitigation
- RLHF / preference fine-tuning
- Red-teaming with adversarial prompts
- Output-side toxicity classifiers (safety rail)
- Debiased/curated fine-tuning datasets

### Privacy, Compliance & Data Governance
- PII redaction at ingestion and at output (output rail)
- Access control + audit logging on prompts/responses
- Data retention & deletion policy alignment (e.g., GDPR data subject rights)
- Governance spans full lifecycle: data prep → fine-tuning → inference → logging

### Exam Traps
- Confusing ROUGE (recall, summarization) with BLEU (precision, translation)
- Assuming LLM-as-judge is bias-free — it is NOT; position/verbosity/self-enhancement bias apply
- Forgetting fact-checking rails require a grounding/retrieval source — they don't work standalone
- Mixing up topical rail (topic scope) with safety rail (harmful content/jailbreaks)
