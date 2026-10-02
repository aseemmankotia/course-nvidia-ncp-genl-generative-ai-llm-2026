## Chapter 11: Evaluating LLMs and Mitigating Hallucination, Bias, and Toxicity — Practice Questions

**Q1.** A team is deploying a customer-facing support chatbot. Requirements: it must refuse to discuss competitors, resist prompt-injection/jailbreak attempts, and only state factual claims that are grounded in the company's knowledge base. Which NeMo Guardrails configuration BEST meets all these requirements?

A) Combine topical rails, safety/jailbreak-detection rails, and fact-checking rails backed by retrieval-augmented grounding
B) Configure only topical rails to keep the conversation on-topic
C) Add a single output toxicity classifier and rely on that alone
D) Rely entirely on system-prompt instructions with no rail configuration

<details>
<summary>Answer</summary>

**Correct: A**

Each requirement maps to a distinct rail category: topic control needs topical rails, jailbreak resistance needs safety rails, and factual grounding needs fact-checking rails tied to a retrieval source — all three must be layered together.

**Why the others are wrong:**
- Topical rails alone don't stop jailbreaks or hallucinated facts
- A toxicity classifier only screens harmful language, not off-topic drift or factual accuracy
- System prompts alone are easily bypassed by adversarial inputs and provide no enforcement layer

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** A team is evaluating a summarization model and wants an automated score that correlates well with human judgment of faithfulness (i.e., detects fluent but unsupported claims), without running a full human study for every model iteration. Which approach is BEST?

A) BLEU score comparing generated summary to a single reference
B) ROUGE-L score comparing generated summary to a single reference
C) LLM-as-judge with a rubric-based prompt comparing the generated summary against the source document for faithfulness
D) Exact match scoring against the reference summary

<details>
<summary>Answer</summary>

**Correct: C**

LLM-as-judge with a faithfulness rubric can reason about semantic support from source text, catching hallucinated claims that n-gram overlap metrics cannot detect.

**Why the others are wrong:**
- BLEU measures n-gram precision against a reference summary, not faithfulness to the source document, and is insensitive to hallucinated but fluent phrasing
- ROUGE-L measures overlap with a reference summary and does not verify factual grounding in the source
- Exact match is designed for short-answer QA tasks, not open-ended summarization

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** A team needs to reduce hallucination in a multi-step reasoning task where no external retrieval source is available. Which technique should they apply?

A) Increase the sampling temperature to encourage more creative reasoning
B) Reduce max_new_tokens to force shorter, terser answers
C) Add a human proofreader to check every single output before it is shown to a user
D) Self-consistency: sample multiple independent reasoning chains and select the majority answer

<details>
<summary>Answer</summary>

**Correct: D**

Self-consistency samples several reasoning paths at nonzero temperature and takes a majority/most-consistent answer, which reduces hallucinated one-off reasoning errors without needing retrieval.

**Why the others are wrong:**
- Raising temperature increases output diversity but also increases hallucination risk rather than reducing it
- Truncating output length does not address whether the reasoning itself is grounded or correct
- Manual human review for every output does not scale and is not a hallucination-mitigation technique, it is a QA process

</details>

---

**Q4.** A retrieval-augmented chatbot uses a NeMo Guardrails fact-checking rail to catch hallucinated answers before they reach the user. Which computation does this rail typically perform?

A) Embedding-based similarity between the generated answer and the retrieved supporting context
B) Raw output perplexity from the generation model
C) BLEU score of the answer against a fixed reference answer
D) A toxicity classifier score on the generated text

<details>
<summary>Answer</summary>

**Correct: A**

Fact-checking rails compare the generated response against retrieved grounding context (often via embedding similarity or an entailment/NLI check) to flag unsupported claims.

**Why the others are wrong:**
- Perplexity reflects model confidence/fluency, not factual grounding, and can be low even for hallucinated text
- BLEU requires a fixed reference answer, which is not how fact-checking rails operate against dynamic retrieved context
- A toxicity classifier addresses harmful content, not factual accuracy

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** Which metric family is BEST suited for scoring machine translation or summarization output where wording may vary but n-gram overlap with a reference is informative?

A) ROUGE/BLEU
B) Exact match
C) Classification accuracy
D) Toxicity score

<details>
<summary>Answer</summary>

**Correct: A**

ROUGE and BLEU compute n-gram overlap (recall-oriented and precision-oriented respectively) against reference text, which is the standard for summarization and translation tasks.

**Why the others are wrong:**
- Exact match is used for short, deterministic answers like QA span extraction, not free-form generation
- Accuracy applies to classification/multiple-choice tasks, not open-ended generated text
- Toxicity score measures harmful content, unrelated to translation/summarization quality

</details>

---

**Q6.** In a NeMo Guardrails config.yml, which rail type is responsible for filtering or masking PII in the model's response before it is returned to the user?

A) Input rails
B) Dialog rails
C) Execution rails
D) Output rails

<details>
<summary>Answer</summary>

**Correct: D**

Output rails inspect and transform the generated response (e.g., redacting PII, filtering toxicity) before it is delivered to the user.

**Why the others are wrong:**
- Input rails validate/sanitize the incoming user message, not the model's generated response
- Dialog rails manage conversation flow and topic transitions, not content filtering
- Execution rails govern tool/function-calling actions, not text output filtering

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** A support bot must never discuss a competitor's products, regardless of how the conversation is steered. Which NeMo Guardrails rail category enforces this restriction?

A) Safety rail
B) Execution rail
C) Topical rail
D) Fact-checking rail

<details>
<summary>Answer</summary>

**Correct: C**

Topical rails constrain the conversation to approved subjects and redirect or refuse when the dialog drifts into disallowed topics such as competitor discussion.

**Why the others are wrong:**
- Safety rails target jailbreaks, toxicity, and harmful content, not topic scope
- Execution rails govern permitted tool/API calls, unrelated to conversation topic
- Fact-checking rails verify factual grounding of claims, not topic boundaries

</details>

---

**Q8.** Which statement BEST describes a well-known bias risk when using an LLM-as-judge for evaluating model outputs?

A) Judges are only sensitive to exact string case and punctuation differences
B) Judges penalize synonyms the same way BLEU does
C) Judges exhibit position bias (favoring the first-listed answer), verbosity bias (favoring longer responses), and self-enhancement bias (favoring outputs from the same model family)
D) Judges suffer only from human rater fatigue over long sessions

<details>
<summary>Answer</summary>

**Correct: C**

Documented LLM-as-judge biases include position bias, verbosity bias, and self-enhancement bias (favoring responses stylistically similar to the judge model's own family), which must be controlled for via randomized ordering and calibration.

**Why the others are wrong:**
- Case/punctuation sensitivity is a trait of string-matching metrics like exact match, not LLM judges
- Synonym penalization is a known weakness of n-gram metrics like BLEU, not LLM judges which use semantic understanding
- Rater fatigue is a human-evaluation concern, not an LLM-as-judge bias

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

