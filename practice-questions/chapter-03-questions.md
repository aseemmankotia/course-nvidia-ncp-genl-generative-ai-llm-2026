## Chapter 3: Advanced Prompting: Chain-of-Thought, ReAct, and Constrained Decoding — Practice Questions

**Q1.** A team needs an LLM to output JSON that must always parse correctly against a fixed schema for a downstream automated pipeline, with zero tolerance for malformed output. Which technique BEST meets this requirement?

A) Add more few-shot examples showing the desired JSON format
B) Use constrained decoding with a grammar/schema that restricts valid output tokens
C) Use chain-of-thought prompting so the model reasons before producing JSON
D) Use self-consistency with majority voting across several JSON outputs

<details>
<summary>Answer</summary>

**Correct: B**

Constrained decoding masks invalid tokens at generation time, guaranteeing syntactic validity against the schema.

**Why the others are wrong:**
- Few-shot examples only increase the probability of correct formatting, they don't guarantee it
- CoT adds reasoning text that typically breaks strict JSON parsing unless separated, and provides no validity guarantee
- Self-consistency increases cost and still doesn't guarantee any single sample is valid JSON

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** A math word-problem system has generous latency and cost budget and needs the highest possible answer accuracy. Which approach is BEST?

A) Zero-shot prompting with temperature=0
B) ReAct with external calculator tool calls only
C) A single greedy chain-of-thought pass
D) Chain-of-thought prompting combined with self-consistency (sample multiple reasoning paths, majority vote)

<details>
<summary>Answer</summary>

**Correct: D**

Self-consistency samples multiple diverse CoT paths at temperature>0 and takes a majority vote, which measurably improves accuracy on reasoning tasks when cost/latency isn't constrained.

**Why the others are wrong:**
- Zero-shot skips reasoning steps, hurting multi-step math accuracy
- ReAct is for tasks needing external tools/information, not purely computational reasoning already solvable by the model
- A single greedy pass has no diversity to vote over, limiting accuracy gains

</details>

---

**Q3.** An agent must answer a question requiring a live stock price lookup via an API, then reason about the retrieved value before giving a final answer. Which prompting pattern fits this scenario?

A) Few-shot prompting
B) Self-consistency sampling
C) Constrained decoding
D) ReAct (interleaved reasoning and tool-calling actions)

<details>
<summary>Answer</summary>

**Correct: D**

ReAct interleaves Thought, Action (tool call), and Observation steps, letting the model incorporate external, real-time information into its reasoning.

**Why the others are wrong:**
- Few-shot only shows format examples, it can't fetch live data
- Self-consistency improves reasoning accuracy but has no tool-calling mechanism
- Constrained decoding only restricts output format/tokens, not information retrieval

</details>

---

**Q4.** What defines zero-shot prompting?

A) The prompt includes exactly one input/output example before the query
B) The prompt includes multiple examples demonstrating the desired input/output pattern
C) The prompt includes only task instructions with no examples
D) The prompt instructs the model to show step-by-step reasoning

<details>
<summary>Answer</summary>

**Correct: C**

Zero-shot means the model receives only an instruction/task description with no demonstration examples.

**Why the others are wrong:**
- That describes one-shot prompting
- That describes few-shot prompting
- That describes chain-of-thought prompting, an orthogonal technique

</details>

---

**Q5.** You need an LLM to output a single field that must strictly be one of the tokens APPROVED, REJECTED, or PENDING, with no other text allowed. Which technique is BEST suited?

A) Chain-of-thought prompting to justify the decision
B) Self-consistency sampling with majority vote over free-text answers
C) ReAct with a tool call to a classification API
D) Constrained decoding using a grammar/regex restricting output to the three allowed tokens

<details>
<summary>Answer</summary>

**Correct: D**

Constrained decoding can enforce an exact enum/regex at the token level, guaranteeing the output is one of the three valid strings.

**Why the others are wrong:**
- CoT adds explanatory text, violating the strict single-token requirement
- Self-consistency on free text still risks malformed or extra text in each sample
- ReAct is unnecessary overhead when no external tool call is actually needed for a simple field constraint

</details>

---

**Q6.** Which decoding configuration is required to get meaningful diversity across sampled reasoning paths in self-consistency?

A) Temperature=0 with greedy decoding
B) Temperature > 0 (e.g., ~0.7) with multiple independent samples, then majority vote on final answers
C) Beam search with beam width=1
D) Top_p=0 with a single generation

<details>
<summary>Answer</summary>

**Correct: B**

Self-consistency requires sampling with temperature>0 across multiple generations (commonly 5-10) to produce diverse reasoning paths before voting.

**Why the others are wrong:**
- Greedy decoding at temperature=0 produces identical outputs every time, eliminating any benefit of voting
- Beam width=1 is equivalent to greedy decoding, no diversity
- Top_p=0 forces deterministic single-token selection, again removing diversity

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** In the ReAct prompting pattern, what is the correct repeated step sequence before the model produces a Final Answer?

A) Question, then Answer only
B) Context, Query, Response
C) System message, User message, Assistant message only
D) Thought, Action, Observation (repeated as needed)

<details>
<summary>Answer</summary>

**Correct: D**

ReAct explicitly structures the trace as Thought (reasoning), Action (tool call), Observation (tool result), repeated iteratively until a Final Answer is reached.

**Why the others are wrong:**
- This omits the reasoning and tool-calling structure entirely
- This is a generic RAG-style framing, not the ReAct labeled loop
- This describes standard chat roles, not the ReAct reasoning trace

</details>

---

**Q8.** A team finds that CoT plus self-consistency triples inference cost while only marginally improving accuracy on a simple binary classification task. What is the BEST course of action?

A) Always apply self-consistency regardless of task complexity for consistency across the pipeline
B) Replace CoT with ReAct for all tasks to reduce cost
C) Use constrained decoding instead of CoT to lower cost while keeping the same accuracy gains
D) Reserve CoT plus self-consistency for tasks needing multi-step reasoning, and use a single-pass prompt for the simple classification task

<details>
<summary>Answer</summary>

**Correct: D**

The value of CoT and self-consistency scales with task reasoning complexity; simple classification tasks rarely need multi-step reasoning, so a single-pass prompt is more cost-effective with similar accuracy.

**Why the others are wrong:**
- Applying self-consistency uniformly ignores the clear cost/benefit trade-off shown by the data
- ReAct solves tool-use needs, not cost from redundant sampling, and doesn't address this scenario
- Constrained decoding controls output format/validity, not reasoning quality, so it doesn't substitute for CoT's accuracy benefit

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

