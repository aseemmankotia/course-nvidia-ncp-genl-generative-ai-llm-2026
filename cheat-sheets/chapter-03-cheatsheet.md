## Advanced Prompting Cheat Sheet

### Prompting Modes
| Mode | Examples in prompt | Use case |
|---|---|---|
| Zero-shot | 0 | Simple/well-known tasks, minimal token cost |
| One-shot | 1 | Format guidance with tight context budget |
| Few-shot | 2+ | Complex formatting/behavior demonstration |

### Chain-of-Thought (CoT)
- Adds instruction like "think step by step" or shows worked reasoning examples.
- Improves multi-step reasoning (math, logic, planning).
- Cost: more output tokens, higher latency.
- Exam note: CoT alone still relies only on the model's parametric knowledge.

### Self-Consistency
- Sample k reasoning paths (temperature > 0, e.g., 0.7; k ≈ 5-10).
- Majority-vote the final answers.
- Trade-off: accuracy ↑, cost/latency ↑ roughly linearly with k.
- Requires non-greedy decoding — temperature=0 defeats the purpose (no diversity).
- Use when: high-stakes reasoning tasks, budget allows extra inference calls.
- Skip when: simple classification/lookup tasks where single-pass suffices.

### ReAct (Reason + Act)
- Loop: **Thought** → **Action** (tool call) → **Observation** (tool result) → repeat → **Final Answer**.
- Use when the task needs external/real-time info or computation (search, calculator, DB, API).
- Differs from CoT: CoT reasons internally only; ReAct incorporates tool outputs mid-reasoning.

### Constrained Decoding
- Enforces output validity at the token level via grammar/schema/regex (JSON schema, enum, regex).
- Guarantees syntactic correctness — prompting alone (even few-shot) cannot guarantee this.
- Tools: NeMo Guardrails, Outlines, lm-format-enforcer.
- Use when: strict downstream parsing requirements (JSON APIs, enum fields, structured extraction).

### Technique Selection Quick Table
| Requirement | Best Technique |
|---|---|
| Guaranteed valid JSON/enum output | Constrained decoding |
| Max accuracy on reasoning task, budget available | CoT + self-consistency |
| Needs live/external data mid-reasoning | ReAct |
| Simple task, low latency/cost priority | Zero-shot or single-pass CoT |
| Format guidance only, tight context | One-shot / few-shot |

### Prompt Evaluation Loop
1. Define metric/rubric
2. Run on test set
3. Identify failure mode (format error → constrained decoding; reasoning error → CoT/self-consistency; missing info → ReAct)
4. Refine prompt/technique
5. Re-test

### Common Exam Traps
- Choosing self-consistency for simple lookup/classification tasks (wastes cost, minimal gain).
- Using CoT/few-shot instead of constrained decoding when strict format validity is required.
- Forgetting temperature=0 breaks self-consistency's sampling diversity.
- Confusing ReAct (tool-augmented loop) with plain CoT (internal reasoning only).
