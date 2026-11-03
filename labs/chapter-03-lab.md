# Chapter 3 Lab: Advanced Prompting: Chain-of-Thought, ReAct, and Constrained Decoding

**Estimated time:** 25 minutes (Part A) + 10 minutes (Part B, optional)

## Part A — offline (no key, no network)

Build a ReAct-style prompt loop calling a mock search tool and enforce JSON-schema constrained output using the provided Python harness. The "model" is a deterministic stand-in so the loop logic is visible and the lab runs anywhere.

## Part B — bring your own key (optional)

Run the same techniques against a live hosted model through the OpenAI-compatible chat API — NVIDIA NIM (free key from build.nvidia.com; default `meta/llama-3.1-8b-instruct`) or OpenAI. Standard library only, no SDK. Three comparisons:

1. **Zero-shot vs one-shot vs few-shot** on a label-only classification task — see the output format and borderline cases change with each example added.
2. **Direct vs chain-of-thought** on a multi-step cost question — see the accuracy/token trade-off.
3. **ReAct with the live model as the policy** — the Part A loop, unchanged, driven by a real model that emits JSON actions.

If no key is set, Part B cells print a notice and skip; Part A still runs.

## Key takeaway

Prompting techniques each solve a different failure mode—reasoning depth, tool access, or output format—and picking the wrong one wastes tokens without fixing the actual problem.
