## Chapter 10 Cheat Sheet: Autoscaling, Rollouts & LLM MLOps Monitoring

### Rollout Strategy Decision Table
| Scenario | Use |
|---|---|
| New version state/cache-incompatible with old (e.g., tokenizer change) | **Blue-green** (atomic cutover) |
| Gradual risk validation, versions compatible, want early rollback with small blast radius | **Canary** (progressive % traffic) |
| Measuring business/quality metric differences over time | **A/B test** |
| No validation gate, replica-by-replica update | Rolling update (higher risk, avoid for major version changes) |

**Canary progression pattern:** 5-10% → monitor → 25% → 50% → 100%, gated by latency/error/quality thresholds at each stage. Rollback immediately on breach.

### Autoscaling Trigger Signals (ranked by reliability for LLM serving)
| Signal | Use For |
|---|---|
| Queue depth | Leading indicator of backlog before latency degrades |
| Latency SLO breach (p95/p99) | Confirms user-facing impact |
| GPU utilization | Supplementary — can be misleading alone due to variable generation length |
| Tokens/sec per replica | Capacity planning, right-sizing replica count |

**Key rule:** GPU utilization staying moderate does NOT mean capacity is sufficient — always cross-check queue depth + latency percentiles.

### Load Balancing Strategies
| Strategy | Best When |
|---|---|
| Least-connections / least-outstanding-requests | Variable response duration (typical for LLM generation) |
| Round-robin | Uniform, short, predictable request duration |
| Sticky sessions | Multi-turn conversation state continuity (not for raw load balancing) |

### Monitoring Metrics — Always Use Percentiles, Not Averages
- p50 / p95 / p99 latency — averages hide tail-latency SLO violations
- Token throughput (tokens/sec) — generation speed & GPU efficiency
- Error rate (4xx/5xx, timeout rate)

### Drift & Retraining
| Drift Type | Definition |
|---|---|
| Data drift | Input distribution shifts over time |
| Model/concept drift | Input→output relationship (correctness) shifts |

**Retraining trigger criteria (combine both before acting):**
1. Statistical signal — rising embedding distance / distribution shift from training data
2. Quality signal — sustained drop in eval scores or rise in negative feedback

**Process:** Detect signal → validate against labeled eval set (rule out false positive) → decide retrain/fine-tune/no-action. Never retrain on drift signal alone without validation.

### Common Exam Traps
- Choosing canary when versions are incompatible (should be blue-green)
- Using average latency instead of p95/p99 to judge SLO compliance
- Treating GPU utilization as the sole autoscaling trigger
- Triggering retraining from a single noisy drift signal without validation
- Using round-robin for highly variable-duration LLM generation workloads
