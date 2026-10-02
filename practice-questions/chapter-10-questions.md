## Chapter 10: Autoscaling, Rollouts, and LLM MLOps Lifecycle Monitoring — Practice Questions

**Q1.** A multi-model LLM serving endpoint shows GPU utilization holding steady at 60% while request queue depth is climbing and p99 latency has breached the SLO. Which autoscaling signal should drive the scale-out decision?

A) GPU utilization, since it directly reflects compute saturation
B) Queue depth combined with latency SLO breach, since utilization can stay moderate while requests back up due to batching/sequencing delays
C) CPU utilization on the host node
D) Number of loaded model versions on the endpoint

<details>
<summary>Answer</summary>

**Correct: B**

In LLM serving, GPU utilization can look 'fine' while requests queue up because of variable generation length, batching windows, or KV-cache contention. Queue depth plus latency SLO breach is the leading indicator of user-facing degradation and is the standard trigger for scale-out.

**Why the others are wrong:**
- GPU utilization alone is a lagging/misleading signal for token-generation workloads with variable sequence lengths
- CPU utilization rarely correlates with LLM inference bottlenecks, which are GPU/memory bound
- Number of loaded model versions is a deployment config detail, not a load signal

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** A team is rolling out a new LLM version whose tokenizer changed in a way that is NOT backward compatible with cached prompts/embeddings from the previous version. Which rollout strategy should they use?

A) Canary rollout, routing 5% of traffic to the new version while both versions run simultaneously
B) Rolling update one replica at a time with no validation gate
C) Blue-green deployment, cutting traffic over atomically once the new version is fully validated in the idle environment
D) A/B testing with random assignment lasting several weeks

<details>
<summary>Answer</summary>

**Correct: C**

Blue-green is preferred when the new version is incompatible with the old one's state/cache, because it avoids running both versions concurrently against shared or overlapping state — the cutover is atomic and easy to roll back.

**Why the others are wrong:**
- Canary rollout runs both versions simultaneously, which is risky when tokenizer/cache incompatibility could produce inconsistent or corrupted results for a subset of users
- Rolling update without a validation gate offers no safety check before user exposure
- A/B testing is for measuring business/quality metrics over time, not for safely deploying an incompatible version

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** You are load balancing across LLM inference replicas where response generation length varies widely (some requests finish in 200ms, others take 8s). Which load balancing strategy is BEST suited?

A) Round-robin, distributing requests evenly regardless of replica load
B) Random routing with no state awareness
C) Least-connections (or least-outstanding-requests) routing, sending new requests to the replica with the fewest active requests
D) Sticky sessions based on client IP

<details>
<summary>Answer</summary>

**Correct: C**

With highly variable response times, round-robin can overload a replica already handling long-running generations. Least-connections/least-outstanding-requests routing accounts for current in-flight load and balances more evenly under variable-duration workloads.

**Why the others are wrong:**
- Round-robin ignores current load and can send new requests to an already-busy replica handling a long generation
- Random routing has the same blind spot as round-robin — no load awareness
- Sticky sessions are for stateful conversation continuity, not for balancing compute load, and can cause hotspotting

</details>

---

**Q4.** During real-time monitoring of a production LLM endpoint, which combination of metrics is MOST directly tied to user-perceived responsiveness?

A) Average latency and total request count
B) p50/p95/p99 latency percentiles and token throughput per second
C) GPU memory allocated and disk I/O
D) Number of model checkpoints stored

<details>
<summary>Answer</summary>

**Correct: B**

Averages hide tail-latency problems that a subset of users experience; tracking p50/p95/p99 latency alongside token throughput (tokens/sec) captures both typical and worst-case responsiveness plus generation speed, which is the standard for LLM SLOs.

**Why the others are wrong:**
- Average latency masks tail latency spikes that violate SLOs for a portion of traffic
- GPU memory and disk I/O are infrastructure health signals, not user-experience metrics
- Checkpoint count is a lifecycle/versioning artifact, unrelated to real-time serving performance

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** What is the key distinction between data drift and model (concept) drift in an LLM production system?

A) Data drift and model drift are the same phenomenon described with different terminology
B) Data drift only applies to structured tabular data, never to text
C) Model drift can only be detected by retraining the model from scratch
D) Data drift refers to a change in the distribution of input data over time, while model/concept drift refers to a change in the relationship between inputs and the correct/expected outputs

<details>
<summary>Answer</summary>

**Correct: D**

Data drift = input distribution shifts (e.g., new topics, vocabulary, prompt styles). Concept/model drift = the mapping from input to correct output changes (e.g., what 'correct' or 'relevant' means has shifted), even if input distribution looks similar.

**Why the others are wrong:**
- They are distinct signals monitored differently — input distribution stats vs. output quality/accuracy trends
- Data drift applies to any input modality including embeddings/text feature distributions
- Model drift is detected via monitoring signals (quality scores, feedback, embedding distance) before any retraining decision is made

</details>

---

**Q6.** A company monitors an LLM's production outputs and observes a sustained rise in user thumbs-down feedback and a growing embedding-distance score between recent prompts and the original training distribution over the past three weeks. What is the correct next step per standard MLOps lifecycle practice?

A) Treat this as a drift trigger: investigate root cause, validate against a labeled evaluation set, then decide whether retraining or fine-tuning is warranted
B) Ignore the signals since occasional user complaints are normal
C) Immediately retrain the model without further investigation
D) Roll back to a much older model version regardless of its performance

<details>
<summary>Answer</summary>

**Correct: A**

Sustained negative feedback trend plus rising embedding/input drift distance are classic drift-trigger criteria. Standard practice is to validate the signal against a held-out/labeled eval set before committing to retraining, since drift signals alone can have false positives.

**Why the others are wrong:**
- Ignoring sustained (not sporadic) negative trend metrics violates monitoring-driven MLOps practice
- Retraining without validating the drift signal risks wasted compute and doesn't confirm root cause
- Rolling back to an older version arbitrarily doesn't address the drift and may reintroduce other regressions

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** For capacity planning of an LLM endpoint expected to serve long-form generation requests, which metric provides the clearest signal for right-sizing GPU replica count?

A) Tokens generated per second per replica, correlated against target throughput SLOs
B) Total number of HTTP requests per day
C) Number of unique users per day
D) Disk space consumed by logs

<details>
<summary>Answer</summary>

**Correct: A**

Because generation length varies, raw request counts don't reflect actual compute load. Tokens/sec per replica directly maps to GPU compute consumption and is the standard basis for capacity/replica-count planning against throughput SLOs.

**Why the others are wrong:**
- Total requests per day doesn't account for variable token counts per request
- Unique users per day is a business metric, not a load/capacity metric
- Log disk space is unrelated to compute capacity planning

</details>

---

**Q8.** When configuring a canary rollout for a new LLM model version, what is the standard progression pattern for traffic shifting?

A) Start with a small percentage of traffic (e.g., 5-10%), monitor key metrics against defined thresholds, and progressively increase traffic only if metrics stay within acceptable bounds
B) Shift 100% of traffic immediately and monitor afterward
C) Split traffic exactly 50/50 for the entire rollout duration with no progression
D) Route only internal test traffic and never expose it to production users

<details>
<summary>Answer</summary>

**Correct: A**

Canary rollouts progressively increase exposure (e.g., 5% → 25% → 50% → 100%) gated by monitoring latency, error rate, and quality metrics at each stage, allowing early rollback with minimal blast radius if a regression is detected.

**Why the others are wrong:**
- Shifting 100% immediately is a blue-green cutover, not a canary — it defeats the purpose of gradual risk exposure
- A static 50/50 split for the whole duration is an A/B test, not a canary progression
- Never exposing to production users prevents validating real-world performance and defeats the purpose of a canary

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

