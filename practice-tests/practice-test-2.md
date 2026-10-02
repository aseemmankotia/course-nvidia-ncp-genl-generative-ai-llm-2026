# Practice Test 2: NCP-GENL: NVIDIA Generative AI & LLMs Cert Prep

- **Questions:** 45
- **Time limit:** 90 minutes
- **Passing score:** 70%%

> Answers are in `answer-key-2.md`. Track results in `score-tracker.md`.

---

**Q1.** A team deploying a 70B-parameter decoder-only LLM for high-throughput inference finds that KV cache memory dominates GPU memory when serving long contexts at batch size 64. They need to cut KV cache size substantially while retaining quality close to full multi-head attention. Which architectural choice best addresses this?

A) Switch to Grouped-Query Attention (GQA) with a small number of shared KV head groups
B) Increase the number of attention heads while keeping head_dim constant
C) Replace self-attention with global average pooling across the sequence
D) Enable Multi-Query Attention only during training, then revert to MHA for inference

---

**Q2.** A pretrained model uses RoPE (rotary position embeddings) with a native training context of 4K tokens. The team must serve 32K-token contexts without retraining the model from scratch. Which approach is most appropriate?

A) Remove positional encodings entirely so the model generalizes to any length
B) Increase the tokenizer vocabulary size to represent longer sequences
C) Replace RoPE with absolute learned positional embeddings at inference time
D) Apply RoPE frequency base scaling (e.g., NTK-aware interpolation or YaRN) and lightly fine-tune on longer sequences

---

**Q3.** Profiling a transformer training run on H100 GPUs shows the attention operation is bottlenecked not by FLOPs but by repeated reads/writes of the O(n^2) attention score matrix to HBM. Which technique specifically targets this bottleneck?

A) FlashAttention, which tiles computation and fuses kernels to keep intermediate scores in SRAM, avoiding materialization of the full attention matrix in HBM
B) Mixed precision training using FP16 weights only
C) Increasing batch size to better utilize tensor cores
D) Gradient checkpointing to reduce activation memory

---

**Q4.** A sparse Mixture-of-Experts model uses top-2 routing among 8 experts per layer with a capacity factor of 1.25. During training, many tokens are dropped due to expert overflow, hurting quality. Which configuration change most directly reduces token dropping while keeping the same number of experts?

A) Decrease the number of experts from 8 to 4
B) Increase the capacity factor (e.g., to 2.0) so each expert can accept more tokens per batch
C) Reduce top-k routing from 2 to 1
D) Switch from top-k routing to random expert assignment

---

**Q5.** Comparing byte-pair encoding (BPE) with SentencePiece's unigram language model tokenizer for a multilingual LLM handling low-resource, morphologically rich languages, which statement is most accurate?

A) BPE and unigram SentencePiece always produce identical vocabularies for the same training corpus
B) BPE requires whitespace pre-tokenization and therefore cannot be applied to languages without spaces, such as Japanese or Thai
C) Unigram-based SentencePiece evaluates multiple candidate segmentations and selects the most probable one, often yielding better subword boundaries for morphologically rich languages than greedy BPE merges
D) SentencePiece can only be used with encoder-decoder architectures, not decoder-only LLMs

---

**Q6.** Given a fixed training compute budget C, Chinchilla scaling law research found that many earlier large models (e.g., GPT-3 175B) were compute-suboptimal because they were:

A) Undertrained relative to their parameter count — compute-optimal training requires scaling both parameters and training tokens roughly proportionally
B) Trained on too many tokens relative to parameter count, causing overfitting
C) Using too few attention heads for their hidden dimension size
D) Trained with a context window too short for their parameter count

---

**Q7.** A team serves a 70B-parameter decoder-only model for long-context (32K token) inference on NVIDIA H100 GPUs. Profiling shows the KV cache dominates GPU memory, limiting batch size and throughput. Switching from standard multi-head attention to multi-query attention (MQA) reduced memory pressure but caused a noticeable drop in generation quality on reasoning benchmarks. Which architectural change BEST balances KV cache memory savings with quality retention for this workload?

A) Increase the number of attention heads while keeping separate KV projections per head to restore quality, accepting the original memory cost
B) Switch to a sparse mixture-of-experts (MoE) architecture to reduce active parameters per token
C) Replace attention with grouped-query attention (GQA), using a small number of shared key/value head groups instead of a single shared KV head
D) Apply FlashAttention kernels only, keeping the original multi-head attention structure unchanged

---

**Q8.** A team implements self-consistency decoding for a multi-step arithmetic reasoning task: they sample 15 chain-of-thought completions at temperature 0.7 and take the majority-vote final answer. Accuracy improves over greedy decoding but inference cost rises significantly. Which statement BEST explains the mechanism behind the accuracy gain?

A) Self-consistency fine-tunes the model weights on the sampled reasoning paths before selecting the final answer
B) Sampling diverse reasoning paths and marginalizing over them via majority vote reduces variance from any single flawed reasoning chain, surfacing the most consistent final answer
C) Higher temperature sampling increases the model's effective context window, allowing it to consider more of the prompt
D) Self-consistency replaces the language model's softmax output with a constrained grammar to force valid arithmetic tokens
E) Self-consistency
F) Self-consistency

---

**Q9.** An engineer builds a ReAct agent that interleaves Thought/Action/Observation steps to answer questions using a search tool and a calculator tool. During testing, the agent frequently emits a syntactically valid Action call to the calculator with a malformed expression, causing a tool error, but the agent then loops repeatedly retrying the same malformed call instead of recovering. What is the MOST likely root cause and fix?

A) The tool schema is missing type annotations, so switching to a vector database as the tool backend will resolve the looping
B) The observation returned from the failed tool call is not being fed back into the prompt context in a way the model can act on, so the fix is to explicitly include the error message as an Observation and instruct the model to revise its Action based on it
C) The model's temperature is set too low, so increasing temperature to 1.5 will force the agent to explore alternative actions and break the loop
D) ReAct fundamentally cannot support calculator tools since they are not differentiable, so a fine-tuned tool-use model must replace the prompted agent entirely
E) The tool schema is missing type annotations, so switching to a vector database as the tool backend will resolve the looping

---

**Q10.** A production system must guarantee that every LLM response is a syntactically valid JSON object matching a fixed schema (specific keys, enum-constrained string values, integer fields), with zero tolerance for malformed output reaching downstream services. Prompt-only instruction to "respond in JSON" produces a 92% valid-output rate. Which approach provides the strongest guarantee of schema compliance?

A) Increase the number of few-shot JSON examples in the prompt from 2 to 10 to reinforce the format through in-context learning alone
B) Apply grammar-constrained (e.g., context-free grammar or JSON-schema-based) decoding that restricts the token sampling distribution at each step to only tokens valid under the schema
C) Lower the temperature to 0 and rely on greedy decoding, since deterministic decoding guarantees valid JSON structure
D) Add a regex-based post-processing step that discards invalid outputs and re-prompts the model up to 3 times, accepting whatever is returned on the final attempt
E) Increase the number of few-shot JSON examples in the prompt from 2 to 10 to reinforce the format through in-context learning alone

---

**Q11.** A team must answer customer questions about a rapidly changing product catalog (prices and stock updated hourly) with source citations required for compliance, using a budget that precludes any model fine-tuning and requires sub-2-second latency. Which adaptation strategy is MOST appropriate?

A) Fine-tune a smaller open-weight LLM nightly on the updated catalog data to embed current facts directly into model weights
B) Rely purely on zero-shot prompting with detailed system instructions describing the catalog structure, without any external data access
C) Implement retrieval-augmented generation that indexes the catalog into a vector store updated hourly, retrieves relevant records at query time, and grounds generation with citations to retrieved passages
D) Use chain-of-thought prompting alone to have the model reason step-by-step about likely current prices based on its training data
E) Fine-tune a smaller open-weight LLM nightly on the updated catalog data to embed current facts directly into model weights

---

**Q12.** During prompt evaluation for a summarization task, a team A/B tests two prompt variants against a held-out set of 200 documents, using ROUGE-L score as the sole evaluation metric. Variant B scores higher on ROUGE-L but human reviewers rate Variant A's summaries as more useful and factually accurate. What is the MOST likely explanation and appropriate corrective action?

A) ROUGE-L is invalid for any generative task and should be replaced entirely with BLEU score for future evaluation
B) Variant B likely generates summaries with high lexical/n-gram overlap to reference summaries without necessarily preserving factual accuracy or usefulness; the team should add human evaluation or an LLM-as-judge rubric assessing faithfulness alongside ROUGE-L
C) The held-out set of 200 documents is too small to produce a statistically meaningful ROUGE-L score, so the sample size should be reduced further to remove noise
D) Human reviewers are inherently unreliable compared to automated metrics, so their ratings should be discarded in favor of the ROUGE-L result
E) The held-out set of 200 documents is too small to produce a statistically meaningful ROUGE-L score, so the sample size should be reduced further to remove noise

---

**Q13.** A team designs a few-shot prompt with 8 in-context examples for a classification task and finds that swapping the order of the examples (same examples, different sequence) causes classification accuracy to vary by over 15 percentage points on the same test set, with no change to model or examples' content. What does this behavior primarily indicate about few-shot in-context learning, and what is the recommended mitigation?

A) The model's context window is too small to hold 8 examples, so the fix is to reduce to 2 examples regardless of order
B) This indicates a bug in the tokenizer that must be patched before few-shot prompting can be used reliably
C) LLMs exhibit order sensitivity in few-shot prompts due to recency and positional biases; mitigation includes testing multiple example orderings, using majority-vote across orderings, or selecting an order empirically validated on a dev set
D) This is expected and no mitigation is needed since few-shot accuracy is always independent of prompt structure by design
E) This is expected and no mitigation is needed since few-shot accuracy is always independent of prompt structure by design

---

**Q14.** A team built a customer support assistant for a rapidly changing product catalog (prices and stock change hourly). Users report the model confidently states outdated prices from its training data. The team has a clean, structured product database updated in real time but no budget for frequent retraining. Which adaptation strategy BEST resolves this issue?

A) Implement retrieval-augmented generation that queries the live product database and injects current price and stock data into the prompt context before generation
B) Increase the temperature parameter to encourage the model to express more uncertainty about specific facts
C) Fine-tune the model weekly on a snapshot of the product database so it memorizes current prices
D) Add a one-shot example in the prompt showing a correctly formatted price response

---

**Q15.** A team is curating a 2TB web-crawled corpus for continued pretraining and needs to remove near-duplicate documents at scale without relying on exact string matching. Which approach is most appropriate?

A) Exact hash-based deduplication using MD5 checksums on full documents
B) Removing all documents shorter than 100 tokens
C) MinHash with Locality-Sensitive Hashing (LSH) to identify near-duplicate document pairs based on n-gram similarity
D) Sorting documents alphabetically and removing consecutive duplicates

---

**Q16.** A company has only 500 labeled instruction-response pairs for a niche legal domain and a single A100 80GB GPU. They want to adapt a 13B-parameter LLM without overfitting or requiring massive compute. Which customization strategy is BEST?

A) Full fine-tuning of all model parameters on the 500 examples
B) Continued pretraining on unlabeled legal text followed by RLHF
C) LoRA or QLoRA fine-tuning targeting attention projection layers with a low rank
D) Training a new tokenizer from scratch on the 500 examples

---

**Q17.** During LoRA fine-tuning in NeMo, increasing the rank (r) hyperparameter primarily affects which trade-off?

A) Higher rank increases the number of trainable parameters and expressive capacity, at the cost of more memory and higher risk of overfitting on small datasets
B) Higher rank reduces tokenization overhead during preprocessing
C) Higher rank always reduces GPU memory usage during training
D) Higher rank has no impact on model capacity, only on inference latency

---

**Q18.** A team wants to align a fine-tuned LLM to human preferences but lacks the infrastructure to train and maintain a separate reward model or run PPO-based reinforcement learning. They do have a dataset of paired chosen/rejected responses. Which approach best fits their constraints?

A) Full RLHF with PPO, since it is required whenever preference data exists
B) P-tuning using virtual tokens conditioned on preference labels
C) Direct Preference Optimization (DPO), which directly optimizes the policy on preference pairs without a separate reward model or RL loop
D) Supervised fine-tuning only, ignoring the preference labels entirely

---

**Q19.** How does p-tuning differ from LoRA as a parameter-efficient fine-tuning method?

A) P-tuning is only compatible with encoder-only models while LoRA works only with decoder-only models
B) P-tuning learns continuous virtual token embeddings prepended to the input via a small prompt encoder, while LoRA injects low-rank update matrices into existing weight layers
C) P-tuning modifies attention weights directly via full backpropagation, while LoRA freezes all model parameters
D) P-tuning requires modifying the vocabulary and tokenizer, while LoRA does not require any prompt engineering

---

**Q20.** When using NVIDIA NeMo Curator to prepare a pretraining dataset, which stage is responsible for identifying and filtering out documents containing toxic or unsafe content before deduplication and training?

A) Tokenization and shuffling stage
B) Quality and classifier-based filtering, which applies heuristic and model-based filters including toxicity/safety classifiers
C) Blending multiple data sources by weighted sampling
D) Exact substring deduplication using suffix arrays

---

**Q21.** A team has collected only 500 human-labeled preference pairs (chosen vs. rejected chat responses) to make an assistant favor safer replies. Compute budget is tight, and the team wants to avoid the overhead of training and maintaining a separate reward model. Which customization approach BEST fits these constraints?

A) Direct Preference Optimization (DPO)
B) RLHF with PPO using a trained reward model
C) Supervised fine-tuning (SFT) only on the chosen responses, discarding the rejected ones
D) Continued pretraining on a large unlabeled safety-related corpus

---

**Q22.** You are using NVIDIA NeMo Curator to deduplicate a 2TB web-crawled pretraining corpus. Exact byte-level matching misses many near-duplicate documents that differ only in HTML boilerplate, ads, or minor edits. Which deduplication approach should you configure to catch these near-duplicates efficiently at this scale?

A) Fuzzy deduplication using MinHash signatures combined with Locality-Sensitive Hashing (LSH)
B) Exact hash-based deduplication using SHA-256 checksums on full document text
C) Manual review of the 1000 largest documents in the corpus
D) Semantic deduplication computing pairwise embedding cosine similarity across all documents

---

**Q23.** A team trains a 70B-parameter model on 64 GPUs arranged as 8 nodes of 8 GPUs each, with NVLink connecting GPUs within a node and InfiniBand connecting nodes. The model does not fit on a single GPU due to activation memory even after optimizer-state sharding. Which parallelism mapping minimizes cross-node communication overhead while still fitting the model in memory?

A) Pipeline parallelism degree 8 within each node, tensor parallelism degree 8 across nodes
B) Data parallelism degree 64 with ZeRO-3 only, no model parallelism
C) Tensor parallelism degree 64 flat across all 64 GPUs
D) Tensor parallelism degree 8 within each node (NVLink), pipeline parallelism degree 8 across nodes, data parallelism degree 1

---

**Q24.** An engineer needs to reduce GPU memory consumed by optimizer states and gradients during training but wants to avoid the parameter all-gather communication overhead that occurs during the forward pass. Which ZeRO stage should be selected?

A) ZeRO Stage 1 (optimizer state sharding only)
B) ZeRO Stage 2 (optimizer state + gradient sharding)
C) ZeRO Stage 3 (optimizer state + gradient + parameter sharding)
D) ZeRO-Infinity (NVMe offload of optimizer states and parameters)

---

**Q25.** When enabling FP8 mixed-precision training on H100 GPUs using NVIDIA Transformer Engine, which components are typically kept in FP8 and which remain in higher precision to preserve model accuracy?

A) FP8 for GEMM operations in linear layers, while layer normalization, softmax, and master weights remain in BF16/FP32
B) FP8 for all operations including normalization and softmax, to maximize throughput
C) FP8 only for the embedding layer, with all other operations remaining in FP32
D) FP8 for gradient accumulation only, with the forward pass remaining in FP32

---

**Q26.** A team implements speculative decoding for a 70B-parameter model to reduce per-token latency using a smaller draft model. During testing, only 20% of speculated tokens are accepted, and overall latency INCREASES compared to standard autoregressive decoding. What is the most likely cause?

A) KV cache is disabled, which is a strict requirement for speculative decoding to function
B) The draft model is too large or slow relative to its acceptance rate, so verification overhead exceeds the savings from skipped target-model calls
C) Speculative decoding requires FP8 quantization to function, and it was not applied to the target model
D) Continuous batching interferes with speculative decoding and must be disabled

---

**Q27.** A team needs to quantize a large language model's weights to 4 bits for deployment on memory-constrained GPUs. The model exhibits significant activation outliers in certain channels, and accuracy degradation must be minimized without an extensive calibration process. Which quantization approach is BEST suited?

A) Naive INT4 round-to-nearest (RTN) quantization applied uniformly to all weights
B) GPTQ, which requires a per-layer calibration dataset and iterative weight reconstruction
C) AWQ (Activation-aware Weight Quantization), which scales weight channels based on activation magnitude to protect salient channels
D) FP8 quantization applied uniformly to both weights and activations

---

**Q28.** Gradient checkpointing (activation recomputation) reduces GPU memory usage during training by discarding intermediate activations and recomputing them during the backward pass. What is the primary tradeoff of enabling this technique?

A) Increased GPU memory usage due to storing extra checkpoint tensors
B) Reduced model accuracy due to lossy activation compression
C) Incompatibility with mixed-precision training
D) Increased training time, typically 20-30% overhead, due to redundant forward-pass computation performed again during the backward pass

---

**Q29.** You are training a 70B-parameter model with a 32K-token context length on A100 80GB GPUs. Profiling shows activation memory (not parameter or optimizer memory) is the dominant bottleneck, and tensor parallelism across 8 GPUs per node is already saturating NVLink bandwidth. Which technique should you add to reduce activation memory further with minimal additional communication overhead?

A) Increase ZeRO stage 3 to shard optimizer states more aggressively across additional data-parallel replicas
B) Increase pipeline parallelism depth to split more layers across additional nodes
C) Enable sequence parallelism in conjunction with the existing tensor-parallel group
D) Apply full activation gradient checkpointing on every transformer layer instead

---

**Q30.** A team is deploying a fine-tuned 13B chat model with TensorRT-LLM for an interactive assistant. The SLA requires minimizing per-request decode latency without any measurable drop in output quality, and a smaller same-family draft model is available. Continuous batching and INT8 weight quantization are already enabled. Which additional technique BEST meets the latency target?

A) Enable speculative decoding using the smaller draft model to propose multiple tokens verified in parallel by the target model
B) Increase the continuous batching max batch size to improve GPU utilization
C) Reduce the KV-cache paging block size to fit more requests in memory
D) Switch the target model's weights from INT8 to FP8 quantization

---

**Q31.** You must train a 175B-parameter model on 64 GPUs (8 nodes x 8 GPUs) where even 8-way tensor parallelism within a node does not make the model fit alongside pipeline parallelism across nodes. Following standard large-scale 3D parallelism practice (as used in Megatron-DeepSpeed style training), which combination of ZeRO stage and parallelism strategy is appropriate?

A) Tensor parallelism + pipeline parallelism + ZeRO stage 3 full parameter and optimizer sharding across all ranks
B) Tensor parallelism + pipeline parallelism + ZeRO stage 1 optimizer state sharding across the remaining data-parallel groups
C) Data parallelism with ZeRO stage 3 only, omitting tensor and pipeline parallelism entirely
D) Pipeline parallelism with ZeRO stage 2 gradient sharding only, omitting tensor parallelism

---

**Q32.** A team serves an LLM via Triton Inference Server behind a strict 50ms p99 latency SLA. Throughput is inadequate at low concurrency even though GPU utilization is low. Which config.pbtxt change BEST increases throughput while still respecting the SLA?

A) Set max_queue_delay_microseconds to a high value like 500000 microseconds to maximize batch fill before execution
B) Disable dynamic batching entirely and rely solely on increasing the instance_group count
C) Set max_batch_size to 1 to guarantee the lowest possible per-request latency
D) Enable dynamic_batching with preferred_batch_size tuned to common request sizes and max_queue_delay_microseconds set well below the SLA's remaining latency budget (e.g., 10000 microseconds)

---

**Q33.** A financial services company must replace a production LLM (v1) with a newly fine-tuned version (v2) with zero downtime, and needs the ability to instantly roll back if the hallucination rate exceeds a threshold once observed on real user traffic. Which deployment pattern BEST satisfies these requirements?

A) Blue-green deployment that switches 100% of traffic from v1 to v2 at once after passing offline evaluation
B) Shadow deployment that duplicates all traffic to v2 without returning any v2 responses to users, then fully switches after one week regardless of findings
C) Canary rollout that routes a small percentage (e.g., 5%) of live traffic to v2 while monitoring quality metrics, with automated rollback triggers tied to the hallucination threshold
D) A/B testing that permanently splits traffic 50/50 between v1 and v2 to gather long-term user preference data

---

**Q34.** A platform must serve 200 customer-specific LoRA-adapted variants of the same base LLM cost-effectively, with sub-second adapter switching and without loading 200 full model copies onto GPUs. Which approach BEST meets this on the NVIDIA stack?

A) Use NVIDIA NIM's multi-LoRA serving to host a single shared base model with adapters dynamically loaded from shared GPU memory
B) Deploy 200 separate Triton model instances, each loading its own full copy of the base model plus adapter
C) Merge all 200 LoRA adapters into the base weights offline and deploy 200 fully merged model checkpoints
D) Compile a separate optimized TensorRT-LLM engine per adapter and store all 200 engines in the Triton model repository

---

**Q35.** In Triton Inference Server, what is the key functional difference between 'dynamic batching' and 'concurrent model execution' (instance groups)?

A) Both mechanisms refer to the same process of splitting a single request's computation across multiple GPUs
B) Dynamic batching only applies to CPU-deployed models, while concurrent execution only applies to GPU-deployed models
C) Concurrent execution merges incoming requests into batches, while dynamic batching creates additional model instances
D) Dynamic batching combines multiple incoming inference requests into a single batch before execution, while concurrent execution runs multiple separate instances of the same model in parallel to serve independent requests

---

**Q36.** A production RAG-based LLM system's monitoring dashboard shows a gradual increase in retrieval-answer mismatch rate over three weeks, with no code, model, or infrastructure changes deployed in that period. What is the MOST likely root cause and appropriate MLOps action?

A) GPU hardware degradation; the action is to replace GPUs and redeploy the identical model with no other changes
B) Network latency increase between Triton and the vector database; the action is to horizontally scale the load balancer
C) Data drift in the underlying knowledge corpus or user query distribution; the action is to trigger corpus re-indexing and evaluate whether the retriever or LLM needs re-tuning
D) Model container image corruption; the action is to redeploy the container from the same image tag

---

**Q37.** An LLM inference service running Triton on Kubernetes experiences bursty traffic. The team configured the Horizontal Pod Autoscaler (HPA) using average CPU utilization, but users still see latency spikes during bursts even though CPU stays below 40%. What is the BEST fix?

A) Increase the CPU utilization scale-out threshold from 40% to 60% to make scaling more conservative
B) Remove HPA entirely and manually scale the deployment to a fixed high replica count at all times
C) Increase the number of CPU cores allocated to each pod without changing the autoscaling metric
D) Switch the autoscaling metric to a custom metric such as GPU utilization or Triton's inference queue depth/pending request count, since CPU is not representative of GPU-bound LLM inference load

---

**Q38.** A team serves a fine-tuned LLM using Triton Inference Server with dynamic batching enabled. Latency SLAs are being violated intermittently under bursty traffic, even though GPU utilization stays below 60%. The max_queue_delay_microseconds is set to 100000 (100ms) and preferred_batch_sizes are [8,16]. Requests often arrive singly with gaps of 150-300ms between them. What is the BEST fix to reduce latency while still benefiting from batching when traffic is dense?

A) Increase max_queue_delay_microseconds to 300000 so more requests can accumulate before batching
B) Lower max_queue_delay_microseconds and add smaller preferred_batch_sizes so isolated requests aren't held waiting for a full batch
C) Disable dynamic batching entirely and switch to a static batch size of 1 for all requests
D) Increase preferred_batch_sizes to [32,64] to maximize throughput per batch

---

**Q39.** An LLM inference service is deployed on Kubernetes behind a canary rollout: 5% of production traffic is routed to a newly retrained model version while 95% goes to the stable version. After several hours, monitoring shows the canary's average response latency and GPU memory usage match the stable version, but its aggregate output quality score (from an automated eval harness) has silently dropped by 12% on a subset of prompts related to a recent knowledge domain. Which monitoring practice would have caught this regression EARLIEST, before it affected a larger rollout percentage?

A) Relying solely on infrastructure metrics (latency, throughput, GPU utilization) as canary promotion gates
B) Increasing the canary traffic percentage faster to get statistically significant infra metrics sooner
C) Running periodic offline batch evaluation only after the canary reaches 50% traffic
D) Continuously scoring canary responses against a held-out quality/drift benchmark in real time and gating promotion on both infra and quality metrics

---

**Q40.** A financial services company deployed a customer-facing LLM assistant. They need the assistant to refuse specific stock buy/sell recommendations while still allowing general financial education topics, and must catch this both when users ask indirectly and when generated responses drift toward recommendations mid-conversation. Which NeMo Guardrails configuration BEST meets this requirement?

A) Define only an input rail that blocks messages containing stock ticker symbols
B) Rely solely on the base LLM's built-in safety tuning without additional guardrails
C) Define dialogue rails with Colang flows for topical boundaries combined with output rails that check generated responses for prohibited recommendation patterns
D) Use execution rails to sandbox any code execution requests

---

**Q41.** A team must evaluate a summarization LLM's output for factual consistency against 50,000 source documents, but budget does not allow full human review of every summary. They still need confidence that automated scores correlate with human judgment. Which evaluation strategy is BEST?

A) Use human evaluation on the full 50,000-sample set
B) Use LLM-as-judge for full-dataset scoring, validated against a human-annotated stratified subsample to measure judge-human agreement
C) Use only ROUGE score since it is a summarization-specific metric
D) Use LLM-as-judge on the full dataset with no validation against human labels

---

**Q42.** Which technique is most effective for detecting hallucinations in a RAG-based LLM system that must ground its responses in retrieved documents?

A) Measure output perplexity alone
B) Compute attribution/grounding scores measuring overlap and entailment between generated claims and the retrieved source passages
C) Increase generation temperature to reduce repetitive phrasing
D) Compute BLEU score against a fixed set of reference summaries

---

**Q43.** You are configuring NeMo Guardrails to detect and block jailbreak attempts (e.g., 'ignore previous instructions' prompt-injection patterns) before the LLM generates a response. Where should this primary detection logic be implemented?

A) As a retrieval rail that filters documents returned by the vector database
B) As an execution rail that restricts tool call permissions
C) As an output rail that filters the final generated text after the LLM has responded
D) As an input rail using a jailbreak detection flow that intercepts and evaluates the user message before it reaches the LLM

---

**Q44.** In responsible AI evaluation for LLMs, 'allocational bias' specifically refers to:

A) Bias introduced solely during the tokenization process
B) Bias caused by using stale training data that no longer reflects current events
C) Overrepresentation of certain writing styles in generated text
D) Unequal distribution of resources, opportunities, or outcomes across demographic groups as a result of model decisions

---

**Q45.** During error analysis of an LLM used to generate resume screening summaries, the team wants to detect whether the model exhibits differential performance across gender-associated names in input resumes. What is the BEST evaluation approach?

A) Run a single LLM-as-judge pass without demographic stratification and assume equal treatment
B) Perform stratified evaluation by perturbing gender-associated names while holding all other resume content constant, comparing output quality and sentiment across groups
C) Increase the model's context window to include more resume detail
D) Compute overall aggregate accuracy across the full test set and compare it to a published benchmark score

---

