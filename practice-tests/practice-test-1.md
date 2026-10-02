# Practice Test 1: NCP-GENL: NVIDIA Generative AI & LLMs Cert Prep

- **Questions:** 45
- **Time limit:** 90 minutes
- **Passing score:** 70%%

> Answers are in `answer-key-1.md`. Track results in `score-tracker.md`.

---

**Q1.** A team is deploying a 70B-parameter LLM for real-time chat inference on GPUs with limited KV-cache memory bandwidth. They need to substantially reduce memory traffic during autoregressive decoding while keeping output quality close to standard multi-head attention. Which attention variant BEST meets this requirement?

A) Grouped-Query Attention (GQA), where heads are grouped to share KV pairs
B) Multi-Query Attention (MQA), where all heads share a single KV pair
C) Standard Multi-Head Attention (MHA) with no KV sharing
D) Local sliding-window attention with a fixed window size

---

**Q2.** An organization wants to increase total model capacity to handle multiple domains (code, math, general chat) but must keep per-token inference compute close to that of a much smaller dense model. Which architecture choice BEST satisfies this constraint?

A) A Mixture-of-Experts (MoE) model with sparse token-level routing
B) A dense transformer scaled to greater depth and width
C) An encoder-decoder model with parameter sharing between encoder and decoder
D) A dense model using Multi-Query Attention only

---

**Q3.** A base LLM was pretrained with RoPE positional encoding for a 4K-token context window. The team wants to extend inference to 32K tokens without full retraining. Which technique directly enables this extension?

A) Replacing RoPE entirely with fixed sinusoidal absolute position embeddings
B) Reducing the tokenizer vocabulary size to shorten average sequence length
C) Removing several attention heads to lower the effective sequence footprint
D) Adjusting the RoPE base frequency (theta) using NTK-aware or linear interpolation scaling

---

**Q4.** What is the primary computational/memory benefit that FlashAttention provides over a standard (naive) attention implementation on GPUs?

A) It avoids materializing the full N×N attention matrix in high-bandwidth memory by using tiling and online softmax recomputation
B) It reduces attention FLOPs by approximating the softmax scores
C) It replaces the softmax nonlinearity with a linear kernel approximation
D) It reduces the number of attention heads required for equivalent quality

---

**Q5.** A team trains on a dataset dominated by non-English text with many rare morphological word forms. Using a BPE tokenizer trained mainly on English corpora produces much longer token sequences for this data, raising training cost and shrinking effective usable context. What is the most effective fix?

A) Increase model depth to compensate for the longer sequences
B) Switch to pure character-level tokenization and disable all subword merges
C) Enlarge the model's maximum context window while keeping the same tokenizer
D) Train a new BPE/SentencePiece vocabulary on the target multilingual corpus

---

**Q6.** A company needs a model primarily for abstractive summarization of long documents, where the source article and the output summary have very different lengths and distinct functional roles. Which architecture historically provides the strongest built-in inductive bias for this asymmetric input/output structure?

A) Decoder-only architecture, the modern default for most generative tasks
B) Encoder-decoder (sequence-to-sequence) architecture
C) Encoder-only (BERT-style) architecture
D) A retrieval-only pipeline without a generative component

---

**Q7.** A team is deploying a 70B-parameter decoder-only LLM for a real-time inference service and needs to reduce KV cache memory footprint and improve inference throughput without significantly harming output quality. Which architectural feature should they prioritize choosing in the model?

A) Absolute positional embeddings instead of rotary positional embeddings
B) A larger number of attention heads with full multi-head attention (MHA)
C) Grouped-query attention (GQA) instead of standard multi-head attention
D) An encoder-decoder architecture instead of decoder-only

---

**Q8.** A financial news aggregator answers customer questions about market events that change every hour. The knowledge base is updated continuously, answers must cite the specific source article, and the team cannot afford to retrain a model every hour. Which adaptation strategy BEST meets these requirements?

A) Fine-tune the base model nightly on the day's news articles
B) Retrieval-augmented generation over a continuously indexed news corpus
C) Use zero-shot prompting only, relying on the model's pretrained knowledge
D) Few-shot prompting with hand-picked examples from last month's top stories

---

**Q9.** A legal document classification service must apply a very specific, consistent tagging taxonomy to every document, has 50,000 labeled examples available, and requires sub-100ms inference latency in production with no external calls. Which approach is MOST appropriate?

A) Retrieval-augmented generation pulling similar labeled examples at inference time
B) Fine-tune the model on the labeled dataset to internalize the taxonomy
C) Chain-of-thought prompting to reason through the taxonomy at inference time
D) Zero-shot prompting with a detailed description of the taxonomy in the system prompt

---

**Q10.** An LLM is given a multi-step arithmetic word problem and is prompted with 'Let's think step by step' before producing the final numeric answer. Which prompting technique is being used?

A) ReAct prompting
B) Constrained decoding
C) Chain-of-thought prompting
D) Retrieval-augmented generation

---

**Q11.** You are building an LLM-based assistant that must look up real-time flight status via an API, decide whether to re-query with different parameters based on the API response, and then summarize the result for the user. Which prompting framework is designed specifically for this interleaved reasoning-and-acting pattern?

A) ReAct prompting
B) Few-shot prompting
C) Self-consistency prompting
D) Zero-shot chain-of-thought

---

**Q12.** A production pipeline requires every LLM response to be a syntactically valid JSON object matching an exact schema, with zero tolerance for malformed output that would break a downstream parser. Simply instructing the model in the prompt to 'always respond in valid JSON' has produced occasional malformed outputs. What should be added to GUARANTEE schema-conformant output?

A) Apply grammar-based constrained decoding that restricts token generation to the JSON schema
B) Increase the few-shot examples showing correctly formatted JSON
C) Raise the sampling temperature to increase output diversity
D) Add a chain-of-thought instruction asking the model to double-check its JSON before responding

---

**Q13.** In self-consistency prompting, how does the technique improve accuracy over standard single-pass chain-of-thought prompting?

A) It fine-tunes the model weights on the reasoning traces generated during inference
B) It retrieves external documents to verify each reasoning step against ground truth
C) It samples multiple independent chain-of-thought reasoning paths and selects the most frequent final answer via majority vote
D) It replaces natural language reasoning with a formal grammar-constrained decoding pass

---

**Q14.** A financial services company builds an internal assistant that must answer employee questions using compliance policy documents that are revised weekly by the legal team. The team has a limited GPU budget for retraining and needs updates to be reflected within hours of a document change, with full traceability to the source passage used to generate each answer. Which adaptation strategy BEST satisfies these requirements?

A) Fine-tune the base model on the full policy corpus every week using LoRA adapters, then deploy the newly trained adapter into production
B) Stuff the complete text of all policy documents into the prompt as few-shot context for every query so the model always sees the full corpus
C) Rely on zero-shot prompting with the base model's pretrained knowledge, since large language models already generalize well to compliance topics
D) Implement retrieval-augmented generation: embed and index the policy documents in a vector database, retrieve the most relevant passages at query time, and ground the model's response in that retrieved context

---

**Q15.** A team needs to adapt a 70B-parameter foundation model to a specialized domain using only 500 labeled examples, and their entire training budget is a single 80GB A100 GPU. Which customization approach BEST meets these constraints?

A) Full fine-tuning of all 70B parameters using gradient checkpointing
B) Continued pretraining on a large unlabeled corpus in the target domain
C) QLoRA: 4-bit quantization of the base model combined with LoRA adapters
D) P-tuning with virtual tokens prepended to every input prompt

---

**Q16.** A company wants to align a model's outputs to human preference data but does not have the resources to train a separate reward model and run PPO-based reinforcement learning. Which preference optimization technique should they use instead?

A) RLHF with a trained reward model and PPO
B) Standard supervised fine-tuning on the chosen responses only
C) P-tuning conditioned on preference labels
D) Direct Preference Optimization (DPO)

---

**Q17.** When configuring a LoRA fine-tuning job in NeMo, increasing the rank (r) hyperparameter primarily affects which aspect of training?

A) The tokenizer's vocabulary size used during preprocessing
B) The number of trainable parameters and the adapter's representational capacity
C) The threshold used for near-duplicate document removal
D) The learning rate warmup schedule applied to the optimizer

---

**Q18.** What is the primary reason for performing near-duplicate document removal on a training corpus before pretraining or fine-tuning an LLM?

A) To prevent the model from memorizing repeated content and to improve generalization
B) To reduce tokenization latency by shortening average document length
C) To increase the vocabulary size available to the tokenizer
D) To guarantee an equal number of training examples per output class label

---

**Q19.** A company has 50GB of unlabeled legal contract text and wants the model to acquire domain-specific terminology and writing style, without requiring it to follow explicit instructions or answer specific questions. Which customization strategy is MOST appropriate?

A) Continued pretraining (domain-adaptive pretraining) on the unlabeled legal corpus
B) Instruction-tuning using a curated set of labeled question-answer pairs
C) RLHF using human preference rankings on model outputs
D) DPO using pairs of chosen and rejected legal responses

---

**Q20.** Which NVIDIA NeMo component is purpose-built for large-scale data curation tasks such as exact and fuzzy deduplication, quality filtering, and PII redaction before model training?

A) NeMo Guardrails
B) TensorRT-LLM
C) NeMo Aligner
D) NeMo Curator

---

**Q21.** A team has only 500 human-curated instruction-response pairs for a domain-specific customer support use case and a limited GPU budget for a 2-week turnaround. The base model already performs reasonably on general language tasks but needs to follow the new instruction format and domain terminology. Which customization approach BEST fits these constraints?

A) Apply LoRA fine-tuning on the 500 instruction pairs, training low-rank adapters while keeping the base model weights frozen
B) Perform continued pretraining on a large unlabeled domain corpus before instruction tuning
C) Perform full fine-tuning of all model parameters using the 500 instruction pairs
D) Collect human preference pairs and run RLHF to align the model to the new instruction format

---

**Q22.** QLoRA reduces the memory footprint of the frozen base model during parameter-efficient fine-tuning. Which quantization configuration does QLoRA use for the base model weights?

A) 8-bit integer quantization with per-channel scaling and no double quantization
B) Dynamic FP16 quantization applied only to the attention projection layers
C) 4-bit NormalFloat (NF4) quantization combined with double quantization and paged optimizers
D) Binary quantization with stochastic rounding applied to all transformer weights

---

**Q23.** A team is training a 70B-parameter dense model across 4 nodes with 8 GPUs each (32 GPUs total, NVLink within node, InfiniBand between nodes). The model does not fit on a single GPU even with sharded optimizer states, and the team wants to minimize expensive inter-node communication. Which parallelism configuration BEST fits this hardware topology?

A) Apply data parallelism only, using ZeRO-3 across all 32 GPUs
B) Apply tensor parallelism across all 32 GPUs regardless of node boundaries
C) Apply pipeline parallelism across all 32 GPUs with a single pipeline stage per GPU and no tensor parallelism
D) Apply tensor parallelism within each 8-GPU node and pipeline parallelism across the 4 nodes, with data parallelism layered on top if replicas are needed

---

**Q24.** Which post-training weight quantization technique identifies salient weight channels based on activation magnitude statistics and protects those channels from precision loss, rather than minimizing layer-wise weight reconstruction error via a Hessian-based approach?

A) GPTQ
B) AWQ
C) Naive round-to-nearest INT8 quantization
D) FP8 dynamic per-tensor quantization

---

**Q25.** A training job enables gradient checkpointing (activation recomputation) to allow a larger batch size on GPUs with limited memory. What is the direct tradeoff of enabling this technique?

A) Reduced numerical precision of the gradients
B) Reduced GPU memory usage during the forward pass only, with no effect on backward pass memory
C) Increased training step time because discarded activations must be recomputed during the backward pass
D) Lower final model accuracy due to approximated activations

---

**Q26.** In DeepSpeed ZeRO memory optimization, which stage shards optimizer states, gradients, AND model parameters across data-parallel ranks (with parameters gathered on demand during forward/backward passes)?

A) ZeRO-3
B) ZeRO-1
C) ZeRO-2
D) ZeRO-Offload

---

**Q27.** An LLM inference service serves requests with highly variable output lengths. GPU utilization is poor because when static batching is used, GPU slots for requests that finish early sit idle until every sequence in the batch completes generation. Which technique BEST resolves this inefficiency?

A) Increase the static batch size further
B) Apply INT8 post-training quantization to the model weights
C) Increase the KV-cache allocation per request
D) Continuous (in-flight) batching, which admits new requests and evicts finished ones at each iteration without waiting for the whole batch to complete

---

**Q28.** A team wants to reduce per-token generation latency of a 70B-parameter model at inference time without altering the model architecture or degrading output quality, by leveraging a smaller, faster draft model to propose candidate tokens that the large model then verifies in parallel. Which technique should they use?

A) Pipeline parallelism
B) Weight-only INT4 quantization
C) KV-cache eviction policies
D) Speculative decoding

---

**Q29.** A team deploys a 13B parameter LLM for a customer support chatbot that receives requests with highly variable prompt and response lengths, arriving continuously throughout the day. GPU memory is being exhausted during peak traffic due to KV-cache allocation, and average request latency spikes when many requests arrive simultaneously. Which approach BEST resolves both the memory exhaustion and latency spikes?

A) Enable in-flight (continuous) batching with paged KV-cache management in TensorRT-LLM
B) Increase the batch size statically to process more requests per GPU cycle
C) Switch the model weights from FP16 to FP32 for higher numerical precision
D) Disable the KV-cache entirely and recompute attention for every generated token

---

**Q30.** An ML engineer needs to quantize a 70B parameter LLM to INT4 weight-only precision for deployment on memory-constrained GPUs. The team has a small calibration dataset and wants to preserve accuracy by protecting the weight channels that correspond to high-magnitude activations, without performing any gradient-based reconstruction of layer outputs. Which technique should be selected?

A) GPTQ, which minimizes layer-wise reconstruction error using second-order (Hessian) information
B) AWQ (Activation-aware Weight Quantization)
C) Naive round-to-nearest INT4 quantization applied uniformly with no calibration data
D) FP8 quantization using per-tensor dynamic scaling factors

---

**Q31.** A team is training a 70B parameter model across 8 GPUs with 80GB of memory each. Combined memory for parameters, gradients, and optimizer states in BF16 mixed precision still exceeds the capacity of a single GPU. Which DeepSpeed ZeRO configuration should be used to make the training run fit in memory?

A) ZeRO Stage 2, which shards gradients and optimizer states but keeps parameters fully replicated
B) ZeRO Stage 3, which shards parameters, gradients, and optimizer states across all GPUs
C) Disable ZeRO entirely and rely solely on activation gradient checkpointing
D) ZeRO Stage 1, which shards only the optimizer states across GPUs

---

**Q32.** A team serves an LLM with NVIDIA Triton Inference Server behind a Kubernetes-based NIM deployment. They need to roll out a new fine-tuned model version to production while minimizing risk of regressions, and they want the ability to automatically roll back based on live error-rate metrics without manual intervention. Which deployment strategy BEST meets these requirements?

A) Canary rollout that routes a small percentage of traffic to the new version, monitors error rate and latency via automated metrics, and auto-rolls-back on threshold breach
B) Blue-green deployment with a manual cutover after a fixed 24-hour soak period
C) Shadow deployment where the new model only logs predictions without ever serving live traffic
D) In-place replacement of the model files in the existing Triton model repository during a maintenance window

---

**Q33.** An LLM inference service on Triton Inference Server receives many concurrent small requests. GPU utilization is low despite high request volume, and per-request latency is acceptable but overall throughput is far below the GPU's theoretical capacity. Which Triton feature should be enabled FIRST to increase throughput without changing the model itself?

A) Increase the number of CPU worker threads in the client application
B) Switch the model backend from TensorRT to plain ONNX Runtime
C) Enable dynamic batching so incoming requests are grouped into larger batches before GPU execution
D) Reduce the model's maximum sequence length to shrink memory footprint

---

**Q34.** In a Triton model repository configuration (config.pbtxt), which setting controls how many parallel model instances are loaded on a GPU to allow multiple inference requests to execute concurrently on that device?

A) max_batch_size
B) instance_group with count and KIND_GPU
C) dynamic_batching preferred_batch_size
D) optimization.execution_accelerators

---

**Q35.** An LLM service must meet a strict p99 latency SLA during traffic spikes. The current Kubernetes Horizontal Pod Autoscaler scales replicas based on average CPU utilization, but the team observes SLA violations before new GPU pods finish initializing and become ready. Which change would BEST address this issue?

A) Increase the CPU utilization scale-up threshold so pods scale more aggressively
B) Disable autoscaling and provision a single very large GPU instance
C) Switch the autoscaling metric to inference request queue depth or GPU utilization and lower the scale-up threshold with pre-warmed pod buffers
D) Increase the readiness probe timeout so pods are marked ready sooner

---

**Q36.** When increasing the inference batch size for an LLM served on GPUs, what is the typical effect on latency and throughput?

A) Per-request latency increases while overall throughput increases, up to a saturation point
B) Both per-request latency and overall throughput decrease
C) Per-request latency decreases while overall throughput decreases
D) Batch size has no measurable effect on either latency or throughput for transformer models

---

**Q37.** A production LLM's monitoring dashboard shows a gradual increase in the rate of low-confidence or out-of-distribution responses over several weeks, even though no code or infrastructure changes were made. Which MLOps practice should be triggered to address this situation?

A) Immediately roll back to the previous container image version
B) Scale up the number of Triton model instances to handle more traffic
C) Increase the dynamic batching window to reduce GPU idle time
D) Trigger drift detection analysis and, if confirmed, initiate a retraining or fine-tuning cycle with recent data

---

**Q38.** A team has fine-tuned a new version of their production LLM and wants to validate its quality on real user traffic before fully replacing the current version, while minimizing the risk of exposing all users to potential regressions. Which deployment strategy BEST meets this requirement?

A) Blue-green deployment that atomically switches 100% of traffic to the new version once offline evaluation passes
B) Canary rollout that routes a small percentage of live traffic to the new version while monitoring key metrics, then gradually increasing traffic
C) Shadow deployment that duplicates incoming requests to the new version without returning its responses to any real users
D) In-place replacement of the model artifacts on all serving instances simultaneously

---

**Q39.** You are serving three different NIM-packaged models on a single Triton Inference Server instance backed by two GPUs. One model receives significantly more traffic than the others; its requests show high queue times even though GPU compute utilization for that model stays low. Which configuration change should you make to increase throughput for that specific model without affecting the other two models?

A) Increase only that model's max_batch_size setting in its model configuration file
B) Increase the 'count' value in that model's instance_group configuration to run multiple concurrent model instances on the GPU
C) Enable model warmup for every model in the repository
D) Increase the number of Kubernetes replicas for the entire Triton deployment

---

**Q40.** A team fine-tuned an LLM for a medical Q&A chatbot. During testing, the model confidently generates plausible-sounding drug dosages that are not supported by any retrieved source document. Which evaluation approach BEST detects this specific failure mode before production deployment?

A) Measure BLEU score against reference answers
B) Measure inference latency under load
C) Run a hallucination-detection evaluation that cross-checks generated claims against retrieved source documents
D) Run a toxicity classifier on the model outputs

---

**Q41.** An organization needs to evaluate the quality of open-ended, creative summaries generated by an LLM at scale, where no single reference answer exists and nuanced judgment (coherence, relevance, style) is required. Which evaluation method is MOST appropriate for this requirement?

A) Exact-match accuracy against a gold-label dataset
B) ROUGE score computed against a single reference summary
C) Perplexity measured on a held-out validation set
D) LLM-as-judge evaluation using a strong reference model with a scoring rubric

---

**Q42.** You are configuring NVIDIA NeMo Guardrails for a customer-support bot and need to prevent the model from discussing competitor pricing, regardless of how the user phrases the request. Which guardrail configuration mechanism should you use?

A) Increase the model's temperature setting in the generation config
B) Add more few-shot examples to the system prompt only
C) Define a dialogue rail with topical flows in the Colang config that detects and blocks off-topic user intents
D) Apply output length truncation in the inference server config

---

**Q43.** Which statement correctly distinguishes hallucination from bias as responsible-AI failure modes in LLM evaluation?

A) Hallucination and bias are the same failure mode and are evaluated using identical metrics
B) Bias only occurs during pretraining and cannot be detected at inference time
C) Hallucination is a subset of toxicity and is measured using toxicity classifiers
D) Hallucination refers to factually incorrect or unsupported generated content, while bias refers to systematic skew in outputs favoring or disadvantaging particular groups or viewpoints

---

**Q44.** An enterprise LLM deployment must comply with a policy requiring that no customer PII (names, SSNs, phone numbers) appear in logged model outputs or training data used for continued fine-tuning. Which responsible-AI practice BEST addresses this requirement across the model lifecycle?

A) Rely solely on the base model's built-in safety alignment from pretraining
B) Implement PII detection and redaction pipelines at data ingestion, logging, and output stages, combined with data governance policies
C) Increase the guardrails' response latency threshold to slow down risky requests
D) Apply a higher sampling temperature to reduce the chance of PII appearing in outputs

---

**Q45.** A team evaluating a newly fine-tuned LLM finds that automated benchmark scores (accuracy, F1) are high, but real users report the responses feel unhelpful and awkwardly phrased in production. Which evaluation step should the team have included earlier in the process to catch this gap?

A) Re-run the same automated benchmark with a larger test set
B) Only measure token-level perplexity on the training set
C) Structured human evaluation with rubric-based ratings on helpfulness and fluency, in addition to automated metrics
D) Increase the number of fine-tuning epochs and re-check the same benchmark

---

