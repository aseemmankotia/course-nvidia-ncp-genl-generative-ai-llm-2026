## Chapter 5: Dataset Curation, Cleaning, and Instruction-Tuning Data Design — Practice Questions

**Q1.** A team is deduplicating a 2TB web-crawl corpus before pretraining. They need to catch documents that are 85% similar (boilerplate reused across sites with minor edits) without doing pairwise comparison across billions of documents. Which approach BEST fits this requirement?

A) Exact match deduplication using SHA-256 hashes of full documents
B) MinHash signatures generated from document shingles, bucketed via Locality Sensitive Hashing (LSH)
C) Sorting documents alphabetically and removing adjacent duplicates
D) A simple string-length filter to remove documents under 100 tokens

<details>
<summary>Answer</summary>

**Correct: B**

MinHash approximates Jaccard similarity between shingle sets cheaply, and LSH buckets similar signatures together so only candidate pairs within the same bucket are compared, avoiding O(n²) pairwise comparison at web scale.

**Why the others are wrong:**
- Exact hashing only catches byte-identical documents, missing near-duplicates with minor edits.
- Alphabetical sorting does not group semantically/textually similar documents that differ in wording or ordering.
- Length filtering removes short documents but does nothing to detect similarity between long documents.

</details>

---

**Q2.** During fuzzy deduplication, an engineer increases the number of bands (b) in the LSH banding scheme while keeping the total number of MinHash hash functions fixed, which decreases the number of rows per band (r). What is the effect on the deduplication result?

A) Recall of near-duplicate pairs decreases and precision increases
B) Both recall and precision remain unchanged because b×r is constant
C) The corpus size doubles because more buckets are created
D) Recall of near-duplicate pairs increases and precision decreases

<details>
<summary>Answer</summary>

**Correct: D**

More bands with fewer rows each lowers the similarity threshold needed for two signatures to collide in at least one band, so more pairs get flagged as candidates — increasing recall but also increasing false positives, lowering precision.

**Why the others are wrong:**
- This describes the opposite effect (fewer bands, more rows per band increases precision, decreases recall).
- b×r being constant does not mean the b/r split is irrelevant — the split changes the collision probability curve.
- LSH banding is a candidate-pair generation technique and does not alter corpus size.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** Which of the following is a true statement about exact vs. near-duplicate deduplication?

A) Exact deduplication requires a similarity threshold parameter such as Jaccard similarity
B) Exact deduplication identifies byte-identical (or normalized-identical) documents, while near-duplicate deduplication identifies documents with high but partial textual overlap
C) Near-duplicate deduplication relies on hashing the full document byte-for-byte
D) Exact and near-duplicate deduplication produce identical results on any corpus

<details>
<summary>Answer</summary>

**Correct: B**

Exact dedup matches identical (or lightly normalized, e.g. whitespace-stripped) content, while near-duplicate/fuzzy dedup uses similarity estimation (MinHash/Jaccard) to catch documents that overlap substantially but are not identical.

**Why the others are wrong:**
- A similarity threshold is a near-duplicate concept, not exact matching.
- Full-document byte hashing is the exact-dedup technique, not the near-duplicate technique.
- The two methods catch different classes of duplicates, so results diverge on real corpora — near-dedup always removes a superset that includes exact duplicates plus more.

</details>

---

**Q4.** A team is building an instruction-tuning dataset for a customer-support assistant. Some examples include a system prompt defining tone and scope, others do not, and formatting of the prompt-response pairs is inconsistent across sources merged from three different vendors. What is the MOST important step before fine-tuning?

A) Increase the learning rate to help the model generalize across inconsistent formats
B) Discard all examples that include a system prompt since the base model was not trained with one
C) Normalize all examples to a single consistent prompt template (including system prompt placement) and remove or standardize outliers
D) Train directly on the mixed data since diverse formatting improves robustness

<details>
<summary>Answer</summary>

**Correct: C**

Inconsistent formatting during instruction tuning teaches the model conflicting prompt/response structures, degrading instruction-following at inference; standardizing to one template (system/user/assistant roles, delimiters) is required before training.

**Why the others are wrong:**
- Learning rate changes do not fix structural format inconsistency in training data.
- Discarding system-prompted examples throws away useful data and doesn't address the actual format-consistency problem.
- Diverse formatting does not improve robustness — it teaches contradictory templates and hurts convergence and instruction adherence.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** A fine-tuning job initializes from a pretrained checkpoint but the fine-tuning pipeline uses a tokenizer with a different vocabulary and merge rules than the one used during pretraining. What is the MOST likely consequence?

A) Training will proceed identically because tokenizers only affect inference speed
B) The model will automatically re-align the new tokenizer to the old embedding table
C) This only affects multilingual models and has no impact on English-only fine-tuning
D) Token IDs will map to different subwords than the model learned during pretraining, corrupting the embedding lookup and degrading model quality

<details>
<summary>Answer</summary>

**Correct: D**

The embedding matrix is indexed by token ID learned during pretraining; a mismatched tokenizer maps different text to the same IDs (or requires resizing), breaking the learned association between token ID and meaning and severely degrading performance.

**Why the others are wrong:**
- Tokenization directly determines what the model sees as input; it is not merely a speed concern.
- There is no automatic re-alignment — mismatched vocab/embedding size requires manual embedding resizing or retraining of new tokens.
- The tokenizer/embedding mismatch problem applies to any language, not just multilingual settings.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q6.** Which combination of data governance actions is required before releasing a fine-tuned LLM trained partly on scraped web text and third-party datasets?

A) Verify dataset licenses permit the intended use, and remove or redact detected PII from training data
B) Only check that the data is publicly accessible on the internet
C) Apply exact deduplication only, since licensing is not a data preparation concern
D) Convert all data to lowercase to anonymize personal information

<details>
<summary>Answer</summary>

**Correct: A**

Data governance for LLM training requires confirming license terms (e.g., commercial-use permission, attribution, no-derivatives clauses) and removing/redacting PII to reduce legal and privacy risk — both are standard curation requirements.

**Why the others are wrong:**
- Public accessibility does not imply a permissive license for training/commercial use — scraping legality and copyright still apply.
- Licensing compliance is a governance requirement independent of deduplication and cannot be skipped.
- Lowercasing text does not remove or anonymize personally identifiable information.

</details>

---

**Q7.** A curation pipeline uses a fastText-based classifier to score documents and drop the lowest-scoring 20% as low quality, in addition to heuristic rules (removing documents with excessive symbol-to-word ratio or repeated n-grams). This is an example of which data cleaning strategy category?

A) Deduplication
B) Tokenization normalization
C) Combined classifier-based and heuristic-based quality filtering
D) Data mixing

<details>
<summary>Answer</summary>

**Correct: C**

Using a trained classifier score (fastText) alongside rule-based heuristics (symbol ratio, repetition) to filter documents is the standard two-pronged quality-filtering approach used in LLM data curation pipelines such as NeMo Curator.

**Why the others are wrong:**
- Deduplication removes redundant documents, not low-quality ones based on content quality signals.
- Tokenization normalization concerns how text is split into tokens, not document-level quality scoring.
- Data mixing concerns the proportion of different data sources sampled, not filtering individual documents by quality.

</details>

---

**Q8.** A team pretrains a model with a data mix of 70% web text, 15% code, 10% books, and 5% academic papers. After fine-tuning on an instruction dataset with a very different domain distribution than pretraining, the model's coding ability noticeably degrades. What is the MOST likely cause?

A) The tokenizer changed vocabulary size during fine-tuning
B) MinHash deduplication removed all code samples from the fine-tuning set
C) The model was fine-tuned with too low a learning rate, preventing any learning at all
D) Catastrophic forgetting due to the fine-tuning data mix under-representing code relative to the pretraining distribution

<details>
<summary>Answer</summary>

**Correct: D**

When fine-tuning data mixing ratios diverge sharply from pretraining ratios (e.g., little to no code), the model's parameters shift away from previously learned code capability — a classic catastrophic forgetting effect driven by data mix imbalance.

**Why the others are wrong:**
- A tokenizer vocabulary change is a separate, unrelated failure mode not implied by this scenario.
- Deduplication removes redundant docs; it isn't described nor would it selectively strip an entire domain.
- A too-low learning rate would cause under-fitting broadly, not domain-specific skill degradation.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

