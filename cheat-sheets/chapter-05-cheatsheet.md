# Chapter 5 Cheat Sheet: Dataset Curation & Instruction-Tuning Data

## Deduplication: Exact vs Near-Duplicate
| Method | Technique | Catches | Cost |
|---|---|---|---|
| Exact | Hash full doc (SHA-256/MD5) | Byte-identical / normalized-identical copies | Cheap, O(n) |
| Near-duplicate (fuzzy) | Shingling → MinHash signature → LSH banding | High-overlap, non-identical docs | Moderate, avoids O(n²) via LSH buckets |

**MinHash**: hash-permutation signatures approximate Jaccard similarity of shingle sets.
**LSH banding**: split signature into `b` bands of `r` rows; docs colliding in ≥1 band become candidate pairs.
- More bands (smaller `r`) → higher recall, lower precision.
- Fewer bands (larger `r`) → higher precision, lower recall.

## Quality Filtering
| Type | Examples |
|---|---|
| Heuristic | symbol-to-word ratio, repeated n-grams, doc length, language ID |
| Classifier-based | fastText/quality classifiers scoring documents |

Best practice: combine both stages; heuristics catch cheap/obvious junk, classifiers catch subtler low-quality text.

## Instruction-Tuning Data Formats
| Format | Structure |
|---|---|
| Alpaca-style | `instruction` / `input` (optional) / `output` |
| ChatML-style | Role-tagged turns: `system`, `user`, `assistant` |

**Rule**: Pick ONE template and apply it consistently across the whole dataset (system prompt placement, delimiters). Mixed formats confuse instruction-following at inference.

## Tokenization Consistency
- Fine-tuning tokenizer MUST match pretraining tokenizer's vocab + merges — token IDs index the pretrained embedding table.
- New special tokens → resize embedding/output layer before training.
- Mismatched tokenizer = silently corrupted embeddings, not a crash — a common exam trap.

## Data Mixing
- Fine-tuning data distribution should roughly preserve important pretraining domain coverage (e.g., code, multilingual) to avoid catastrophic forgetting.
- Upsampling high-quality/rare domains and downsampling redundant boilerplate-heavy web text is standard practice.

## Data Governance & Licensing
- Confirm license terms (commercial-use, attribution, share-alike/no-derivative clauses) for every data source, including scraped web data.
- Public availability ≠ permission to train/redistribute.
- Detect and redact/remove PII before training or release.
- Maintain data provenance documentation for audit/compliance.

## Common Exam Traps
- Confusing MinHash (similarity estimation) with exact hashing (identity check).
- Believing more LSH bands only affects speed — it actually shifts the recall/precision tradeoff.
- Assuming mismatched tokenizers only slow training rather than corrupt embeddings.
- Assuming diverse/inconsistent instruction formats improve robustness (they don't — they hurt instruction-following).
- Treating deduplication and quality filtering as the same step (they are separate pipeline stages).
