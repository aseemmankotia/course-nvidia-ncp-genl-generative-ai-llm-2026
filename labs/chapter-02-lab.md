# Chapter 2 Lab: Scaling Laws, Tokenization, and Context-Window Mechanics

**Estimated time:** 30 minutes (two parts of 15 minutes)

## Part A — `chapter-02a-lab.ipynb`: BPE and out-of-vocabulary words

Build a byte-pair-encoding tokenizer from a tiny corpus, inspect the learned merges, and encode an out-of-domain word. Optional cell: compare with a production BPE vocabulary via `tiktoken`.

## Part B — `chapter-02b-lab.ipynb`: SentencePiece-style unigram tokenization and context-window arithmetic

Segment the same words with a unigram (SentencePiece-style) tokenizer and compare the splits and token counts with BPE. Optional cell: train a real `sentencepiece` model. Then turn token counts into numbers: which documents fit in an 8K / 32K / 128K window, what a day of requests costs, and how a compute-optimal (Chinchilla-style) budget splits between parameters and tokens.

Both parts run on the Python standard library — no API key, no GPU, no installs.

## Key takeaway

Scaling laws and tokenization choices determine training efficiency and effective context length—both are cost levers the exam expects you to reason about quantitatively.
