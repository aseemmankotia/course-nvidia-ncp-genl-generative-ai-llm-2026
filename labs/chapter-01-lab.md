# Chapter 1 Lab: Transformer Internals: Attention, Positional Encoding, and Efficiency Variants

**Estimated time:** 20 minutes

## Exercise

Trace attention computation shapes and KV-cache size for a 7B decoder-only model with GQA vs MHA using a provided Python notebook (read-and-run, no training)

## Key takeaway

Architecture choices like GQA, FlashAttention, and MoE each target a different bottleneck—memory, bandwidth, or compute—and the exam tests knowing which is which.
