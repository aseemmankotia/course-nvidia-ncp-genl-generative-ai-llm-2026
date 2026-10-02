# Chapter 8 Lab: Inference Acceleration: TensorRT-LLM, Quantization, and Speculative Decoding

**Estimated time:** 30 minutes

## Exercise

Build a TensorRT-LLM engine for a quantized (INT8) 13B model checkpoint following a provided build script, then benchmark throughput vs the unquantized baseline

## Key takeaway

Inference optimization is about matching quantization precision and batching strategy to the GPU architecture and SLA—there is no universally best choice.
