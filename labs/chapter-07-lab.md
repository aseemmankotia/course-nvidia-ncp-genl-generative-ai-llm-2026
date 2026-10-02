# Chapter 7 Lab: Distributed Training: Data, Tensor, Pipeline, and Sequence Parallelism

**Estimated time:** 25 minutes

## Exercise

Read and configure a Megatron-style parallelism config (tensor_parallel_size, pipeline_parallel_size, data_parallel_size) for a multi-node cluster, computing effective GPU count and batch size

## Key takeaway

Parallelism strategy selection depends on where the bottleneck lives—layer size, sequence length, or GPU count—and interconnect speed dictates which strategy goes within vs across nodes.
