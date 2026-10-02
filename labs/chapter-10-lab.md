# Chapter 10 Lab: Autoscaling, Rollouts, and LLM MLOps Lifecycle Monitoring

**Estimated time:** 20 minutes

## Exercise

Configure a Kubernetes Horizontal Pod Autoscaler policy (read-and-modify YAML) driven by custom GPU-utilization metrics for a Triton deployment, and simulate a canary rollout traffic split

## Key takeaway

Effective LLM serving operations require GPU-aware autoscaling signals and staged rollout strategies (canary/blue-green) matched to the risk level of the change.
