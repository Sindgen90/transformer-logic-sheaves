# Initial smoke results

Date: 2026-09-01

These runs validate the pipeline; they do not support the research hypothesis yet.

## GPU smoke run

Environment: PyTorch 2.6.0+cu126, NVIDIA RTX 4090. Both conditions used a one-layer, 32-dimensional Transformer for 300 updates. The evaluation sets and initialization seed were shared.

| condition | unique train expressions | train accuracy | ID accuracy | mean depth-4/5 OOD accuracy | local probe | cycle holonomy error |
|---|---:|---:|---:|---:|---:|---:|
| low diversity | 96 | 1.000 | 0.651 | 0.594 | 0.496 | 1.216 |
| higher diversity | 768 | 0.678 | 0.609 | 0.615 | 0.508 | 0.404 |

The low-diversity condition is a successful memorization control: it fits every training expression but generalizes only weakly. The higher-diversity condition has not converged, so its lower holonomy error cannot be interpreted as evidence for coherence-driven generalization.

## Extended one-layer run

The same architecture was trained for 2,000 CPU updates as a cheap diagnostic.

| condition | train accuracy | ID accuracy | mean depth-4/5 OOD accuracy | cycle holonomy error |
|---|---:|---:|---:|---:|
| low diversity | 1.000 | 0.635 | 0.586 | 1.024 |
| higher diversity | 0.880 | 0.568 | 0.581 | 1.173 |

More optimization did not produce compositional generalization, and the cycle metric did not separate the conditions in the desired direction. This is useful negative evidence: the one-layer configuration is not an adequate scientific comparison.

## Next experiment

Run the three-layer pilot across multiple seeds and first establish a behavioral spread with high training accuracy. Representation metrics should remain uninterpreted until both regimes fit their training data. If the pilot still fails to generalize, redesign the task/model interface—likely with a causal or stack-oriented serialization—before adding sheaf machinery.
