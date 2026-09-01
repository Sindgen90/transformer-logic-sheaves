# Transformer Logic Sheaves

A controlled test of the hypothesis:

> Compositional generalization emerges when a neural network's locally decodable logical representations become globally coherent under semantics-preserving transformations.

This repository starts with the cheapest experiment that can disprove the idea. Tiny Transformers evaluate prefix-serialized Boolean expressions. Models trained on low- and higher-diversity depth-3 datasets are compared on deeper trees, local linear probes, ordinary representation similarity, and a held-out De Morgan/commutativity cycle score.

This is **not yet a sheaf implementation**. The point of Experiment 0 is to establish whether global consistency contains information beyond probes and pairwise similarity. See [the experiment design](docs/experiment_design.md) for the exact claim and stop criteria.

## Quick start

Python 3.10+ and PyTorch are required. For this machine's CUDA 12.6 driver, create
the tested Conda environment from Anaconda Prompt:

```powershell
conda create -n logic-sheaves python=3.12 pip -y
conda activate logic-sheaves
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[plots,dev]"
python -m unittest discover -s tests
python -m logic_sheaves smoke --device cuda --output runs\smoke
```

For the first useful multi-seed pilot on a GPU:

```powershell
python -m logic_sheaves pilot --device cuda --output runs\pilot
```

For the complete width-128 architecture-depth sweep with training dynamics,
activation patching, and plots A–J:

```powershell
python -m logic_sheaves depth-sweep --device cuda --output-root runs\depth_sweeps
```

Every invocation creates a new timestamped subdirectory and never overwrites an
earlier run. See [the depth-sweep protocol](docs/depth_sweep.md) for the exact
controls, patching intervention, output tables, and figure definitions.

Override the training budget or seeds without editing code:

```powershell
python -m logic_sheaves pilot --steps 5000 --seeds 0 1 2 3 4
```

Each run writes:

- `config.json`: exact experiment configuration;
- `checkpoints/*.pt` and matching histories;
- `results.csv` and `results.json`: behavioral and representation metrics;
- `summary.png`: OOD accuracy against holonomy and local-probe scores.

## What the metrics mean

- `local_probe_*`: balanced accuracy for decoding each operator subtree's truth value from its operator-token state on deeper OOD trees.
- `cycle_identity_energy`: raw disagreement among representations of equivalent forms.
- `cycle_transport_error`: held-out residual after a constrained orthogonal alignment for each rewrite.
- `cycle_holonomy_error`: failure to return to the starting representation after composing all four fitted rewrite transports.
- `cycle_edge_cka`: ordinary linear representational similarity baseline.

Cycle errors are normalized by activation variance. Lower is more coherent. A compelling result requires cycle measures to explain generalization beyond ID accuracy, local probes, CKA, and pairwise energy across many seeds—not merely two attractive points in a smoke plot.

The current smoke-test measurements and their limitations are recorded in
[initial results](docs/initial_results.md).

## Repository layout

```text
src/logic_sheaves/
  logic.py       Boolean ASTs, evaluation, and equivalence cycles
  data.py        deterministic ID/OOD generation and tokenization
  model.py       small Transformer encoder
  training.py    reproducible training and evaluation
  metrics.py     probes, CKA, constrained transports, and holonomy
  experiment.py  low-vs-high-diversity experiment runner
  depth_sweep.py  1–6 layer sweep and checkpoint trajectory analysis
  patching.py     controlled counterfactual activation patching
  plotting.py     automatic A–J research figure suite
tests/           logic, data, model, and metric checks
docs/            experimental rationale and falsification criteria
```

## Immediate roadmap

1. Run the two-condition smoke test and verify both regimes can fit their training data.
2. Tune the task—not the coherence metric—until independent runs span low to high OOD accuracy.
3. Run at least 20 models across seeds, depth curricula, and model sizes.
4. Fit preregistered regressions comparing cycle metrics with probes, CKA, pairwise energy, ID accuracy, and train size.
5. Replicate with associativity and double-negation cycles.
6. Only then formalize a cellular sheaf/discrete connection and add causal interventions.
