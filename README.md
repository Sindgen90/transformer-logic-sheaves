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

For the second-phase symbolic experiment with variables, n-ary and fixed-arity
operators, arbitrary equivalence diagrams, competing paths, and plots A–K:

```powershell
python -m logic_sheaves complex-sweep --device cuda --output-root runs\equivalence_complexes
```

This evaluates eleven held-out diagram families, including involutions, De Morgan
squares, an associativity pentagon, permutation hexagons, a commutativity cube,
and a distributivity diamond. Rewrite transports are fitted on isolated pairs;
complete test loops and alternative-path topologies remain held out. See the
[equivalence-complex protocol](docs/equivalence_complex.md). A completed 18-model
reference run, including all figures and machine-readable tables, is available in
[the extended experiment report](runs/equivalence_complexes/equivalence_complex_20260902_211352_321084Z/report.md).

To test whether the fitted connection's loop closure is meaningful rather than a
degenerate cancellation, audit any completed run against identity and shuffled
connections:

```powershell
python -m logic_sheaves holonomy-audit runs\equivalence_complexes\RUN_NAME --device cuda
```

The reference run's [holonomy audit](runs/equivalence_complexes/equivalence_complex_20260902_211352_321084Z/holonomy_audit/report.md)
shows why closure must be reported together with held-out edge fidelity, operator
defect, affine drift, and competing-path endpoint error.

For layerwise attention localization, fit balanced rewrite transports independently
to queries, keys, and values at `<CLS>` and across content tokens:

```powershell
python -m logic_sheaves qkv-holonomy runs\equivalence_complexes\RUN_NAME --device cuda
```

The completed reference [Q/K/V holonomy report](runs/equivalence_complexes/equivalence_complex_20260902_211352_321084Z/qkv_holonomy/report.md)
contains joint-head and per-head results for every available layer.

Run exact causal pre-attention Q/K/V patching, followed by the
context-conditioned transport and equivalent-path causal tests:

```powershell
python -m logic_sheaves qkv-patching runs\equivalence_complexes\RUN_NAME --device cuda
python -m logic_sheaves contextual-holonomy runs\equivalence_complexes\RUN_NAME --device cuda --calibration-pairs 96
python -m logic_sheaves path-patching runs\equivalence_complexes\RUN_NAME --device cuda
```

These commands write separate `qkv_patching`, `contextual_holonomy`, and
`path_patching` directories inside the selected run. The contextual analysis
reports local and common-`<CLS>` fibers, separate and coupled Q/K transport,
identity/target-shuffled/label-shuffled/rank-8 controls, operator eigenphases,
and Q/K geometry for every layer and head.

Fit the paper-style local-chart gauge connection and compare it directly with
held-out state-return error:

```powershell
python -m logic_sheaves gauge-atlas runs\equivalence_complexes\RUN_NAME --device cuda --bit-flips x0 x1
```

Run the corrected typed-loop analysis—separate `Q` and `P` loop transports,
`H_rel = H_P^-1 H_Q`, frozen discovery/confirmation circuits, persistence and
bootstrap gates, same-mask nulls, and a paired variable intervention:

```powershell
python -m logic_sheaves typed-gauge runs\equivalence_complexes\RUN_NAME --device cuda
```

The legacy `gauge-atlas` command accepts `--heads 1 3` for an explicit head
restriction. `typed-gauge` instead freezes head sets automatically from the
discovery/confirmation patching split and includes their complements and all
heads as localization controls. Every invocation gets a new timestamped folder.
The exact definitions, normalization correction, circuit protocol, and
acceptance criteria are in the [gauge-atlas design](docs/gauge_atlas.md).
The first corrected run and its caveats are summarized in the
[typed-gauge results](docs/typed_gauge_results_20260912.md).

Compare global, family-global, and vertex-local charts on balanced correct and
incorrect predictions, including shared-connection state return, connection-sheaf
section energy, and the sheaf-Laplacian spectrum:

```powershell
python -m logic_sheaves local-global runs\equivalence_complexes\RUN_NAME --device cuda
```

The first results are summarized in the
[local/global correctness report](docs/local_global_correctness_results_20260913.md).

Run the confirmatory six-step protocol on entirely new diagrams: fixed
family-global charts, per-example correctness prediction with controls,
higher-diversity leave-one-seed-out validation, a frozen low-diversity transfer,
and discovery/confirmation Q/K/V path circuits with rank-matched controls:

```powershell
python -m logic_sheaves confirmatory-sheaf runs\equivalence_complexes\RUN_NAME --device cuda --diagrams-per-family 128 --chart-dimension 8 --circuit-rank 4
```

The completed run is summarized in the
[confirmatory sheaf and circuit report](docs/confirmatory_sheaf_results_20260913.md).

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
  equivalence.py  arbitrary logical diagrams, paths, loops, and rewrite suites
  diagram_metrics.py  generalized transport, path agreement, and holonomy
  complex_experiment.py  symbolic multi-diagram experiment runner
  complex_plotting.py  automatic figures for the extended experiment
  holonomy_audit.py  null connections and operator-level holonomy diagnostics
  qkv_holonomy.py  balanced layerwise and per-head Q/K/V localization
  qkv_patching.py  exact causal pre-attention Q/K/V interventions
  contextual_holonomy.py  bidirectional context-conditioned connections and geometry
  path_patching.py  causal comparison of equivalent rewrite routes
  gauge_atlas.py  chart, connection, gauge, and fundamental-cycle primitives
  gauge_experiment.py  layerwise Q/K/V atlas, controls, circuit filters, and bit flips
  circuit_selection.py  discovery/confirmation causal circuit nominations
  typed_gauge_experiment.py  type-correct loop holonomy and persistence analysis
  local_global_experiment.py  chart-scale, correctness, and connection-sheaf analysis
  confirmatory_experiment.py  disjoint correctness prediction and targeted path circuits
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
