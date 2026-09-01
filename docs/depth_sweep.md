# Width-128 architecture-depth sweep

## Scope

The `depth-sweep` command compares Transformer depths 1 through 6 at a fixed
width of 128. For every depth it trains low- and higher-diversity conditions
across three seeds by default, for 36 independently trained models total.

The shared controls are:

- width 128, 4 attention heads, feed-forward width 256;
- train expressions of depth at most 3;
- exact-depth OOD tests at depths 4, 5, and 6;
- low-diversity training size 256;
- higher-diversity training size 8,000;
- 3,000 AdamW updates and batch size 256;
- seeds 0, 1, and 2;
- the same train pool, validation/test sets, equivalence cycles, and patching
  pairs for every trained model.

## Training dynamics

At step 1, every 500 updates, and the final update, the runner records:

- train and validation losses and validation accuracy;
- ID and per-depth OOD accuracy;
- operator-specific local probe accuracy;
- identity, orthogonal-transport, holonomy, CKA, and activation-variance metrics;
- activation-patching summaries.

Trajectory analyses use fixed smaller evaluation subsets so repeated
measurement remains cheap. Final metrics use the full evaluation, cycle, and
patching suites.

## Activation patching

Each recipient has the controlled form

```text
XOR(target_subtree, context)
```

The target subtree operator is one of `NOT`, `AND`, `OR`, or `XOR` and is always
at token position 2 after `[CLS]` and the root `XOR`.

Two donors are constructed:

1. A **counterfactual donor** uses the same target operator but the opposite
   target truth value. Because the root is XOR and the context is held fixed,
   the correct final output must flip.
2. An **equivalent donor** uses independently generated syntax with the same
   target truth value, so the correct final output must stay unchanged.

The target operator state is patched from each donor into the recipient after
the embedding stage and after every Transformer layer. The main measurements
are:

- counterfactual target accuracy;
- prediction flip rate;
- counterfactual logit-effect fraction;
- same-truth prediction preservation;
- same-truth ground-truth accuracy.

Stage 0 and the final layer are negative controls. At stage 0 the operator
embedding is identical between donor and recipient. After the final layer a
patched operator state has no remaining path to the `[CLS]` classifier. A
meaningful causal semantic signal should therefore appear at intermediate
stages.

For one-layer models no such intermediate stage exists, so aggregate "best
causal patch" fields are recorded as missing rather than manufacturing a score.

## Output layout

Every command invocation creates a new UTC-timestamped directory:

```text
runs/depth_sweeps/depth_sweep_YYYYMMDD_HHMMSS_microsecondsZ/
  config.json
  status.json
  checkpoints/
  trajectories/
  tables/
    results.csv
    dynamics.csv
    patching_final.csv
    patching_dynamics.csv
    ... matching JSON files
  plots/
    A_training_curves.png
    B_accuracy_by_expression_depth.png
    C_train_id_ood_bars.png
    D_coherence_vs_generalization.png
    E_baseline_comparison.png
    F_local_vs_global.png
    G_operator_probes.png
    H_correlation_matrix.png
    I_activation_patching.png
    J_training_dynamics.png
```

Tables are rewritten after each completed model, so completed work survives an
interrupted long sweep. `status.json` changes from `running` to `complete` only
after all plots are generated.

## Figures

- **A:** train loss, validation loss, and validation accuracy over updates.
- **B:** accuracy versus expression depth.
- **C:** grouped train, ID, and mean OOD accuracy.
- **D:** holonomy error versus OOD generalization.
- **E:** identity, transport, holonomy, and CKA baseline comparison.
- **F:** local decodability versus global coherence, colored by OOD accuracy.
- **G:** operator-specific OOD probes.
- **H:** model-level Pearson correlation matrix.
- **I:** counterfactual transfer and equivalent-preservation patching curves.
- **J:** behavioral, representational, and causal training dynamics together.

The unit of evidence in figures D–H is a trained model, not an individual
expression. Expression-level repetitions must not be treated as independent
samples when reporting uncertainty.
