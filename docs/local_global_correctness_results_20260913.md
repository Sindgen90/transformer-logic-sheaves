# Global, family-global, and local charts on correct versus incorrect predictions

**Reference model run:** `equivalence_complex_20260902_211352_321084Z`  
**Analysis run:** `local_global_20260912_214346_580591Z`  
**Models:** three width-128, six-layer, higher-diversity checkpoints  
**Chart dimension:** 8

## Question and controls

This experiment tests whether logical representations require independent local charts, a
single global chart, or an intermediate family-level atlas, and whether representation
consistency differs when the model predicts the final Boolean value correctly.

Correctness is defined by the prediction on canonical vertex 0 of each equivalence diagram.
Within each model and logical family, correct and incorrect samples are matched to contain
equal numbers of true-0 and true-1 expressions. Separately fitted correct and incorrect
connections use the intersection of retained edges. A second, cleaner state-level comparison
fits one connection on the combined balanced calibration data and evaluates correct and
incorrect held-out states through exactly the same charts, transports, and graph.

The chart conditions are:

- **global:** one PCA mean and basis shared across all families and expression vertices;
- **family-global:** one PCA chart for each logical family, shared by its vertices;
- **vertex-local:** an independent PCA chart at each expression vertex.

For an edge `u -> v`, the connection sheaf uses edge stalk `F_v` with restrictions

\[
\rho_{u\to e}=Q_{v\leftarrow u},\qquad \rho_{v\to e}=I.
\]

For a held-out collection of vertex states `s`, its normalized section energy is based on

\[
\|\delta s\|^2=\sum_{e=(u,v)}\|Q_{v\leftarrow u}z_u-z_v\|^2.
\]

The sheaf Laplacian is `L_F = delta^T delta`. Its smallest eigenvalue measures the obstruction
to a perfectly consistent global section. This is kept separate from type-correct loop
holonomy and from affine state-return error.

## Aggregate results

| Chart | Correct H_rel | Incorrect H_rel | Gap | Correct shared return | Incorrect shared return | Correct section energy | Incorrect section energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| global | 0.4337 | 0.4424 | +0.0087 | 1.6279 | 1.8223 | 0.9933 | 1.0812 |
| family-global | **0.3497** | **0.3588** | +0.0091 | 0.8352 | 0.9099 | **0.6516** | **0.6798** |
| vertex-local | 0.4519 | 0.4662 | +0.0143 | **0.7800** | **0.8386** | 0.9055 | 0.9824 |

Incorrect-minus-correct gaps by model seed were:

| Chart | Seed | Operator H_rel | Shared return | Shared section energy |
|---|---:|---:|---:|---:|
| global | 0 | -0.0006 | +0.1542 | +0.1127 |
| global | 1 | +0.0078 | +0.3066 | +0.0567 |
| global | 2 | +0.0163 | +0.1732 | +0.0880 |
| family-global | 0 | +0.0062 | +0.0548 | +0.0356 |
| family-global | 1 | +0.0127 | +0.1468 | +0.0263 |
| family-global | 2 | +0.0100 | +0.0521 | +0.0228 |
| vertex-local | 0 | +0.0214 | +0.0726 | +0.1159 |
| vertex-local | 1 | +0.0135 | +0.0604 | +0.0847 |
| vertex-local | 2 | +0.0089 | +0.0455 | +0.0352 |

## Findings

The clearest result is state-level rather than operator-level. Incorrect examples have higher
shared-connection loop-return error and higher sheaf section energy under every chart scale in
all three seeds. Separately fitted operator holonomy is also higher in eight of nine
chart-by-seed comparisons, but the mean difference is small relative to the absolute
holonomy. We should therefore describe correctness as associated with how actual activation
states inhabit and traverse the connection, not primarily with a large change in the fitted
loop operator.

The family-global construction is best for operator holonomy, section energy, and PCA
reconstruction. The completely global chart is markedly worse. Vertex-local charts slightly
improve actual loop-return error but do not produce lower operator holonomy or section energy.
This supports an intermediate picture: representations appear to share a coordinate system
within each kind of logical equivalence, while different families benefit from different
charts. Independent vertex charts may overfit their smaller calibration sets.

Incorrect activations are also less well captured by every shared chart. Reconstruction error
rises from 0.2702 to 0.3005 globally, 0.1847 to 0.2046 family-globally, and 0.1923 to 0.2134
vertex-locally. Some of the consistency gap therefore reflects incorrect states lying farther
from the dominant activation manifold. A follow-up should residualize section and return
errors against reconstruction error and prediction confidence.

The sheaf-Laplacian obstruction is slightly larger for incorrect groups in every chart mode:
approximately 0.00027 versus 0.00030 globally, 0.00017 versus 0.00018 family-globally, and
0.00020 versus 0.00022 vertex-locally. The values are small and depend on graph size, so the
direction is more informative than their raw scale.

Circuit localization remains negative. In family-global charts, nominated heads have
holonomy 0.3675, complement heads 0.3519, and all heads 0.3438. The causal patching screen
finds real intervention-sensitive heads, but lower holonomy is not concentrated in those
heads.

## Conclusion and next step

The first robust sheaf-style result is that correctly classified logical expressions form
better approximate sections of a shared fitted connection. The strongest chart model is not
fully global or maximally local, but family-global. The immediate confirmatory experiment
should evaluate per-example return and section energy as predictors of correctness while
controlling for reconstruction error, confidence, family, true label, seed, and expression
length. It should then test the frozen model on new diagram samples and on the three
low-diversity checkpoints without refitting the analysis decision rule.

