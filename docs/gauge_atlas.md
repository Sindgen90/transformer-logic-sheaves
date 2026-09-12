# Gauge-atlas holonomy for logical equivalence

This document fixes the definitions for the next experiment before any expensive run. It
implements the construction in *A Gauge Theory of Superposition: Toward a Sheaf-Theoretic
Atlas of Neural Representations* while retaining our earlier operational transport metric as
a separate quantity.

## 1. Base space and fibers

The base space is one logical-equivalence diagram. Its vertices are syntactically different
but truth-functionally equivalent expressions. Its edges are primitive rewrites. The existing
suite supplies involutions, De Morgan squares, a distributivity diamond, an associativity
pentagon, permutation/braid hexagons, a commutativity cube, ITE expansion, majority duality,
and higher-arity operators.

For a fixed checkpoint, layer, scope, and activation component, vertex `v` has a collection of
ambient activation vectors

\[
X_v = \{x_v^{(1)},\ldots,x_v^{(n)}\}\subset\mathbb R^d,
\]

where instance `i` uses the same sampled operands and assignment at every vertex. These
matched instances replace the paper's Voronoi-boundary overlaps. They are a stronger and more
meaningful overlap for this controlled problem: `x_u^(i)` and `x_v^(i)` differ by the exact
rewrite represented by edge `(u,v)`.

The implemented fibers are residual, Q, K, V, and coupled Q–K activations at `<CLS>`, the
expression root, or the mean over expression tokens. Q/K/V can be restricted to an explicit
set of heads. Consequently, an atlas can describe the whole representation, one attention
component, or a circuit nominated by causal patching.

## 2. Charts and edge transports

Each vertex gets an independently fitted, centered PCA chart

\[
B_v\in\mathbb R^{d\times k},\qquad B_v^\top B_v=I,
\]

using only the calibration half of the held-out diagram instances. Coordinates are

\[
z_v=B_v^\top(x_v-\mu_v).
\]

For canonical edge orientation `u < v`, stack matched coordinates as columns in `Z_u` and
`Z_v`. The paper-style ridge transport is

\[
\widehat T_{vu}=Z_vZ_u^\top(Z_uZ_u^\top+\lambda I)^{-1}.
\]

Its orthogonal part is

\[
Q_{vu}=\operatorname{polar}(\widehat T_{vu}).
\]

The basis-overlap correspondence transport is

\[
\widehat P_{vu}=\operatorname{polar}(B_v^\top B_u),
\]

Both `Q_vu` and `P_vu` are correctly typed maps from the source chart fiber
`F_u` to the target chart fiber `F_v`. The edgewise relative operator

\[
g_{vu}=\widehat P_{vu}^\top Q_{vu}\in O(k),\qquad g_{uv}=g_{vu}^{-1}=g_{vu}^\top.
\]

is instead an endomorphism of the **source** fiber: under independent chart changes it
transforms as `g_vu -> U_u g_vu U_u^T`. Consequently, `g` operators on consecutive edges
generally live in different fibers and cannot be multiplied directly. The original
`gauge-atlas` output is retained as a clearly labeled legacy diagnostic; the corrected
`typed-gauge` experiment does not use its edge-defect product as holonomy.

These operators are not network weights. They are dimensionless orthogonal maps fitted in
the learned `k`-dimensional coordinate charts of a chosen activation fiber. Q, K, V, and
coupled Q-K therefore receive distinct fitted connections at every measured layer and circuit.

## 3. Shearing, fidelity, and persistence

For each edge, the implementation records the paper's normalized proxy disagreement

\[
D_{\mathrm{shear}}(u,v)=
\frac{\|Q_{vu}-\widehat P_{vu}\|_F}{2\sqrt{k}},
\]

the empirical transfer mismatch, its covariance lower bound, and
`sigma_min(T_hat)`. A persistence threshold drops ill-conditioned edges and then analyzes the
largest remaining connected component. Because edge filtering changes topology, every result
includes both edge coverage and independent-cycle coverage.

Held-out edge fidelity is normalized reconstruction MSE after mapping evaluation activations
through the fitted source chart, `Q`, and target chart. No evaluation vector participates in
PCA or transport fitting. Holonomy is not interpretable when edge fidelity is poor, even if
the loop product happens to be near identity.

## 4. Fundamental-cycle holonomy

A deterministic breadth-first spanning tree is chosen for each retained connected component.
Each non-tree edge (chord) defines one fundamental cycle. For a directed cycle
`gamma: c0 -> ... -> cL=c0`, the two typed loop transports are

\[
H_Q(\gamma)=Q_{c_0\leftarrow c_{L-1}}\cdots Q_{c_2\leftarrow c_1}Q_{c_1\leftarrow c_0},
\]

\[
H_P(\gamma)=P_{c_0\leftarrow c_{L-1}}\cdots P_{c_2\leftarrow c_1}P_{c_1\leftarrow c_0}.
\]

Both are endomorphisms of the same base fiber `F_c0`, so their relative holonomy is

\[
H_{\mathrm{rel}}(\gamma)=H_P(\gamma)^{-1}H_Q(\gamma).
\]

Under a chart gauge `U_v in O(k)`, edges transform as

\[
Q_{vu}\mapsto U_vQ_{vu}U_u^\top,\qquad
P_{vu}\mapsto U_vP_{vu}U_u^\top,
\]

so all three loop operators change only by conjugation at the start vertex. Their Frobenius
distances from identity and eigenvalues/eigenphases are therefore gauge invariant. Tests
verify this numerically from actual chart reparameterizations. The primary statistic is the
relative loop `H_rel`; `H_Q` and `H_P` are also always reported separately.

Two normalizations are reported:

\[
D_{\mathrm{paper}}(H)=\frac{\|H-I\|_F}{\sqrt{2k}},
\qquad
D_{[0,1]}(H)=\frac{\|H-I\|_F}{2\sqrt{k}}.
\]

The first exactly matches the paper. However, its true range on `O(k)` is `[0,sqrt(2)]`, not
`[0,1]`; `h=-I` attains `sqrt(2)`. The second is the corrected unit normalization. Both are
stored so that paper comparisons do not silently change scale.

## 5. Comparison with our earlier holonomy

The two metrics answer different questions and must not share a column name:

| Metric | Object composed | Units | Question |
|---|---|---|---|
| `transport_holonomy` | typed learned transports `Q` | dimensionless Frobenius distance | Does the learned connection close around the loop? |
| `proxy_holonomy` | typed chart-overlap transports `P` | dimensionless Frobenius distance | Does the chart-overlap reference close? |
| `relative_holonomy` | `H_P^-1 H_Q` in one base fiber | dimensionless, bounded by 1 | Do the learned and proxy connections have the same loop action? |
| `state_return_error` | affine chart transports | activation MSE / activation variance | Does an actual held-out state return after a loop? |
| legacy contextual holonomy | rewrite-conditioned affine maps | activation MSE / activation variance | Do globally/shared rewrite maps close on observed states? |

Operator holonomy can be large while state-return error is small if the evaluated states occupy
directions on which the loop operator acts weakly. The converse can occur through chart means,
projection loss, or poor edge fit. The generated scatter plot makes this disagreement visible
rather than forcing one metric to stand in for the other.

## 6. Nulls and controls

Every site and family is compared against:

- `proxy_connection`: set `Q=P`, making relative holonomy exactly zero;
- `target_shuffled`: arbitrary calibration correspondences are used on each edge;
- `assignment_shuffled`: correspondences are shuffled within truth-value strata, preserving
  the output label while breaking the matched environment/operand identity;
- `topology_shuffled`: source-fiber relative operators are moved between edges only after
  parallel transport into a common root frame;
- `random_basis`: chart bases are random orthonormal subspaces of the same dimension;
- `spectrum_matched`: every learned source-relative operator is independently conjugated by a random orthogonal
  matrix, exactly preserving eigenvalues and `rank(g-I)` while destroying cross-edge alignment.

The rank-matched control is more stringent than merely generating a low-rank random matrix: it
preserves the full per-edge spectrum. A signal must therefore depend on coherent orientation
between edges, not only on how far each edge is from identity.

## 7. Circuit-restricted and bit-flip experiment

Circuit restriction should be downstream of causal discovery. First use independent Q/K/V
patching to nominate a layer and one or more heads by a specific causal effect relative to
matched shuffles. Then pass those head indices to the atlas analysis. Concatenating only those
head slices defines a circuit fiber. Running Q, K, V, and coupled Q–K separately distinguishes
where the connection is represented; it does not claim that a head is a circuit merely because
its atlas is geometrically clean.

The bit-flip condition changes one environment assignment (`x0`, `x1`, and so on) while keeping
the expression, operands, graph topology, checkpoint, data split, and feature selector fixed.
This supports paired comparisons:

\[
\Delta D_{\mathrm{hol}} =
D_{\mathrm{hol}}(\text{assignment with }x_j\text{ flipped})-
D_{\mathrm{hol}}(\text{original assignment}).
\]

A circuit carrying the value of `x_j` should show reproducible changes under this intervention,
especially on families whose expressions causally depend on `x_j`, while irrelevant-variable
flips provide a natural negative control. The first run should stratify results by whether the
flip changes the diagram's Boolean value; otherwise value-changing and value-preserving effects
can cancel. Causal patch effects and atlas deltas should then be compared at the same layer,
component, heads, and token scope.

## 8. Outputs and acceptance criteria

Each corrected invocation creates a new timestamped folder beneath `RUN/typed_gauge/`. It writes:

- `tables/summary.csv`: family-balanced site/control summaries;
- `tables/edges.csv`: `sigma_min`, shearing, mismatch, and bound per retained edge;
- `tables/cycles.csv`: every independent fundamental cycle and every eigenphase in radians;
- `tables/bitflip_deltas.csv`: paired intervention-minus-original differences, split into
  absent, globally irrelevant, value-preserving, and value-changing strata;
- `plots/`: eight figures (A-H) for persistence, circuit localization, nulls, typed loop
  decomposition, coverage, edge fidelity, causal bit flips, and chart-dimension sensitivity;
- `config.json`, `status.json`, and `report.md`.

Before interpreting a candidate site, require: good held-out edge fidelity relative to shuffles,
adequate minimum singular value, nontrivial retained cycle coverage, stability across seeds, and
separation from random-basis, assignment-shuffled, topology-shuffled, and spectrum/rank-matched
nulls. A circuit-localized result additionally requires a causal patching effect above its own
matched-shuffle control.

## 9. Commands

Full Q/K/V atlas with paired flips of `x0` and `x1`:

```powershell
python -m logic_sheaves.cli gauge-atlas RUN_DIRECTORY --device cuda --bit-flips x0 x1
```

Corrected typed experiment with frozen causal nominations, a persistence grid, bootstrap
stability, paired `x0` intervention, and same-mask nulls:

```powershell
python -m logic_sheaves typed-gauge RUN_DIRECTORY --device cuda
```

Circuit-restricted example after heads 1 and 3 at the relevant checkpoint have been nominated:

```powershell
python -m logic_sheaves.cli gauge-atlas RUN_DIRECTORY --device cuda --heads 1 3 --bit-flips x0
```

A smaller definition-checking run can restrict scopes/components and chart size:

```powershell
python -m logic_sheaves.cli gauge-atlas RUN_DIRECTORY --device cuda --components query key --scopes cls --chart-dimension 16
```
