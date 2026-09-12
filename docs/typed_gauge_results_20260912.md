# Type-correct circuit holonomy: first results

**Reference model run:** `equivalence_complex_20260902_211352_321084Z`  
**Typed-gauge run:** `typed_gauge_20260912_184349_746654Z`  
**Models:** three width-128, six-layer, higher-diversity checkpoints  
**Status:** complete

## Correction to the gauge typing

For an oriented edge from logical form `u` to equivalent form `v`, both the learned
orthogonal transport and chart-overlap transport have type

\[
Q_{v\leftarrow u},P_{v\leftarrow u}:F_u\rightarrow F_v.
\]

The edgewise relative operator `P^T Q` is an endomorphism of `F_u`, not another map
from `F_u` to `F_v`. Relative operators based at consecutive vertices therefore cannot
be multiplied directly. The corrected experiment composes the two connections separately:

\[
H_Q(\gamma)=Q_n\cdots Q_1,\qquad
H_P(\gamma)=P_n\cdots P_1,\qquad
H_{rel}(\gamma)=H_P(\gamma)^{-1}H_Q(\gamma).
\]

All three loop operators act on the same base fiber. Under independently chosen orthogonal
chart gauges `U_v`, each transforms by conjugation at the loop base. Eigenvalues,
eigenphases, and distances from identity are therefore gauge invariant. The reported unit
distance is

\[
D(H)=\frac{\lVert H-I\rVert_F}{2\sqrt{k}}\in[0,1].
\]

This is dimensionless. It is fitted in activation-chart coordinates and is not a distance
between network weights.

## Experimental design

The causal circuit screen was frozen before the atlas analysis. Seeds 0 and 1 nominated
sites and seed 2 confirmed them. The confirmed sets were layer-6 operator-token K heads
1 and 3, layer-5 operator-token Q head 2, and layer-5 operator-token K heads 0 and 3. The
strong layer-6 `<CLS>` Q-head-1 discovery hit failed confirmation and was retained as a
negative reference.

For every nominated set, the analysis measures its circuit heads, complementary heads,
and all heads in Q, K, V, and coupled Q-K fibers. It fits chart dimensions 8 and 16 with a
disjoint fit/evaluation split. Edges must pass held-out fidelity versus target shuffle,
condition ratio, bootstrap stability, and a singular-value threshold. Fundamental cycles
are recomputed after filtering on the largest connected component. All null connections are
evaluated using the learned connection's retained-edge mask.

The primary coverage-oriented run uses 16 bootstrap resamples, maximum normalized bootstrap
instability 0.35, minimum condition ratio 0.02, and singular-value thresholds 0, 0.015,
0.03, 0.06, 0.12, and 0.24. A separate strict run used maximum bootstrap instability 0.15.

## Main results

At the reference threshold `sigma_min >= 0.015`, 194 of 792 confirmed-circuit strata retain
at least one cycle. Mean relative holonomy is 0.2995. This is much lower than arbitrary
target matching (0.7034), truth-preserving assignment shuffle (0.7055), and random chart
bases (0.6909). More importantly, it remains below the topology-shuffled control (0.3741)
and the spectrum-matched control (0.3958). The seed-wise learned means are 0.2750, 0.2945,
and 0.3289, while topology-shuffled means are 0.3474, 0.3831, and 0.3918. Thus the direction
of the learned-versus-topology effect is stable across all three model seeds.

The component means are:

| Fiber | H_Q | H_P | H_rel | Held-out edge error | State-return error |
|---|---:|---:|---:|---:|---:|
| Q | 0.3053 | 0.2086 | 0.2842 | 0.5798 | 0.4900 |
| K | 0.3330 | 0.2490 | 0.3057 | 0.6337 | 0.5224 |
| V | 0.3575 | 0.2431 | 0.3338 | 0.6895 | 0.6331 |
| coupled Q-K | 0.2868 | 0.1992 | 0.2719 | 0.6102 | 0.4647 |

The signal is not strongly circuit-localized. Confirmed circuit heads have mean relative
holonomy 0.2995, their complements 0.3036, and all heads 0.3061. The rejected `<CLS>`
reference is also nearby at 0.3160. Causal patching and geometric loop consistency therefore
do not yet point to a uniquely localized circuit.

The persistence curve drops from 0.2995 at thresholds up to 0.015, to 0.2895 at 0.03,
0.2348 at 0.06, 0.1659 at 0.12, and 0.1126 at 0.24. Coverage falls at the same time: the
number of cycle-bearing strata goes from 194 to 52. The low value at strong filtering cannot
be interpreted as global flattening; it describes a small, well-conditioned surviving
subgraph. Under the stricter bootstrap cap of 0.15, only 42 of 792 strata retain cycles,
which is too sparse for the main comparison.

For the paired `x0` intervention at the reference threshold, value-changing examples raise
mean relative holonomy by 0.0236 in Q, 0.0565 in K, 0.0593 in V, and 0.0527 in coupled Q-K.
However, these estimates use only 6-11 finite strata per component and are not uniformly
stable across seeds. They are candidates for replication, not established causal effects.
Value-preserving and globally irrelevant strata are similarly sparse; there are no finite
structurally absent-variable cycles in this dataset slice.

## Conclusion

The corrected experiment supports a narrow claim: learned Q/P connection disagreement has
cross-edge organization that is not explained entirely by independent edge spectra or by
placing the same edge information on the wrong logical topology. It does not yet support a
claim that low holonomy is localized to the causally patched heads, and the apparent
assignment sensitivity needs a purpose-built balanced intervention dataset with substantially
more cycle-bearing examples.

Every invocation writes eight plots, edge and cycle tables, operator eigenphases, paired
intervention deltas, configuration, and a generated report to a new timestamped
`RUN/typed_gauge/` directory.

