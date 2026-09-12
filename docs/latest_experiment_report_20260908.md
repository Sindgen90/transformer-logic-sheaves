# Latest experiment report: logical equivalence complexes and Q/K/V holonomy

**Experiment run:** `equivalence_complex_20260902_211352_321084Z`  
**Report date:** 2026-09-10  
**Status:** Complete  
**Models:** 18 width-128 Transformers  
**Compute:** Approximately 12 minutes 15 seconds on an RTX 4090

## 2026-09-10 update: contextual transport and causal Q/K/V interventions

Three follow-up analyses have now been completed on the three six-layer,
higher-diversity checkpoints:

1. exact pre-attention Q, K, and V activation patching at `<CLS>` and the
   rewritten operator token;
2. bidirectionally constrained, context-conditioned rewrite transport in local
   residual/Q/K/V fibers, a coupled Q-K fiber, and common `<CLS>` Q/K/V fibers;
3. causal patching from the penultimate expressions along two equivalent rewrite
   paths at layers 5 and 6.

### Exact connection and holonomy definition

For a canonically oriented rewrite relation (r), row-vector activation (x),
and context vector (c), the fitted connection is

\[
T_r(x;c)=(x-\mu_r^s)R_r+\mu_r^t+(c-\mu_r^c)B_r,
\]

where (R_r) is the orthogonal Procrustes solution and (B_r) is a ridge fit to
the remaining context-dependent translation. Its reverse is constrained to be
the exact algebraic inverse:

\[
T_r^{-1}(y;c)=(y-\mu_r^t-(c-\mu_r^c)B_r)R_r^\top+\mu_r^s.
\]

For a directed cycle

\[
\gamma=(r_1^{s_1},\ldots,r_m^{s_m}),\qquad s_i\in\{-1,+1\},
\]

the transported state and scalar return error are

\[
H_\gamma(x)=T_{r_m}^{s_m}\circ\cdots\circ T_{r_1}^{s_1}(x),
\qquad
h_\gamma(x)=
\frac{d^{-1}\lVert H_\gamma(x)-x\rVert_2^2}{\sigma_F^2}.
\]

Here σ²_F is the held-out feature variance averaged over examples and
coordinates. Thus (h_\gamma), edge error, and path-agreement error are
dimensionless variance-normalized squared errors. The eigenphases of
(R_\gamma=R_{r_1}^{s_1}\cdots R_{r_m}^{s_m}) are reported in radians. Q-K
centroid angles are also radians; normalized separation and stable rank are
dimensionless; maximum QK logit is a pre-softmax logit.

The local fiber is the root token of the rewritten subtree, with `<CLS>` as its
context. Because an associativity pentagon and commutator cube move among token
positions, their local fibers cannot be composed without an additional
identification. For those relations the experiment also uses a common `<CLS>`
Q/K/V fiber, conditioned by the rewritten-subtree activation. This avoids
pretending that activations at different token positions inhabit the same fiber.

### Held-out transport results

| Fiber | Identity | Global | Contextual | Target shuffled | Rank 8 |
|---|---:|---:|---:|---:|---:|
| local residual | 0.782 | **0.253** | 0.327 | 0.984 | 0.317 |
| local query | 0.778 | 0.196 | **0.187** | 0.625 | 0.190 |
| local key | 0.779 | 0.172 | **0.162** | 0.543 | 0.166 |
| local value | 0.849 | 0.202 | **0.191** | 0.588 | 0.195 |
| local coupled Q-K | 0.779 | **0.183** | 0.185 | 0.629 | 0.186 |
| `<CLS>` query | 0.761 | 0.527 | 0.540 | 1.193 | **0.516** |
| `<CLS>` key | 0.780 | 0.573 | 0.599 | 1.218 | **0.569** |
| `<CLS>` value | 0.787 | **0.598** | 0.629 | 1.262 | 0.599 |
| `<CLS>` coupled Q-K | 0.769 | **0.547** | 0.599 | 1.275 | 0.564 |

There is genuine rewrite-specific predictability: Q/K/V transports beat identity
and target-shuffled fits by large margins. The new context term yields a small
gain for local Q, K, and V separately, but not for coupled Q-K, residual, or any
common `<CLS>` fiber. It also increases average holonomy and path-disagreement
error. The proposed context model is therefore not supported as the better
connection.

At layer-6 `<CLS>` query, for example, the associativity pentagon return error is
0.294 for the contextual connection, versus 1.978 target-shuffled and 4.031
label-shuffled—but the simpler global map scores 0.045 and the rank-8 null 0.188.
For the commutativity cube the corresponding errors are 0.202, 3.155, 4.181,
0.095, and 0.051. Semantic labels matter, but small holonomy is not unique to the
full context-conditioned model. Rank-8 maps often match edge fidelity while
closing cycles more tightly, suggesting that much of the fitted full-rank
rotation is nuisance or underdetermined structure.

Exact two-edge inverse loops close at numerical zero for all baselines because
the inverse constraint guarantees this. Those values validate implementation;
they are not evidence that the network learned an involution.

### Q/K geometry

The clearest geometric transition is from diffuse to concentrated Q/K token
clouds in late layers. Normalized Q-K centroid separation rises from roughly
1.3-1.7 in layers 2-3 to 3.2-3.52 in layer 6, while query and key stable ranks
fall from roughly 4.5-5.3 to 3.0-3.2. Head 2 is distinctive at layer 5: its
attention entropy falls to about 2.51 while its maximum QK logit rises to about
2.55. The centroid angle itself does not show a universal convergence, remaining
about 1.6-1.85 radians across heads and layers.

### Causal Q/K/V results

Direct opposite-truth patching found the largest raw causal effect at layer-6
`<CLS>` query: donor-direction effect fraction 0.188, compared with 0.155 for the
matched context-shuffled donor, a specificity increment of only about 0.033.
The earlier layer-5 `<CLS>` query geometric candidate is therefore readable but
is not the strongest causal truth site. Local operator K/V interventions are
smaller and are generally comparable to their shuffled controls.

The equivalent-path experiment contains 576 path pairs from the associativity
pentagon, commutativity cube, and distributivity diamond. It patches Q, K, V, or
QKV from the penultimate expression on each route into the common start at
`<CLS>`. Layer-6 all-head query patching changes the true-class margin by about
1.13 and 1.46 logits on the two routes, with an absolute route disagreement of
1.454. The family-and-truth-matched shuffled control is 1.370, so the matched
routes are not more consistent. Layer-5 query disagreement is 0.984 versus 0.947
shuffled. K and V effects are much smaller; only layer-6 K has a modest positive
route-specificity difference (0.034 logits). The two query patches change the
predicted class differently on 7.35% of cases at layer 6. These models achieve
only 58.9% clean accuracy on the path subset, so this is evidence of a causal but
context-sensitive query channel, not a stable logical circuit.

### Circuit-restricted holonomy decision

Measuring holonomy inside a mechanistically identified circuit is sensible, but
only after selecting that circuit on separate discovery data and freezing it.
For vertex-dependent circuit projectors (P_v), the restricted edge map should
be (P_{v'}T_{v'\leftarrow v}P_v^\top). Every result must be accompanied by
leakage such as

\[
\ell_{v'\leftarrow v}=
\frac{\lVert(I-P_{v'}^\top P_{v'})T_{v'\leftarrow v}P_v^\top\rVert_F^2}
{\lVert T_{v'\leftarrow v}P_v^\top\rVert_F^2},
\]

because projection can make a cycle appear flat simply by discarding the
inconsistent directions. Bit flips are an excellent controlled family: each
flip satisfies (f_i^2=e), distinct flips satisfy (f_if_j=f_jf_i), and parity
provides a known circuit-level target. The current causal specificity is too
weak to freeze a defensible circuit, so circuit-restricted numbers were not
manufactured post hoc. The next targeted run should train an explicit bit-flip/
parity task, discover heads or sparse features on a disjoint split, then test
within-circuit holonomy and leakage on held-out flips.

## Executive summary

The latest experiment expanded the original logical-square study into a held-out
suite of eleven logical equivalence complexes: involutions, De Morgan squares,
an associativity pentagon, permutation hexagons, a commutativity cube,
distributivity paths, and higher-arity Boolean rewrites. Eighteen Transformers
were trained at depths 2, 4, and 6, under low- and higher-diversity data regimes,
with three random seeds per condition. A post-hoc audit then measured affine
orthogonal transports and loop closure in the residual stream and separately in
the query, key, and value activations of every layer and attention head.

Four conclusions stand out:

1. **Diverse data improved generalization.** The best condition was the six-layer,
   higher-diversity model, reaching **68.1% mean OOD accuracy** across expression
   depths 4 and 5. Its advantage over the matched low-diversity condition was
   4.3 percentage points.
2. **Generalization was operator-specific.** `OR3` generalized strongly, reaching
   **81.3% balanced accuracy** in the six-layer higher-diversity condition, while
   `XOR3` and `XOR4` remained around chance.
3. **The original scalar holonomy hypothesis failed.** Better-generalizing models
   did not have smaller aggregate loop-closure error. The memorizing controls
   often closed loops more tightly. Once depth and data regime were controlled,
   the partial correlation between state holonomy and OOD accuracy was only
   **-0.08**.
4. **Q/K/V localization produced a lead, not a causal conclusion.** Middle-to-late
   `<CLS>` queries were more transportable than keys or values, with the best
   global query site at layer 5. However, the low-diversity controls frequently
   had equal or lower errors. The geometry is therefore not yet specific to
   generalizing logical knowledge.

The scientifically important result is negative but constructive: a transport
must first predict individual logical rewrites better than identity and shuffled
controls before its loop closure can be interpreted as holonomy. The current
global map per rewrite captures some structure, but is too context-insensitive to
support a strong claim that logical knowledge is organized as a flat connection.

## 1. Experimental design

### Models and training

| Setting | Values |
|---|---|
| Transformer depth | 2, 4, and 6 layers |
| Width and heads | Width 128, four 32-dimensional heads |
| Feed-forward width | 256 |
| Seeds | 0, 1, and 2 |
| Training regimes | 512 low-diversity or 12,000 higher-diversity examples |
| Training expression depth | Up to 3 |
| OOD expression depths | Exactly 4 and exactly 5 |
| Optimization | 2,000 AdamW steps, batch size 256, learning rate 0.0003 |
| Evaluation sizes | 1,000 ID, 1,000 depth-4 OOD, 1,000 depth-5 OOD |

Examples contain a variable assignment followed by a prefix logical expression,
for example:

```text
<CLS> <ENV> x0=1 x1=0 x2=1 x3=0 <SEP> MAJ3 x0 x1 x2
```

The language supports constants, six variables, `NOT`, binary `AND`, `OR`, and
`XOR`, arity-marked three- and four-input operators, and the fixed-arity
operators `MAJ3`, `ITE3`, `EXACT1_3`, and `ATLEAST2_4`.

### Held-out equivalence complexes

The evaluation set contains 96 data-disjoint instances from each of eleven
families:

1. commutativity involution;
2. double-negation loop;
3. De Morgan `AND` square;
4. De Morgan `OR` square;
5. associativity pentagon;
6. n-ary `AND3` permutation hexagon;
7. commutativity cube and its square faces;
8. distributivity diamond with competing paths;
9. `ITE3` expansion/reduction loop;
10. `MAJ3` negation-duality loop;
11. `EXACT1_3` permutation hexagon.

This yields 1,056 diagrams, 4,320 diagram vertices, 1,632 loop instances, and 672
pairs of competing paths per model. Complete diagram topologies are held out from
transport fitting. Primitive rewrite labels are not held out, so this is topology
generalization rather than zero-shot generalization to a new logical rule.

## 2. Behavioral generalization

![ID and OOD accuracy by architecture depth](../runs/equivalence_complexes/equivalence_complex_20260902_211352_321084Z/plots/B_behavior.png)

**Figure 1.** Mean accuracy with sample-standard-deviation error bars across three
seeds. The higher-diversity condition gives substantially better ID accuracy at
all depths and its clearest OOD improvement at six layers.

| Layers | Training regime | Train accuracy | ID accuracy | Mean OOD | Depth 4 | Depth 5 |
|---:|---|---:|---:|---:|---:|---:|
| 2 | Low diversity | 1.000 | 0.662 | 0.635 | 0.622 | 0.647 |
| 2 | Higher diversity | 0.894 | 0.715 | 0.662 | 0.659 | 0.665 |
| 4 | Low diversity | 1.000 | 0.664 | 0.639 | 0.636 | 0.642 |
| 4 | Higher diversity | 0.992 | **0.747** | 0.658 | 0.662 | 0.653 |
| 6 | Low diversity | 1.000 | 0.676 | 0.638 | 0.635 | 0.641 |
| 6 | Higher diversity | 0.994 | 0.741 | **0.681** | **0.696** | **0.666** |

The low-diversity models perfectly memorize their 512 training examples but stop
near 64% OOD accuracy. The higher-diversity models learn more transferable
structure. Depth does not help monotonically: four layers have the best average ID
score, while six layers are needed for the best deeper OOD result.

Most validation improvement occurs during the first 250 training steps. Across
higher-diversity models, validation performance peaks near 73.1% around step 750,
while training loss continues falling afterward. Later optimization primarily
sharpens the training fit rather than improving held-out behavior.

## 3. Higher-arity operators

![Balanced OOD accuracy by operator](../runs/equivalence_complexes/equivalence_complex_20260902_211352_321084Z/plots/D_operator_accuracy.png)

**Figure 2.** Balanced accuracy on depth-5 OOD examples in the higher-diversity
condition. Balanced accuracy is essential because raw accuracy is strongly
confounded by the truth-value distribution of operators such as wide `AND` and
`OR`.

For the six-layer higher-diversity models, the leading operators were `OR3`
(0.813), `ATLEAST2_4` (0.664), and `NOT` (0.651). Binary `AND` and `OR` achieved
0.607 and 0.635. `EXACT1_3` reached only 0.520, while `XOR3` and `XOR4` were 0.494
and 0.488. The architecture can therefore process more-than-binary logical
operators, but it does not learn all of them equally. Parity-like operations are
the clearest remaining behavioral bottleneck.

## 4. What the original holonomy measurement computes

For each primitive rewrite label (e), an affine orthogonal transport is fitted
from balanced calibration pairs:

\[
T_e(x)=xR_e+b_e, \qquad R_e^\top R_e=I.
\]

For a closed path (p=e_1,\ldots,e_m), the maps are composed and evaluated on
held-out start states. The reported state-closure error is the variance-normalized
return-to-start displacement:

\[
\mathcal H_{\mathrm{state}}(p)
=
\mathbb E_x\left[
\frac{\|T_{e_m}\circ\cdots\circ T_{e_1}(x)-x\|_2^2}
{\operatorname{Var}(x)}
\right].
\]

The audit additionally measures the linear operator defect

\[
\mathcal H_{\mathrm{operator}}(p)
=\frac{\|R_p-I\|_F^2}{d},
\]

along with affine drift, context-dependent dispersion, edge fidelity, and
agreement between competing paths.

## 5. Why edge fidelity changes the interpretation

![Holonomy null controls](../runs/equivalence_complexes/equivalence_complex_20260902_211352_321084Z/holonomy_audit/plots/M_holonomy_nulls.png)

**Figure 3.** The identity connection has perfect closure but poor edge fidelity;
therefore zero loop error alone cannot demonstrate logical coherence. The fitted
connection improves individual rewrites, but still has substantial closure error.

| Connection | Held-out edge error | State closure error |
|---|---:|---:|
| Identity | 1.372 | **0.000** |
| Target-shuffled | 1.822 | 1.897 |
| Fitted | **1.081** | 0.642 |

The fitted transports reduce held-out edge error by 21.2% relative to identity and
substantially outperform target-shuffled fitting. They therefore capture genuine
rewrite-associated structure. But the identity counterexample is decisive: a
connection can close every loop perfectly while transporting no rewrite at all.

Of the fitted state-closure error, only 8.2% is systematic mean displacement;
91.8% is context-dependent dispersion. A single global affine map for each
rewrite is therefore a weak approximation to the actual, context-dependent change
in representation.

## 6. Holonomy does not yet predict generalization

![Holonomy correlations with OOD accuracy](../runs/equivalence_complexes/equivalence_complex_20260902_211352_321084Z/holonomy_audit/plots/N_holonomy_correlations.png)

**Figure 4.** Blue bars are pooled Pearson correlations across all 18 models;
green bars are partial correlations after linearly removing architecture depth and
training-data regime. The large pooled correlations mostly disappear after this
control.

Pooled state closure correlates **positively** with OOD accuracy (+0.60), contrary
to the initial expectation that lower holonomy should mark better reasoning.
Pooled path-agreement error behaves similarly (+0.67). These relationships are
confounded by the experimental conditions. After controlling depth and data
regime, the correlations are -0.08 for state closure, +0.12 for path agreement,
and -0.12 for held-out edge error.

With only 18 models, these are descriptive correlations rather than inferential
evidence. Nevertheless, the expected strong negative relationship between
holonomy and OOD behavior is absent. This falsifies the simplest metric-level
hypothesis; it does not falsify the broader sheaf or coherence perspective.

## 7. Layerwise Q/K/V localization

The post-hoc Q/K/V audit records the projected query, key, and value activations
immediately before attention in every layer. It evaluates both the `<CLS>` vector
and the mean over non-padding, non-`<CLS>` tokens. All 128 dimensions are used for
the concatenated-head result, and 32-dimensional per-head results are also saved.

![QKV edge transport by layer](../runs/equivalence_complexes/equivalence_complex_20260902_211352_321084Z/qkv_holonomy/plots/O_qkv_edge_transport.png)

**Figure 5.** Held-out rewrite transport error for the six-layer,
higher-diversity models. Lower is better. Layer-1 `<CLS>` is omitted because it is
constant before the first attention operation and therefore geometrically
degenerate.

For the global `<CLS>` representation, query transport improves from 0.830 at
layer 2 to a minimum of **0.660 at layer 5**, then rises to 0.708 at layer 6. At
layer 5, key and value errors are 0.757 and 0.803. Queries are therefore the most
transportable component in the middle-to-late global representation.

Token-mean errors are much smaller, beginning around 0.04 in layer 1 and growing
with depth. This should not be read as superior reasoning: the layer-1 token mean
largely reflects lexical and bag-of-tokens invariance before attention has mixed
the sequence.

![QKV state holonomy by layer](../runs/equivalence_complexes/equivalence_complex_20260902_211352_321084Z/qkv_holonomy/plots/P_qkv_holonomy.png)

**Figure 6.** Variance-normalized return-to-start error for Q, K, and V. The best
supported global site is again the layer-5 query, with state closure 0.510 versus
0.661 for keys and 0.686 for values. Token-mean closure is much smaller but is
dominated by lexical invariance and must be interpreted together with edge gain.

![Comparison of QKV geometry across data regimes](../runs/equivalence_complexes/equivalence_complex_20260902_211352_321084Z/qkv_holonomy/plots/T_qkv_condition_comparison.png)

**Figure 7.** The crucial negative control. Low-diversity models frequently have
lower Q/K/V transport error than the better-generalizing higher-diversity models.
The discovered geometry is therefore shared structural regularity, not a unique
location of generalizing logical knowledge.

The evidence supports a narrower statement: rewrite-related variation is visible
in Q, K, and V, and global query geometry becomes relatively more regular in
middle-to-late layers. It does **not** show that the model causally uses that query
subspace or that logical knowledge is stored there. This experiment did not yet
perform Q/K/V activation patching.

## 8. Overall verdict

The behavioral experiment succeeded: the data generator, higher-arity language,
diagram suite, balanced evaluation, and multi-seed depth comparison all work, and
they reveal a real gain from diverse training data. The geometric experiment also
succeeded as an audit because it exposed a flaw in the original interpretation.

What is supported:

- the fitted maps contain real edge-level rewrite information;
- associativity and commutativity families are often easier to transport than
  parity-based permutation hexagons;
- Q/K/V geometry changes systematically across layers;
- `<CLS>` queries are the strongest current localization candidate.

What is not supported:

- that small loop closure alone measures learned logical coherence;
- that lower current holonomy predicts better OOD generalization;
- that Q rather than K or V causally carries the logical computation;
- that one global transport per rewrite is an adequate connection.

## 9. Recommended next experiment

The next run should replace the global rewrite map with a context-conditioned,
bidirectionally constrained transport at the rewritten subtree token. It should
fit forward and inverse transformations jointly and explicitly test algebraic
relations such as involutions, commutators, braid relations, and the associativity
pentagon. Every cycle result should be conditioned on held-out edge fidelity and
compared with identity, target-shuffled, label-shuffled, and low-rank nulls.

Alongside transport holonomy, record the new geometric diagnostics suggested by
the RoPE work:

- Q-K centroid angle and normalized separation;
- stable rank and leading-singular-value energy;
- attention entropy and maximum QK logit;
- separate and coupled Q/K transport consistency;
- operator eigenvalues or eigenphases as gauge-robust holonomy descriptors.

Finally, patch Q, K, and V independently at the strongest candidate sites—starting
with the layer-5 `<CLS>` query—and compare the downstream logit effects along two
equivalent rewrite paths. That intervention is needed to distinguish a readable
geometric signature from a representation the classifier actually uses.

## Artifact index

- Main run report: `runs/equivalence_complexes/equivalence_complex_20260902_211352_321084Z/report.md`
- Holonomy audit: `runs/equivalence_complexes/equivalence_complex_20260902_211352_321084Z/holonomy_audit/report.md`
- Q/K/V audit: `runs/equivalence_complexes/equivalence_complex_20260902_211352_321084Z/qkv_holonomy/report.md`
- Behavioral tables: `runs/equivalence_complexes/equivalence_complex_20260902_211352_321084Z/tables/`
- Q/K/V detailed tables: `runs/equivalence_complexes/equivalence_complex_20260902_211352_321084Z/qkv_holonomy/`
