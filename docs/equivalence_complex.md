# Symbolic equivalence-complex protocol

## Question

Do Transformers that generalize Boolean evaluation learn internal states whose
rewrite transports remain compatible across variables, higher-arity operators,
different diagram topologies, and alternative paths?

This phase deliberately goes beyond a single De Morgan square. It treats a
logical-equivalence family as a finite directed diagram: vertices are equivalent
expressions, edges are named rewrites, loops are closed edge sequences, and path
pairs are different edge sequences with the same endpoints.

## Language

Each example contains an explicit environment followed by a prefix expression:

```text
<CLS> <ENV> x0=1 x1=0 x2=1 x3=0 <SEP> MAJ3 x0 x1 x2
```

The language includes constants, variables, `NOT`, binary `AND`/`OR`/`XOR`,
arity-marked `AND3`/`AND4`, `OR3`/`OR4`, `XOR3`/`XOR4`, and the fixed-arity
operators `MAJ3`, `ITE3`, `EXACT1_3`, and `ATLEAST2_4`. Arity-marked prefix tokens
keep serialization unambiguous. Binary forms are retained so associativity remains
observable instead of being erased by flattening.

Training expressions have maximum depth 3. ID validation and test expressions use
the same depth range but are syntactically disjoint. OOD sets use exact depths 4
and 5. Labels are approximately balanced and all architecture/seed combinations
share the same split.

## Diagram families

The held-out suite contains:

1. an `AND` commutativity involution;
2. a double-negation loop;
3. an `AND` De Morgan/commutativity square;
4. an `OR` De Morgan/commutativity square;
5. the five-vertex associativity pentagon;
6. an `AND3` permutation hexagon;
7. a three-rewrite, eight-vertex commutativity cube with six square faces;
8. a distributivity diamond comparing a direct rewrite with a commuted route;
9. an `ITE3` expansion/reduction loop;
10. a `MAJ3` negation-duality loop;
11. an `EXACT1_3` permutation hexagon.

Every family is generated under independent variable assignments and balanced
between Boolean results 0 and 1. Diagram vertices are checked for semantic
equivalence. They are also checked against the train, validation, ID, and OOD
sets so the evaluated strings are data-disjoint.

## Transport calibration and topology holdout

For every named rewrite, an affine orthogonal map is fitted on isolated source and
target pairs:

\[
T_r(h) = (h - \mu_s) R_r + \mu_t, \qquad R_r^T R_r = I.
\]

Calibration observes individual edges, never a complete evaluation loop or a
pair of competing routes. A second independently generated set of isolated pairs
tests primitive rewrite generalization. Complete diagram instances use a third
seed and are wholly held out.

This is a topology holdout rather than a rewrite-label holdout: the model knows
which map belongs to each primitive rewrite, but transport fitting cannot exploit
the closed shape it will later be asked to traverse.

## Metrics

All representation errors use final-layer `<CLS>` states and are divided by the
held-out activation variance.

- **Identity error:** adjacent-state MSE without alignment.
- **Rewrite transport error:** held-out MSE after the fitted edge transport.
- **Edge CKA:** ordinary linear similarity baseline.
- **Holonomy error:** MSE between the initial state and the result of composing
  transports around a closed loop.
- **Path-agreement error:** MSE between states produced by two transported routes
  having identical start and end vertices.
- **Path-endpoint error:** error from each transported route to the actual endpoint.
- **Semantic accuracy:** classification accuracy over all diagram vertices.
- **Prediction consistency:** fraction of diagrams receiving one common prediction
  at every equivalent vertex.

Low holonomy is not sufficient when edge transport is poor: inaccurate maps may
still compose to something close to the identity. Conclusions therefore require
edge fidelity, loop closure, path agreement, behavioral accuracy, and baselines
to move together.

## Default run

The default compares 2-, 4-, and 6-layer width-128 Transformers, three seeds, and
two data regimes: 512 versus 12,000 training examples. Each model trains for 2,000
steps. There are 192 calibration pairs per rewrite and 96 held-out diagrams per
family.

```powershell
python -m logic_sheaves complex-sweep --device cuda
```

Every invocation creates a unique directory containing checkpoints, raw CSV/JSON
tables, a generated Markdown report, exact configuration and status files, and
figures A-K.
