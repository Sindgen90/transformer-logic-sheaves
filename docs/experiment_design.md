# Experiment 0: cheap falsification screen

## Research question

Do models that generalize Boolean evaluation to deeper expression trees develop more globally coherent internal representations than models that fit a smaller, low-diversity training set?

The first experiment is intentionally pre-sheaf-theoretic. It tests whether there is a phenomenon worth formalizing before adding cohomology.

## Controlled comparison

The two conditions use the same Transformer, optimizer, number of updates, validation set, and OOD sets. They differ only in the number of unique depth-at-most-3 training expressions:

- **low diversity:** encourages exact-sequence fitting;
- **higher diversity:** encourages learning a reusable evaluator.

Evaluation uses exact depths 4, 5, and (in the pilot) 6. Multiple seeds are essential: the proposed unit of evidence is a trained model, not an individual expression.

## Equivalence cycle

For held-out operands `a` and `b`, construct the square

```text
NOT(AND(a,b))  --commute-->  NOT(AND(b,a))
      ^                            |
      |                            | De Morgan
 De Morgan                        v
      |                       OR(NOT(b),NOT(a))
      |                            |
      +-- OR(NOT(a),NOT(b)) <------+ commute
```

All four expressions have the same truth value. Each directed edge receives a single affine orthogonal transport fitted on calibration cycles and tested on disjoint cycles. This restriction prevents a powerful learned alignment network from manufacturing coherence.

## Measurements

- behavioral train, ID, and per-depth OOD accuracy;
- operator-specific linear probes for intermediate subtree truth values;
- identity edge energy between equivalent representations;
- held-out orthogonal-transport error;
- transport holonomy error after traversing the full square;
- linear CKA between adjacent representations.

The cycle quantities are normalized by held-out activation variance. Lower error means greater coherence; higher CKA means greater similarity.

## Go/no-go criterion

This screen only earns a larger study if, across at least 20 independently trained models:

1. train accuracy is high in both regimes;
2. OOD accuracy spans a useful range rather than forming one cluster;
3. held-out cycle/holonomy measures predict OOD accuracy after controlling for ID accuracy and training-set size;
4. the cycle measure adds predictive value over local probes, CKA, and raw pairwise identity energy;
5. the relationship replicates on a second equivalence family (associativity or double negation).

Failure on items 2–4 is a reason to redesign or stop, not to add more topology.

## Known limitations of Experiment 0

- A bidirectional encoder is not forced to implement the parse-tree algorithm.
- CLS states describe whole expressions; operator-token states are only a proxy for subtree computation.
- One equivalence square is not a sheaf and its holonomy is not a cohomology class.
- Low training diversity may change optimization dynamics in addition to memorization pressure.
- Correlation across trained models is not a causal result.

The next phase should add a stack-oriented causal model, activation patching, more independent equivalence cycles, checkpoint trajectories, and statistical controls only if this screen finds a robust signal.

