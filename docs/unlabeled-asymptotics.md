# Unlabeled asymptotics

`c1f_unlabeled_asymptotics` analyzes the ordinary generating functions built
by the exact monadic Polya construction:

```python
from unaryfomc import c1f_unlabeled_asymptotics

analysis = c1f_unlabeled_asymptotics("TRUE")
print(analysis.exponential_growth)  # 2.955765285651995...
print(analysis.power)               # -0.5
print(analysis.leading_constant)    # 0.4428767697822064...
```

The leading approximation is

```text
UFOMC(phi, n)
    ~ leading_constant * exponential_growth**n * n**power.
```

Unlike exact unlabeled counting, the asymptotic API may also handle a
higher-arity relation by an asymptotic rigidity transfer from the labeled
problem. The free relation entries contribute their known power-of-two factor,
and the labeled count is divided by `n!`. This does not claim an exact
finite-`n` unlabeled count for higher-arity vocabularies.

## Polya hierarchy

For monadic vocabularies, the analyzer solves the subcritical hierarchy of
Adams evaluations `T(x^k)`. `polya_tail_terms` controls the finite numerical
tail used to locate the characteristic point. General finite guarded profile
systems are treated as multitype systems; complete compatibility is retained
as a faster scalar path.

```python
guarded = c1f_unlabeled_asymptotics(
    "forall x: exists<=2 y: (f(y) = x & TRUE)",
    polya_tail_terms=192,
    expansion_order=4,
)
print(guarded.regime)
print(guarded.expansion_estimate(100))
```

The formal engine expands the retained Polya hierarchy to every requested
order. Periodic implicit systems and pure periodic cycle systems expose
`periodicity`, `residue_powers`, and `residue_expansions`, just like labeled
results.

## Interval certificates

With the `certify` extra installed, use `certify=True` to enclose the radius,
growth constant, leading constant, and requested higher coefficients:

```python
certified = c1f_unlabeled_asymptotics(
    "TRUE",
    certify=True,
    precision_bits=192,
    max_precision_bits=768,
    expansion_order=4,
)
print(certified.certificate.verified_properties)
print(certified.certificate.dominance_certified)
```

For higher terms, Cauchy balls account for coefficients omitted from the
finite Adams hierarchy. The certificate path covers complete compatibility,
scalar and regular multitype guarded systems, one critical tree component or
simple cycle pole over analytic trees, repeated independent critical blocks,
periodic root-of-unity orbits, and capped global faces recognized by the
automatic classifier.

As in the labeled case, the proof inventory is authoritative and there is no
finite-`n` remainder bound. Radius-one monadic OGFs, nested nonlinear critical
components, and subleading fixed-global-cardinality expansions are outside the
current automatic path.
