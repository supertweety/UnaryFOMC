# Labeled asymptotics

UnaryFOMC computes effective labeled asymptotic expansions from the same finite
profile system used for exact counting. The concise entry point is

```python
from unaryfomc import c1f_labeled_asymptotics

analysis = c1f_labeled_asymptotics("TRUE")
```

The leading fields describe

```text
FOMC(phi, n)
    ~ leading_constant * n! * exponential_growth**n * n**power,
```

where `exponential_growth == 1 / radius`. The result also records the critical
tree values, square-root directions, Perron vectors, finite-cycle correction,
denominator coefficient, numerical residual, and structural regime.

## Finite specification

`compile_c1f_labeled_specification` exposes the compiled EGF system. For
ordinary transient profile types its tree equations have the form

```text
T_q = x * m_q * exp(sum(T_p for p -> q)).
```

Guarded predecessor formulas replace the unrestricted child set by a finite
capped-state projection. The specification evaluates the tree map and its
derivatives, cycle transfer matrices, finite short-cycle correction, and long
cycle determinant. It is a finite symbolic-numerical representation, not a
truncated coefficient series.

```python
from unaryfomc import compile_c1f_labeled_specification

specification = compile_c1f_labeled_specification(
    "forall x: exists<=2 y: (f(y) = x & TRUE)"
)
print(specification.tree_dimension)
print(specification.dependency_matrix())
```

Compiled problems provide the same workflow:

```python
from unaryfomc import compile_c1f_fomc

compiled = compile_c1f_fomc("TRUE")
specification = compiled.labeled_specification()
analysis = compiled.labeled_asymptotics(expansion_order=4)
```

## Continued expansions

`expansion_order` counts half-steps after the leading power. For unrestricted
endofunctions, the expansion agrees with Stirling's series:

```python
expanded = c1f_labeled_asymptotics("TRUE", expansion_order=6)

for term in expanded.expansion_terms(100):
    print(term.power, term.constant)

print(expanded.estimate(100))
print(expanded.expansion_estimate(100))
```

`estimate(n)` uses only the leading term. `expansion_estimate(n)` uses every
computed term. The formal Laurent-Puiseux recurrence continues to arbitrary
requested order within the regular regime.

## Periodic systems

The implementation detects exact support congruences and the corresponding
root-of-unity singularity orbit. Every asymptotic order is Fourier-projected
onto residue classes, so a vanishing leading projection can reveal a later
nonzero term.

```python
periodic = c1f_labeled_asymptotics(
    "forall x: ((exists=0 y: (f(y) = x & TRUE)) | "
    "(exists=3 y: (f(y) = x & TRUE)))",
    expansion_order=4,
)

print(periodic.periodicity)        # 3
print(periodic.residue_powers)     # (-0.5, None, None)
print(periodic.residue_term(102))
print(periodic.residue_expansions)
```

The analyzer also combines independent periodic critical components when they
have a common exact orbit. Repeated isomorphic blocks can be certified
automatically.

## Interval certificates

Install the `certify` extra and pass `certify=True` to request rigorous Arb
enclosures:

```python
certified = c1f_labeled_asymptotics(
    "forall x: exists<=2 y: (f(y) = x & TRUE)",
    certify=True,
    precision_bits=192,
    max_precision_bits=768,
    expansion_order=4,
)

certificate = certified.certificate
print(certificate.radius)
print(certificate.leading_constant)
print(certificate.residue_expansions)
```

The backend uses Arb automatic differentiation, adaptive Krawczyk boxes,
interval Perron and Collatz inequalities, exact modular support calculations,
complex root-of-unity evaluation, and formal interval Laurent-Puiseux
recurrences. Precision may be escalated through `max_precision_bits`.

`certificate.verified_properties` states exactly what was proved.
`certificate.dominance_certified` is true only when the construction also
establishes the complete dominant singularity set. A false flag can accompany
a valid local characteristic-point certificate. No certificate supplies a
finite-`n` remainder bound. If the requested interval proof is inconclusive,
the call raises `C1FAsymptoticCertificationError`.

## Covered regimes

The automatic analyzer covers regular rank-one and multitype tree-critical
systems, reducible systems, independent critical components, universal
guarded predecessor constraints, fixed and saturated capped global states,
nullary branch sums, periodic root-of-unity orbits, tree-independent cycle
poles of arbitrary finite order, and simple cycle poles over analytic trees.
Higher-arity relational symbols use the labeled rigidity transfer.

The implementation rejects nested nonlinear or degenerate critical
components, entire labeled EGFs, and unsupported exact transcendental
cancellations. Subleading terms after extracting a fixed global cardinality
need a bivariate marker recurrence and are not yet automatic. These are
reported as errors rather than assigned heuristic constants.
