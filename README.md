# UnaryFOMC

UnaryFOMC is a research package for exact finite-model counting in
one-variable first-order logic with counting, equality, and one total unary
function. The supported base fragment is denoted `C1_=[f]`.

Given a sentence `phi` and a domain size `n`, UnaryFOMC can compute:

- the number of labeled models of `phi` on the domain `[n]`; and
- for monadic relational vocabularies, the number of `n`-element models up to
  isomorphism.
- effective labeled and unlabeled asymptotic expansions in the regular
  finite-radius regimes developed in Part II; and
- optional Arb interval certificates for the recognized asymptotic regimes.

Both counts are exact integers. Unlabeled counting is performed directly; it
is not obtained by dividing the labeled count by `n!`.

## Papers

UnaryFOMC accompanies the following work:

1. **Part I:** [*Unary Functions, Automorphisms, and Unlabeled First-Order
   Model Counting*](https://arxiv.org/abs/2608.30580).
2. **Part II:** companion paper in preparation. **Title and arXiv link
   forthcoming.**

<!-- Replace the Part II placeholder above when the manuscript is public. -->

Part II develops the generating-function and exact unlabeled constructions,
guarded predecessor counting, and further extensions implemented in this
repository.

## Installation

UnaryFOMC requires Python 3.10 or newer and has no mandatory third-party
dependencies. From a checkout of this repository, run:

```bash
python -m pip install .
```

For development, use an editable installation:

```bash
python -m pip install -e .
```

The optional `speed` extra installs GMP-backed exact arithmetic through
`gmpy2`:

```bash
python -m pip install -e '.[speed]'
```

On Python 3.11 or newer, the optional `certify` extra installs the Arb-backed
interval path through `python-flint`:

```bash
python -m pip install -e '.[certify]'
```

Install `.[full]` to enable both optional backends.

## Quick Start

The following sentence requires exactly three elements to lie on directed
cycles of length three:

```python
from unaryfomc import c1f_fomc

sentence = "exists=3 x: (f(f(f(x))) = x & f(x) != x)"
count = c1f_fomc(sentence, 4)

print(count)  # 32
```

Use `c1f_ufomc` to count isomorphism classes. For the tautological sentence,
this gives the classical numbers of unlabeled endofunctions:

```python
from unaryfomc import c1f_ufomc

values = [int(c1f_ufomc("TRUE", n)) for n in range(8)]
print(values)  # [1, 1, 3, 7, 19, 47, 130, 343]
```

When several domain sizes are needed, compile the sentence once:

```python
from unaryfomc import compile_c1f_fomc, compile_c1f_ufomc

labeled = compile_c1f_fomc("forall x: x = f(x)")
unlabeled = compile_c1f_ufomc("forall x: x = f(x)")

print([int(labeled.fomc(n)) for n in range(6)])
print([int(unlabeled.ufomc(n)) for n in range(6)])
```

## Supported Logic

Terms are formed by iterating one unary function symbol. The parser accepts
the following constructs.

| Construct | Example |
| --- | --- |
| Boolean constants | `TRUE`, `FALSE` |
| Relation atoms | `P(x)`, `R(x,f(x))`, `Z` |
| Function-term equality | `f(f(x)) = x`, `f(x) != x` |
| Boolean connectives | `~`, `&`, `\|`, `->`, `<->` |
| Universal quantification | `forall x: C(x)` |
| Existential quantification | `exists x: C(x)` |
| Counting quantification | `exists=3 x: C(x)`, `exists>=2 x: C(x)` |
| Guarded predecessor counting | `exists=2 y: (f(y) = x & C(y))` |

All comparisons `=`, `<`, `<=`, `>`, and `>=` are available for counting
quantifiers. Parentheses may be used freely. A different function symbol can
be selected with the `function_symbol` API argument or the CLI option
`--function-symbol`.

The formula must remain one-variable after accounting for bound variables.
For labeled counting, relation symbols may have arbitrary fixed arity, but all
arguments of a local atom are forward terms rooted at the same variable. Exact
unlabeled counting supports only nullary and unary relation symbols.

## Guarded Predecessor Counts

UnaryFOMC also supports local formulas of the form

```text
exists<cmp>m y: (f(y) = x & C(y))
```

where `<cmp>` is a counting comparison. After crossing the guard `f(y) = x`,
the old anchor `x` must be forgotten: the filler `C(y)` may use forward terms
rooted at `y`, unary predicates, equality, Boolean connectives, and nested
guarded predecessor counts.

For example, the following sentence says that every element has a predecessor.
On a finite domain this forces `f` to be a permutation:

```python
from unaryfomc import c1f_fomc, c1f_ufomc

sentence = "forall x: exists>=1 y: (f(y) = x & TRUE)"

labeled = [int(c1f_fomc(sentence, n)) for n in range(6)]
unlabeled = [int(c1f_ufomc(sentence, n)) for n in range(7)]

print(labeled)    # [1, 1, 2, 6, 24, 120]
print(unlabeled)  # [1, 1, 2, 3, 5, 7, 11]
```

See [Guarded predecessor counting](docs/guarded-predecessor-counting.md) for
the precise syntactic boundary and compilation method.

## Counting Semantics

The unary function is a genuine total function and is always part of the
counted vocabulary, even when it is not mentioned in the sentence. Thus
`c1f_fomc("TRUE", n)` counts all endofunctions: it returns `n**n` for positive
`n`, and one for the empty domain.

Labeled counting treats the domain as a fixed `n`-element set. Unlabeled
counting identifies structures related by a bijection preserving the function
and every predicate. Nullary predicates are supported in both modes.

The current package performs unweighted model counting. All accepted
interpretations of predicates and the unary function contribute weight one.

## Effective Asymptotics

Use `c1f_labeled_asymptotics` for the labeled expansion

```text
FOMC(phi, n) ~ C * n! * growth**n * n**power.
```

For example, bounding every indegree by two gives a guarded tree-critical
system:

```python
from unaryfomc import c1f_labeled_asymptotics

sentence = "forall x: exists<=2 y: (f(y) = x & TRUE)"
analysis = c1f_labeled_asymptotics(sentence, expansion_order=4)

print(analysis.regime)             # guarded-tree-critical
print(analysis.radius)             # 0.4142135623730951...
print(analysis.exponential_growth) # 2.414213562373095...
print(analysis.leading_constant)   # 0.5212432086958386...
print(analysis.expansion_estimate(100))
```

`expansion_order` requests successive half-orders after the leading term.
Periodic systems are represented exactly by residue class through
`periodicity`, `residue_expansions`, and `residue_term(n)`. For instance, if
every indegree is either zero or three, models occur only when `n` is a
multiple of three:

```python
periodic = c1f_labeled_asymptotics(
    "forall x: ((exists=0 y: (f(y) = x & TRUE)) | "
    "(exists=3 y: (f(y) = x & TRUE)))"
)
print(periodic.periodicity)    # 3
print(periodic.estimate(100))  # 0.0
print(periodic.estimate(102))  # positive
```

Monadic vocabularies also support direct unlabeled Polya asymptotics:

```python
from unaryfomc import c1f_unlabeled_asymptotics

unlabeled = c1f_unlabeled_asymptotics("TRUE")
print(unlabeled.exponential_growth)  # 2.955765285651995...
print(unlabeled.power)               # -0.5
print(unlabeled.leading_constant)    # 0.4428767697822064...
```

Pass `certify=True` to either entry point to attach rigorous interval
enclosures where the automatic Arb backend recognizes the required
certificate:

```python
certified = c1f_labeled_asymptotics(
    sentence,
    certify=True,
    precision_bits=192,
    expansion_order=4,
)
print(certified.certificate.radius)
print(certified.certificate.verified_properties)
print(certified.certificate.dominance_certified)
```

The certificate's `verified_properties` field is the precise proof inventory.
`dominance_certified` distinguishes a complete global singularity proof from
a local characteristic-point certificate. No returned certificate currently
includes a finite-`n` remainder bound. An inconclusive certification attempt
raises `C1FAsymptoticCertificationError`; it is never silently reported as a
proof.

The analyzer covers regular reducible and multitype systems, independent
critical components, capped global states, nullary branch sums, guarded
predecessor systems, periodic root-of-unity orbits, and finite-order cycle
poles. See [Labeled asymptotics](docs/labeled-asymptotics.md) and [Unlabeled
asymptotics](docs/unlabeled-asymptotics.md) for the detailed scope.

## Command Line

Installation provides the `unaryfomc` command:

```bash
unaryfomc 'exists=3 x: (f(f(f(x))) = x & f(x) != x)' 4
unaryfomc 'TRUE' 10 --unlabeled
```

Useful options include:

```text
--unlabeled              count isomorphism classes
--function-symbol NAME   use a function symbol other than f
--profile-statistics     print compiler profile counts
--no-profile-pruning     disable universal-profile pruning
--no-twin-compression    disable exact twin compression
```

Run `unaryfomc --help` for the complete interface.

## How It Works

For a fixed sentence, the compiler records the equality pattern and unary
colors visible along each finite orbit window

```text
x, f(x), ..., f^d(x),
```

where `d` is the largest function-term depth. Equality patterns are generated
directly from their canonical tail-cycle schemes. Universally forced
conditions are applied during generation, and successor compatibility is
indexed by overlapping orbit windows. Formula-sensitive twin compression can
then quotient indistinguishable profiles.

Labeled functional digraphs are assembled from directed cycles decorated by
compatible rooted in-trees using multitype exponential generating functions.
Counting quantifiers are carried by a finite capped-state algebra. Guarded
predecessor formulas refine the profiles with local count states.

For unlabeled counting, sets of children and connected components become
Polya multisets. Adams operations repeat both object size and logical state,
while Burnside averages account for rotations of directed cycles. This yields
an ordinary generating function whose coefficient is the number of
isomorphism classes.

For every fixed supported sentence, coefficient extraction is polynomial in
the domain size. Compilation is formula dependent and may be exponential in
the term depth or in the number of unary predicates; this dependence is not
hidden by the public API.

See [Unlabeled counting](docs/unlabeled-counting.md) for the construction and
its correspondence with the implementation.

## API Overview

The main entry points are:

- `c1f_fomc(formula, n)` for one labeled count;
- `c1f_ufomc(formula, n)` for one unlabeled count;
- `compile_c1f_fomc(formula)` for repeated labeled evaluation;
- `compile_c1f_ufomc(formula)` for repeated unlabeled evaluation;
- `compile_c1f_labeled_specification(formula)` for the finite labeled EGF
  system;
- `c1f_labeled_asymptotics(formula)` for labeled expansions;
- `c1f_unlabeled_asymptotics(formula)` for unlabeled expansions;
- `analyze_c1f_labeled_asymptotics(formula)` as the advanced labeled entry
  point;
- `parse_c1f_formula(...)` and `parse_c1f_sentence(...)` for syntax trees; and
- `render_c1f_formula(...)` for normalized textual output.

Compiled objects expose `profile_statistics()` for inspecting raw profiles,
profiles surviving logical pruning, twin classes, and guarded types.

## Reproducing the Examples

The Part I example driver prints exact values for the prescribed-cycle,
quasi-kernel, and bounded-height examples:

```bash
python examples/paper_examples.py --max-n 15
```

The unlabeled driver prints the initial unlabeled endofunction sequence and a
unary-colored identity example:

```bash
python examples/unlabeled_counts.py --max-n 15
```

The asymptotics driver exercises labeled, unlabeled, and periodic analysis:

```bash
python examples/asymptotics.py
python examples/asymptotics.py --certify
```

## Tests

The tests compare the algorithms with literal finite-structure enumeration
and, in the unlabeled case, literal canonicalization of isomorphism classes:

```bash
python -m unittest discover -s tests -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## Scope and Status

UnaryFOMC is research software and currently has an alpha API. In particular:

- only one total unary function symbol is supported;
- exact unlabeled counting is restricted to monadic relational vocabularies;
- counting is currently unweighted;
- large formula-dependent profile or capped-state spaces can be expensive.

The automatic asymptotic implementation deliberately rejects nested nonlinear
or degenerate critical components, entire labeled EGFs, and radius-one monadic
unlabeled OGFs. Exact transcendental cancellations may require a structural
identity beyond interval arithmetic. Subleading terms after extracting a
fixed global cardinality still require a bivariate marker recurrence. These
are reported explicitly rather than guessed.

Counts within the supported fragment are exact. The parser, diagnostics, and
public API may continue to evolve alongside the papers.

## Citation

Please cite the accompanying papers when using UnaryFOMC. Machine-readable
metadata for Part I is provided in [CITATION.cff](CITATION.cff); it will be
updated when Part II becomes public.
