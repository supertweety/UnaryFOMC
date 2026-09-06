# Exact Counting With Monadic MSO

For five small compiled examples expressing global properties, see
[the MSO gallery](mso-global-gallery.md).

The MSO backend counts finite structures with one **total unary function**
and finitely many free unary or nullary predicates. Both labeled counting
on `[n]` and direct counting of isomorphism classes are supported. It is
separate from the specialized guarded C1 backend, which remains the default.

## API

```python
from unaryfomc import compile_mso, mso_fomc, mso_ufomc

sentence = "exists X: forall x: (X(f(x)) <-> ~X(x))"
compiled = compile_mso(sentence)

assert compiled.fomc(6) == 9945
assert compiled.ufomc(6) == 27
assert mso_fomc(sentence, 4) == 57
assert mso_ufomc(sentence, 4) == 5
print(compiled.sequence(8, labeled=False))
print(compiled.statistics)
assert compiled.has_model()  # Existence at any positive finite size.
```

`sequence(N)` returns an exact tuple for sizes `0,...,N`. Labeled and
unlabeled prefixes are cached separately. Increasing the bound recomputes
the longer prefix; there is currently no incremental extension of a series.
`compile_mso_fomc` and `compile_mso_ufomc` are aliases in behavior: the same
compiled object supports both counting modes. The corresponding
`evaluate_mso_fomc` and `evaluate_mso_ufomc` wrappers are also exported.

The empty domain is included when asking for size zero. Quantification has
its usual empty-domain semantics, including the single empty subset.
`has_model()` asks for a **nonempty** model, while
`has_model(nonempty=False)` also permits the empty model. This query uses
finite-state reachability, not search up to a domain-size bound.

## Syntax

| Construct | Syntax |
| --- | --- |
| Elements | `exists x: ...`, `forall x,y: ...` |
| Sets | `exists X: ...`, `forall X,Y: ...` |
| Explicit sorts | `exists set s: ...`, `forall node X: ...` |
| Membership | `X(x)`, `x in X` |
| Free unary predicate | `P(x)`, `P(f(x))` |
| Free nullary predicate | `Q()`, `Q` |
| Equality | `x=y`, `f(x)!=f(y)` |
| Iterated terms | `f(f(x))`, `f^2(x)` |
| Graph of the function | `E(x,y)`, meaning `f(x)=y` |
| Boolean operations | `TRUE`, `FALSE`, `~`, `&`, `\|`, `->`, `<->` |

Uppercase quantified names default to sets; other quantified names default
to elements. Explicit `set`/`node` overrides this convention. A colon or
dot separates a binder from its body. Quantifiers extend over the following
formula; parenthesize a quantified subformula when combining it with an
outside connective. Shadowing is supported. The renderer emits explicit
sorts and parentheses for an unambiguous round trip.

All element variables must be bound. Set equality can be written
`forall x: (X(x) <-> Y(x))`. Fixed cardinalities can be expressed using
distinct element witnesses, but native `exists=...` counting syntax belongs
to the specialized C1 backend and is not accepted by this parser.

`function_symbol="g"` selects a different function name, and
`edge_predicate="Edge"` selects its graph predicate. The function and graph
names are reserved. `E` is not an additional arbitrary binary relation.
Extra free relations of arity greater than one and second function symbols
are rejected.

Free unary predicates are counted as part of each structure. Quantified
sets are only witnesses: they are projected away **before** enumeration.
Their number, and whether an automorphism preserves a witness, does not
affect the count. A two-cycle has two alternating sets; the even-cycle
sentence nevertheless has exactly one unlabeled model of size two.

The optional `predicates=("P", "Q")` argument retains additional unused
unary symbols in the counted vocabulary. For example,
`compile_mso("TRUE", predicates=("P",)).fomc(n)` is `2**n * n**n`.
Nullary symbols occurring in the sentence are handled by global truth-value
case splitting, including on the empty domain.

## Compilation And Evaluation

The evaluator implements the four-sorted finite specification of Part II,
Section 6, equations (32)--(43). Its states describe forests, rooted trees,
dotted cycle segments, and complete function graphs. Tables implement
disjoint union, adding a colored root, forgetting its designation,
concatenating dotted segments, and closing the cycle.

The practical compiler builds a **sentence-specific accepting congruence**,
not the full spaces of bounded-rank MSO theories from the theoretical
compiler in Section 6.3:

1. Alpha-rename variables and translate iterated terms into paths in the
   graph of the total function.
2. Represent element variables by singleton unary markers. Finite atomic
   algebras recognize membership, equality, and an edge between markers.
3. Implement Boolean combinations by reachable products and complement.
4. Existentially forget set markers by powerset projection on all four
   sorts. For element quantifiers, require a singleton before projection.
   Universal quantifiers use complement, projection, and complement.
5. Saturate reachable states and minimize by refinement over constructor
   contexts after each algebra construction.

The induction behind projection is important. A projected state records
the **set of possible colored states**, not their multiplicities. Disjoint
pieces choose witnesses independently; each constructor applies to all
combinations of possible states. Even isomorphic copies of a piece can
choose different witnesses. Every concrete uncolored structure therefore
has one deterministic projected state. Acceptance means that at least one
possible state accepted before projection. Boolean operations preserve
recognition, and context minimization preserves all subsequent constructor
results and final acceptance.

Counts are then obtained from these deterministic states using exact
generating-function coefficients. Labeled children and components use SET;
unlabeled children and components use MSET. Cycles use directed cyclic
rotation, not reflection. In every unlabeled repetition, **both degree and
algebra state** are repeated. Graph acceptance is applied only after the
complete component SET/MSET, not separately to components. Arithmetic uses
Python integers and `Fraction`, with an integrality check on every final
count. The counter neither enumerates input structures nor divides labeled
counts by `n!` to obtain unlabeled ones.

For fixed explicit tables, coefficient evaluation has polynomial bit
complexity in the table size and requested degree. The implementation uses
straightforward truncated convolutions, not fast power-series algorithms.
The nonelementary worst-case compilation barrier remains: repeated
powerset projection can be very expensive. This implementation does not
claim to attain the paper's particular bounded-rank compiler bound.

## Limits And Inspection

```python
from unaryfomc import MSOCompilerLimits, MSOCompilationLimitError, compile_mso

limits = MSOCompilerLimits(max_states=10000, max_transitions=4000000)
compiled = compile_mso("forall x: f(x)=x", limits=limits)
print(compiled.statistics.branch_state_counts)
```

Default limits are 5,000 total states across the four sorts of any single
intermediate algebra, 2,000,000 units of accumulated construction work,
1,024 colors in any intermediate alphabet, and 5,000 formula nodes (also
accounting for term expansion). Work counts include constructor evaluation
and the Cartesian products in binary powerset transitions; they are not a
wall-clock limit. A limit can be disabled explicitly by setting it to
`None`. Exceeding a limit raises `MSOCompilationLimitError` without returning
any approximate or partial count. These are practical guards, not a promise
that every formula within them fits in memory or the Python recursion limit.

`compiled.specification` exposes the minimized tables when there is one
nullary case. Otherwise use `compiled.branches`. `to_dict()` and
`MSOSpecification.from_dict(...)` support JSON serialization. Validation
checks dimensions, state ranges, and optionally algebraic laws; it cannot
prove that an externally supplied table correctly represents a sentence.
`evaluate_mso_specification` evaluates one such explicit specification.

The command line accepts `--logic mso`, `--unlabeled`,
`--compiler-statistics`, `--mso-max-states`, and `--mso-max-transitions`.
Profile-pruning flags apply only to the C1 backend.

## Scope And Validation

Tests independently enumerate small total functions and predicate
interpretations, evaluate formulas directly, and canonicalize models under
all domain permutations. They cover quantifier alternation, shared
witnesses, shadowing, nullaries, empty domains, cycle rotations, and
repetition-sensitive child/component states. Classical sequences and
closed formulas provide additional checks through larger sizes.

MSO support is currently **unweighted exact counting and finite
satisfiability**. It adds no MSO asymptotic API, interval certificates,
binary second-order quantifiers, second function symbol, or higher-arity
free relation support. Existing C1/guarded asymptotic APIs are unchanged.

For a concrete linear-size sentence family with a proved tower-sized
explicit-state requirement, bounded benchmark runs, and an independent
small-instance reference, see [the compilation stress test](mso-compilation-stress.md).
