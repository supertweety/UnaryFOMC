# Guarded predecessor counting

UnaryFOMC extends `C1_=[f]` with local predecessor-count formulas

```text
exists<cmp>m y: (f(y) = x & C(y))
```

where `<cmp>` is one of `=`, `<`, `<=`, `>`, or `>=`. An unbounded
`exists y` is interpreted as `exists>=1 y`. The anchor `x` and witness `y`
must be distinct. After removing the guard, the filler `C(y)` may have only
`y` free. It may contain unary atoms, equality between forward `f`-terms,
Boolean connectives, and further guarded predecessor counts. Labeled counting
also permits arbitrary fixed-arity relation atoms whose arguments are forward
terms rooted at `y`.

For example, nested counts can be written directly:

```text
forall x: (P(x) <-> exists>=1 y: (
    f(y) = x & exists=1 z: (f(z) = y & P(z))
))
```

The parser recognizes the guard in either conjunct order and normalizes it to
`C1FPredecessorCount`. Unguarded formulas that retain both variables, such as
`exists y: R(x,y)`, are rejected.

## Compilation

Let the sentence contain local predecessor-count subformulas
`alpha_1, ..., alpha_s`. Each ordinary forward profile is refined with `s`
truth bits, one for each `alpha_i`. A guarded type therefore specifies both
the forward behavior visible from a vertex and the local count statements that
must hold at that vertex.

For every guarded type `tau`, the compiler computes a contribution vector
`a(tau)` in `{0,1}^s`. Its `i`-th coordinate says whether a child of type
`tau` satisfies the filler of `alpha_i`. Coordinates are capped at `m_i + 1`,
where `m_i` is the corresponding threshold. The labeled tree recurrence uses
constrained sets of compatible child trees; the unlabeled recurrence uses
constrained multisets. A parent type is retained exactly when its capped child
vector realizes all of its declared truth bits.

On a directed cycle, the previous cycle vertex is also a predecessor. Cycle
decorations therefore depend on an edge `(previous, current)`: the compiler
adds `a(previous)` to the off-cycle child vector before checking the local
constraints of `current`. Labeled counts use traces of powers of this transfer
matrix. Unlabeled counts apply Adams operations to the accepted edge transfers
and use the usual Burnside average over cycle rotations.

The existing forward-profile evaluator remains the fast path for sentences
without guarded predecessor counts. A population-vector representation is not
used for the extension because profile populations do not retain the local
indegree information needed by these formulas.
