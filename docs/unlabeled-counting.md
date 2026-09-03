# Unlabeled counting

UnaryFOMC counts isomorphism classes directly for `C1_=[f]` sentences whose
relation symbols are nullary or unary. It does not obtain this count by dividing
the labeled count by `n!`.

## Finite logical state

For a fixed sentence, the compiler constructs a finite set `Q` of forward
profiles. A profile records the unary colours and equality pattern visible in

```text
x, f(x), ..., f^d(x),
```

where `d` is the largest function-term depth in the sentence. The relation
`p -> q` records whether `q` can be the profile of the successor of an element
with profile `p`.

The equality patterns are generated directly. Besides the unrepeated orbit
window, each pattern has a unique tail length `mu` and cycle length `lambda`.
This gives `1 + d(d+1)/2` equality patterns instead of requiring an enumeration
of all `B_(d+1)` set partitions. Universal profile filters run during this
generation, before the successor matrix and guarded types are constructed.

Counting quantifiers are represented by a finite capped commutative monoid
`M`. Each profile contributes a state `e(q)`, and component states combine by
capped addition. The final accepting projection retains exactly the states that
satisfy the sentence.

## Adams operations

An unlabeled multiset may contain repeated isomorphic objects. Repeating an
object `k` times repeats both its number of vertices and its logical state. The
`k`th Adams operation therefore sends

```text
x^m Y^u  ->  x^(k*m) Y^(k*u),
```

where multiplication of `u` means `k`-fold capped addition. For a series `G`
with zero constant term, the unlabeled multiset construction is

```text
MSET(G) = exp(sum(k >= 1, Adams_k(G) / k)).
```

Only `k <= n` contributes when computing through degree `n`.

## Rooted in-trees

Every transient vertex lies in a rooted in-tree directed toward a cycle. For an
infinity profile `q`, the rooted-tree ordinary generating function is

```text
T_q = x weight(q) MSET(sum(p -> q, T_p)).
```

The same multiset of children decorates a vertex that will lie on a cycle. The
leading root factor makes degree-by-degree computation well founded: the
coefficient of degree `m` depends only on tree coefficients below degree `m`.

## Directed cycles

For an actual cycle length `ell`, only the profile set `Q_ell` is admissible.
Burnside's lemma averages over rotations. A rotation with order `k` dividing
`ell` has `ell/k` position orbits, and every decoration in an orbit repeats `k`
times. Its contribution is obtained by applying `Adams_k` to the cycle-vertex
decorations and taking the trace of the compatible transition matrix to the
power `ell/k`.

The connected-component series is the sum of these Burnside averages for
`1 <= ell <= n`. A complete functional digraph is then an unlabeled multiset of
connected components, so one final `MSET` operation produces the ordinary
generating function.

After applying the accepting projection, the coefficient of `x^n` is the exact
unlabeled model count. There is no factorial multiplier.

## Implementation map

- `_series_adams`: repeats size and capped state;
- `_series_multiset`: implements the Polya multiset exponential;
- `_CompiledMonadicC1F.evaluate_unlabeled`: rooted trees, Burnside cycles, the
  outer component multiset, and acceptance;
- `compile_c1f_ufomc` and `c1f_ufomc`: enforce the monadic scope;
- `examples/unlabeled_counts.py`: reproduces initial exact sequences.

All series are truncated at the requested domain size. The profile and state
spaces are fixed by the sentence, while Adams indices and cycle lengths are at
most `n`; the resulting exact coefficient algorithm is polynomial-time in `n`
for every fixed supported sentence.

## Why higher arity is rejected

The labeled compiler can replace inspected higher-arity orbit traces by unary
data and restore unconstrained relation entries with a power-of-two factor.
That factor is not valid after quotienting by isomorphism: the number of orbits
of free relation entries depends on the automorphism group of the functional
skeleton. With the identity function and one unconstrained binary relation, the
problem already contains unlabeled directed graphs with loops.

UnaryFOMC therefore raises an explicit error for exact unlabeled requests that
contain a relation symbol of arity at least two.
