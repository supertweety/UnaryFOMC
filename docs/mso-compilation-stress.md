# A Nonelementary Compilation Witness

This benchmark family has a provable nonelementary lower bound on the size
of any explicit, domain-size-independent composition algebra recognizing
its sentences. It is not just a collection of examples that happen to
time out. The lower bound applies even to sentence-specific, perfectly
minimized algebras, not only to complete bounded-rank theory compilation.

The construction adapts the hereditary-tree comparison used in Section 8,
Lemma 30 of [Frick and Grohe, *The complexity of first-order and monadic
second-order logic revisited*, 2004](https://www.labri.fr/perso/courcell/CoursMaster/FrickGrohe.pdf).
Here the trees are uncolored, their parent relation is one total unary
function with a self-loop at each root, and disjoint-union contexts provide
a direct state-output lower bound. This is distinct from that paper's
conditional model-checking lower bounds.

## The Sentences

Write

\[
 C(u,x) := (f(u)=x \land u\ne x)
\]

for a proper child. Define the bounded-depth branch signature recursively:

\[
 b_0(x)=*,\qquad
 b_{h+1}(x)=\{b_h(u):C(u,x)\}.
\]

These are sets, not multisets: multiple children with the same signature
contribute a single member. No unary predicates or vertex labels are used.

Let \(Q_h(x,y)\) assert equality of these signatures. A linear-size formula
is given by \(Q_0(x,y)=\top\) and

\[
\begin{split}
Q_{h+1}(x,y) := \forall u\bigl[
 &(C(u,x)\lor C(u,y))\ \to\\
 &\exists v\bigl(
 [(C(u,x)\land C(v,y))\lor(C(u,y)\land C(v,x))]
 \land Q_h(u,v)\bigr)\bigr].
\end{split}
\]

For distinct parents, a vertex cannot be a child of both because f is a
function. Thus every child on either side must have an equivalent child
on the opposite side. If x=y the same formula also holds, by taking v=u.
Induction gives \(Q_h(x,y)\) exactly when \(b_h(x)=b_h(y)\).

The recursive subformula occurs **once**, not twice. At each level the two
old variable names can be reused inside the recursive subformula, since
the old parents are no longer needed there. Consequently four first-order
variable names suffice and the expanded sentence has length O(h), with no
macros, DAG sharing, large numerical constants, or long function terms.

The benchmark sentence is

\[
 \varphi_h := \exists x\exists y\,
 (x\ne y\land f(x)=x\land f(y)=y\land Q_h(x,y)).
\]

In words: **two distinct fixed-point components have equal depth-h branch
signatures**. The sentence imposes no restriction on the other components.
It uses no second-order quantifier at all: this is an FO subfamily handled
by the full MSO backend. It is outside the guarded one-variable syntax
because it compares independently chosen branches.

For the first level the generator simplifies the comparison to

```text
(exists u: (f(u)=x & u!=x)) <-> (exists v: (f(v)=y & v!=y))
```

Thus phi_0 asks for at least two fixed points. Phi_1 asks for two fixed
points which either both have proper children or both have none. Phi_2
compares the sets of leaf/nonleaf child kinds. Higher levels recursively
compare sets of the preceding kinds, still ignoring multiplicities.

## The Tower Lower Bound

Let

\[
 t_0=1,\qquad t_{h+1}=2^{t_h}.
\]

There are exactly t_h possible depth-h branch signatures. For realizability,
start with one leaf at depth zero. Given one representative rooted tree for
each depth-h signature, make a fresh root whose children are one copy of
each representative from any chosen subset. This realizes every subset,
so there are exactly \(2^{t_h}\) signatures at the next level. Representatives
have height at most h. Make each root a fixed point to obtain a total
function component.

For each signature a choose one such component T_a. For any subset S of
the t_h signatures, let A_S be the disjoint union of one T_a for every
\(a\in S\). Then A_S rejects phi_h: its fixed points have pairwise distinct
signatures. But for any signature a,

\[
 A_S\sqcup T_a\models\varphi_h \quad\Longleftrightarrow\quad a\in S.
\]

If S and R differ, choose a in their symmetric difference. The context
that adds T_a distinguishes A_S from A_R. In any sound deterministic
composition algebra their complete-graph states must therefore differ.
This proves at least \(2^{t_h}\) distinct rejecting graph states. An
accepting structure, for example two copies of a leaf component, needs
one additional state. Hence

\[
 \boxed{|G_{\varphi_h}|\ \ge\ 2^{t_h}+1=t_{h+1}+1.}
\]

The bound is exact for the graph sort: remember the set of fixed-point
signatures encountered so far until a repetition occurs, then enter one
absorbing accepting state. Non-fixed-point cycle components contribute
the identity. Disjoint union unions the sets and detects intersections.

| h | Source characters in the generator | Necessary graph states |
| --- | ---: | ---: |
| 0 | 43 | 3 |
| 1 | 100 | 5 |
| 2 | 245 | 17 |
| 3 | 390 | 65,537 |
| 4 | 535 | 2^65,536 + 1 |
| 5 | 680 | 2^(2^65,536) + 1 |

The source size is linear while the height of the exponential tower grows
with h. No fixed-height exponential bound in the source length can bound
these state counts. An algorithm writing an explicit specification must
at least produce that many states; in our representation it also writes
the quadratic graph-addition table. Already h=3 needs 4,295,098,369 entries
in that table alone, even with the smallest possible graph state set.

This is an unconditional **explicit output-size** argument. It does not
prove that all algorithms for counting phi_h at a specified finite n must
do this work. A size-dependent procedure, symbolic tables, or a different
representation may avoid materializing these states. Nor does it say that
every input at a given depth is hard. The sentences phi_h are all trivially
satisfiable, so this family alone is not a satisfiability-hardness reduction.

## Measurements

The generic compiler was run with its usual limits of 5,000 intermediate
states and 2,000,000 construction-work units. Every worker also had a
15-second wall-clock limit. The following are local Python 3.14 results,
not a controlled performance benchmark:

| h | Generic compiler result | Measured time | Final (F,H,D,G) |
| --- | --- | ---: | --- |
| 0 | Compiled | 0.007 s | (1,1,2,3) |
| 1 | Compiled | 0.032 s | (2,2,3,5) |
| 2 | Work limit | 2.29 s | No specification returned |
| 3 | Work limit | 2.53 s | No specification returned |
| 4 | Work limit | 2.50 s | No specification returned |

Successful times measure compilation only; failure times measure worker
time until the limit. The full report additionally records process-launch
overhead, limits, statistics, and counts through size four in
[the JSON results](../examples/mso_compilation_stress_results.json).
The h=2 case also exceeded a separately tested work limit of 20,000,000.

**These early failures are not the proof of nonelementarity.** In
particular, h=2 has a tiny final graph algebra. Its failure demonstrates
intermediate overhead in the generic formula compiler. Conversely, at
h=3 and above, the proved final-state requirements themselves exceed the
default state limit. Improvements to intermediate construction cannot
remove the tower across this whole family.

## Independent Checks

The example includes a specialized specification constructor for small h,
independent of the generic sentence compiler. It tracks sets of truncated
child signatures, distinguishes a one-tree dotted segment from a longer
cycle segment, and uses the graph subset/duplicate monoid above. It
refuses to materialize large graph tables.

For h=2 its state counts are **(4,4,5,17)**, only 30 states in total. Using
the ordinary exact coefficient evaluator gives, for n=0,...,8:

```text
labeled:   0, 0, 1, 1, 31, 321, 4151, 72913, 1547295
unlabeled: 0, 0, 1, 1, 4, 9, 25, 61, 171
```

This constructor is an independent reference for this family, not an
optimization added to `compile_mso`. It exhibits how much room remains
for improving the generic compiler on its small instances.

Five regression tests check formula length and variable reuse, compare
direct formula semantics with independently calculated branch signatures,
exercise all subset-distinguishing contexts through h=2, and construct all
16 depth-three signature representatives. The specialized counts are
checked against exhaustive labeled functions through n=5 and explicit
isomorphism orbits through n=4. For h=0,1, generic and specialized
specifications also agree in both modes through n=8.

## Reproduce

From an installed checkout:

```bash
python examples/mso_compilation_stress.py --depths 0 1 2 3 4
python examples/mso_compilation_stress.py --depths 2 --formula-only
python examples/mso_compilation_stress.py --depths 2 --max-transitions 20000000
```

With no editable installation, prefix these commands with `PYTHONPATH=src`.
`--output report.json` records a JSON array. Workers are separate processes
and are terminated on wall-time exhaustion; compilation-limit exceptions
are reported as failures, never as partial counts. Do not disable limits
in an attempt to complete the higher-level instances.

For the small specialized reference:

```python
# Run from the checkout with PYTHONPATH=src:examples.
from mso_compilation_stress import duplicate_branches_specification
from unaryfomc import evaluate_mso_specification

s = duplicate_branches_specification(2)
print(s.state_counts)
print(evaluate_mso_specification(s, 8, labeled=False))
```
