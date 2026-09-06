# Five Compact Global MSO Examples

Each example is specified by a fixed sentence, independent of n, over one
total unary function. There are no external connectedness or tree axioms.
All counts below are for n=1,...,8. L means labeled models, U means
isomorphism classes. Rooted trees have edges pointing toward the root and
a self-loop at the root.

The [runnable gallery](../examples/mso_global_gallery.py) contains fully
expanded parser-compatible sentences. Its [recorded results](../examples/mso_global_gallery_results.json)
include counts for n=0,...,8, compilation statistics, and local timings.

## Logical Abbreviations

Reachability along the function, including a path of length zero, is

\[
R(x,y):=\forall X\bigl[(X(x)\land
  \forall z(X(z)\to X(f(z))))\to X(y)\bigr].
\]

Belonging to the same weakly connected functional component is

\[
K(x,y):=\forall X\bigl[(X(x)\land
  \forall z(X(z)\leftrightarrow X(f(z))))\to X(y)\bigr].
\]

The sets in the second formula are exactly unions of complete components.
Both abbreviations expand to ordinary MSO; neither is a built-in graph
constraint added to the counting engine.

## 1. Rooted Trees Of Arbitrary Height

Require a fixed point reachable from every element:

\[
\exists r\,(f(r)=r\land\forall x\,R(x,r)).
\]

The formula expresses a single rooted tree with no depth bound. The root
is not an extra counted constant; it is determined by the function graph.
The labeled count is n^(n-1) for positive n.

```text
L: 1, 2, 9, 64, 625, 7776, 117649, 2097152
U: 1, 1, 2, 4, 9, 20, 48, 115
```

## 2. Rooted Trees With Every Leaf At Even Depth

Existentially choose a parity set A, put the root in A, alternate membership
on every nonloop edge, and require every leaf to belong to A. A leaf means
an element with no proper children:

\[
\operatorname{Leaf}(x):=\forall y\,(f(y)=x\to y=x).
\]

The complete sentence is

\[
\begin{split}
\exists A\exists r\,[&f(r)=r\land A(r)\land\forall x\,R(x,r)\\
 &\land\forall x\,(f(x)\ne x\to(A(x)\leftrightarrow\neg A(f(x))))\\
 &\land\forall x\,(\operatorname{Leaf}(x)\to A(x))].
\end{split}
\]

Different leaves may occur at different even depths, however large. The
one-vertex tree is included, with its root a leaf at depth zero. A is a
quantified witness and adds no color interpretations to the count.

```text
L: 1, 0, 6, 12, 200, 1470, 20832, 282296
U: 1, 0, 1, 1, 3, 4, 10, 17
```

## 3. Exactly Two Functional Components

Choose representatives in different components, and require every vertex
to lie in one of them:

\[
\exists x\exists y\,[\neg K(x,y)\land
  \forall z\,(K(x,z)\lor K(y,z))].
\]

The two components can have arbitrary cycle lengths and arbitrary incoming
trees. They are not ordered or named in the counted structure: different
choices of x,y do not multiply the count.

```text
L: 0, 1, 9, 95, 1220, 18694, 334369, 6852460
U: 0, 1, 2, 7, 17, 48, 127, 352
```

## 4. All Cycle Lengths Divisible By Three

Existentially quantify three sets A_0,A_1,A_2, require them to partition
the domain, and require

\[
\forall x\;\bigwedge_{i=0}^2
  (A_i(x)\to A_{i+1\bmod 3}(f(x))).
\]

This accepts precisely functions whose cycles have lengths 3,6,9,...,
with unrestricted trees attached. For a function with c components there
are 3^c valid phase assignments, but the function is counted only once:
the phase sets are quantified witnesses, not free predicates.

```text
L: 0, 0, 2, 24, 300, 4480, 78750, 1591296
U: 0, 0, 1, 1, 3, 11, 25, 71
```

## 5. One Marked Cycle Entry Point Per Component

Now retain a **free unary predicate P** as part of the counted structure.
Require every vertex to reach exactly one P-vertex:

\[
\forall x\exists p\,[P(p)\land R(x,p)\land
  \forall q\,((P(q)\land R(x,q))\to q=p)].
\]

This forces exactly one marked vertex on each cycle and no marked
transient vertices. Indeed, a cycle vertex must reach a mark on its own
cycle; any marked transient vertex would then reach both itself and that
cycle mark, violating uniqueness.

These are functional components with distinguished cycle entry points.
Unlike example 4, P is part of the combinatorial object. Distinct markings
are identified only when a function-graph isomorphism preserves them.

```text
L: 1, 5, 40, 437, 6036, 100657, 1965160, 43937385
U: 1, 3, 8, 24, 69, 209, 621, 1878
```

## Size And Validation

The four sorts are forests, rooted trees, dotted cycle segments, and
complete graphs. These are sentence-specific states, not all rank types.

| Example | States (F,H,D,G) | Total | Compilation time |
| --- | --- | ---: | ---: |
| Rooted trees | (1,1,2,3) | 7 | 0.007 s |
| Even-depth leaves | (4,3,2,3) | 12 | 0.045 s |
| Exactly two components | (1,1,1,4) | 7 | 0.019 s |
| Cycle lengths divisible by three | (1,1,3,2) | 7 | 0.066 s |
| Marked cycle entry points | (2,3,3,2) | 10 | 0.025 s |

The largest intermediate algebra among these examples has 300 states.
In a separate local Python 3.14 run, computing both labeled and unlabeled
prefixes through degree 50 took respectively approximately 0.16, 0.24,
0.17, 0.16, and 0.69 seconds, excluding compilation. These measurements
describe these particular inputs and machine, not a general performance
guarantee for MSO compilation.

Tests independently decompose functions into cycles/components and check
the properties without evaluating logical formulas. Labeled counts and
unlabeled isomorphism orbits are exhaustively checked through n=4. Direct
formula truth is checked against the graph conditions through n=3.
Independent combinatorial recurrences verify all five labeled sequences
through n=10, including the two parity types for example 2, the two-component
decomposition for example 3, and cycle-length weights for examples 4 and 5.

At size zero, examples 1--3 have count zero; examples 4--5 have count one.
The existing MSO backend computes exact counts here; this gallery does not
add an MSO asymptotic-analysis path.

```bash
PYTHONPATH=src python examples/mso_global_gallery.py --max-n 8
```
