"""Finite composition algebras and exact evaluation for MSO function graphs.

The four sorts and their operations are those of Part II, Section 4.1.
``evaluate_mso_specification`` evaluates the finite specification of
Part II, Section 4.2, equations (12)--(17).  These equations count
isomorphism classes.  Labeled counts use the same equations without the
repetition terms (only k = 1), read as exponential generating functions.
States need only form a sound sentence-recognizing congruence; complete
rank-r theories are one possible choice, not required by the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial


@dataclass(frozen=True)
class MSOSpecification:
    predicates: tuple[str, ...]
    forest_add: tuple[tuple[int, ...], ...]
    root: tuple[tuple[int, ...], ...]
    forget: tuple[int, ...]
    tree_to_dotted: tuple[int, ...]
    dotted_product: tuple[tuple[int, ...], ...]
    close: tuple[int, ...]
    graph_add: tuple[tuple[int, ...], ...]
    accepting: frozenset[int]
    forest_identity: int = 0
    graph_identity: int = 0

    @property
    def state_counts(self):
        return dict(forests=len(self.forest_add), trees=len(self.forget),
                    dotted=len(self.dotted_product), graphs=len(self.graph_add))

    def validate(self, *, check_laws=False):
        f, h, d, g = self.state_counts.values()
        if min(f, h, d, g) < 1 or len(set(self.predicates)) != len(self.predicates):
            raise ValueError("State spaces must be nonempty and predicate names distinct")

        def vector(values, size, target):
            if len(values) != size or any(type(v) is not int or not 0 <= v < target for v in values):
                raise ValueError("Invalid MSO state map")

        def table(values, rows, columns, target):
            if len(values) != rows:
                raise ValueError("Invalid MSO table dimensions")
            for row in values:
                vector(row, columns, target)

        table(self.forest_add, f, f, f)
        table(self.root, 1 << len(self.predicates), f, h)
        vector(self.forget, h, f)
        vector(self.tree_to_dotted, h, d)
        table(self.dotted_product, d, d, d)
        vector(self.close, d, g)
        table(self.graph_add, g, g, g)
        vector(tuple(self.accepting), len(self.accepting), g)
        vector((self.forest_identity,), 1, f)
        vector((self.graph_identity,), 1, g)
        if check_laws:
            for operation, identity, commutative in (
                (self.forest_add, self.forest_identity, True),
                (self.graph_add, self.graph_identity, True),
                (self.dotted_product, None, False),
            ):
                for a in range(len(operation)):
                    if identity is not None and (operation[a][identity] != a or operation[identity][a] != a):
                        raise ValueError("Invalid monoid identity")
                    for b in range(len(operation)):
                        if commutative and operation[a][b] != operation[b][a]:
                            raise ValueError("Noncommutative set/multiset state operation")
                        for c in range(len(operation)):
                            if operation[operation[a][b]][c] != operation[a][operation[b][c]]:
                                raise ValueError("Nonassociative MSO composition")
            for a in range(d):
                for b in range(d):
                    if self.close[self.dotted_product[a][b]] != self.close[self.dotted_product[b][a]]:
                        raise ValueError("Closed cycle states must be rotation invariant")
        return self

    def to_dict(self):
        return dict(format="mso-function-algebra-v1", predicates=list(self.predicates),
                    forest_add=[list(r) for r in self.forest_add], root=[list(r) for r in self.root],
                    forget=list(self.forget), tree_to_dotted=list(self.tree_to_dotted),
                    dotted_product=[list(r) for r in self.dotted_product], close=list(self.close),
                    graph_add=[list(r) for r in self.graph_add], accepting=sorted(self.accepting),
                    forest_identity=self.forest_identity, graph_identity=self.graph_identity)

    @classmethod
    def from_dict(cls, value, *, check_laws=True):
        if value.get("format") != "mso-function-algebra-v1":
            raise ValueError("Unknown MSO specification format")
        result = cls(
            tuple(value["predicates"]), tuple(map(tuple, value["forest_add"])),
            tuple(map(tuple, value["root"])), tuple(value["forget"]), tuple(value["tree_to_dotted"]),
            tuple(map(tuple, value["dotted_product"])), tuple(value["close"]),
            tuple(map(tuple, value["graph_add"])), frozenset(value["accepting"]),
            value["forest_identity"], value["graph_identity"])
        return result.validate(check_laws=check_laws)

    def has_model(self, *, nonempty=True):
        """Finite-table reachability, including nonempty realizations of identity."""
        forests, trees, dotted = {self.forest_identity}, set(), set()
        changed = True
        while changed:
            previous = len(forests), len(trees), len(dotted)
            forests |= {self.forest_add[a][b] for a in tuple(forests) for b in tuple(forests)}
            trees |= {row[s] for row in self.root for s in forests}
            forests |= {self.forget[t] for t in trees}
            dotted |= {self.tree_to_dotted[t] for t in trees}
            dotted |= {self.dotted_product[a][b] for a in tuple(dotted) for b in tuple(dotted)}
            changed = previous != (len(forests), len(trees), len(dotted))
        graphs = {self.close[d] for d in dotted}
        while True:
            enlarged = graphs | {self.graph_add[a][b] for a in graphs for b in graphs}
            if enlarged == graphs:
                break
            graphs = enlarged
        if not nonempty:
            graphs.add(self.graph_identity)
        return bool(graphs & self.accepting)


def _add(target, source, scale=1):
    for state, coefficient in source.items():
        if coefficient:
            target[state] = target.get(state, 0) + scale * coefficient


def _multiply(left, right, operation):
    result = {}
    for a, ca in left.items():
        for b, cb in right.items():
            state = operation[a][b]
            result[state] = result.get(state, 0) + ca * cb
    return result


def _repeat(state, number, operation):
    # Semigroups need no artificial identity for positive powers.
    if number < 1:
        raise ValueError("State repetition must be positive")
    result = None
    while number:
        if number & 1:
            result = state if result is None else operation[result][state]
        state = operation[state][state]
        number >>= 1
    return result


def _exponential_coefficient(logarithm, exponential, degree, operation):
    result = {}
    for i in range(1, degree + 1):
        _add(result, _multiply(logarithm[i], exponential[degree - i], operation), Fraction(i, degree))
    return result


def _series_product(left, right, operation, bound):
    result = [{} for _ in range(bound + 1)]
    for i, a in enumerate(left):
        if a:
            for j in range(bound + 1 - i):
                if right[j]:
                    _add(result[i + j], _multiply(a, right[j], operation))
    return result


def _totients(bound):
    result = list(range(bound + 1))
    for p in range(2, bound + 1):
        if result[p] == p:
            for multiple in range(p, bound + 1, p):
                result[multiple] -= result[multiple] // p
    return result


def evaluate_mso_specification(specification: MSOSpecification, bound: int, *, labeled=True):
    """Exact counts for sizes 0,...,bound; no floating-point arithmetic.

    All distributions retain their algebra states until final acceptance.
    Repetition changes both degree and state, at children, cycles, and SETs.
    """
    if type(bound) is not int or bound < 0:
        raise ValueError("The domain bound must be a nonnegative integer")
    if type(labeled) is not bool:
        raise ValueError("labeled must be a bool")
    s = specification.validate()
    trees = [{} for _ in range(bound + 1)]
    child_log = [{} for _ in range(bound + 1)]
    forests = [{s.forest_identity: Fraction(1)}] + [{} for _ in range(bound)]
    for degree in range(1, bound + 1):
        for color_map in s.root:
            for state, coefficient in forests[degree - 1].items():
                h = color_map[state]
                trees[degree][h] = trees[degree].get(h, 0) + coefficient
        for h, coefficient in trees[degree].items():
            state = s.forget[h]
            for k in range(1, (1 if labeled else bound // degree) + 1):
                repeated = _repeat(state, k, s.forest_add)
                child_log[k * degree][repeated] = child_log[k * degree].get(repeated, 0) + coefficient / k
        forests[degree] = _exponential_coefficient(child_log, forests, degree, s.forest_add)

    decorations = [{} for _ in range(bound + 1)]
    for degree in range(1, bound + 1):
        for h, coefficient in trees[degree].items():
            d = s.tree_to_dotted[h]
            decorations[degree][d] = decorations[degree].get(d, 0) + coefficient
    components = [{} for _ in range(bound + 1)]
    words = decorations
    totient = _totients(bound)
    for length in range(1, bound + 1):
        # A rotation of order k repeats a word of length `length` k times.
        for k in range(1, (1 if labeled else bound // length) + 1):
            scale = Fraction(totient[k], length * k)
            for degree in range(length, bound // k + 1):
                for d, coefficient in words[degree].items():
                    g = s.close[_repeat(d, k, s.dotted_product)]
                    target = components[k * degree]
                    target[g] = target.get(g, 0) + scale * coefficient
        if length < bound:
            words = _series_product(words, decorations, s.dotted_product, bound)

    graph_log = [{} for _ in range(bound + 1)]
    for degree in range(1, bound + 1):
        for g, coefficient in components[degree].items():
            for k in range(1, (1 if labeled else bound // degree) + 1):
                repeated = _repeat(g, k, s.graph_add)
                target = graph_log[k * degree]
                target[repeated] = target.get(repeated, 0) + coefficient / k
    graphs = [{s.graph_identity: Fraction(1)}] + [{} for _ in range(bound)]
    for degree in range(1, bound + 1):
        graphs[degree] = _exponential_coefficient(graph_log, graphs, degree, s.graph_add)
    result = []
    for degree, distribution in enumerate(graphs):
        coefficient = sum((v for g, v in distribution.items() if g in s.accepting), Fraction())
        value = coefficient * (factorial(degree) if labeled else 1)
        if value.denominator != 1 or value < 0:
            raise ArithmeticError("MSO specification produced a nonintegral or negative count")
        result.append(int(value))
    return tuple(result)
