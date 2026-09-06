"""Reachable finite algebras, products, projection, and context minimization."""

from collections import deque
from dataclasses import dataclass

from .mso_specification import MSOSpecification


class MSOCompilationLimitError(RuntimeError):
    """Compilation stopped without returning a partial or approximate answer."""


@dataclass(frozen=True)
class MSOCompilerLimits:
    max_states: int | None = 5000
    max_transitions: int | None = 2_000_000
    max_colors: int | None = 1024
    max_formula_nodes: int | None = 5000

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be positive or None")


class Budget:
    def __init__(self, limits):
        self.limits = limits
        self.transitions = 0
        self.peak_states = 0
        self.algebras = 0

    def check(self, name, value):
        limit = getattr(self.limits, name)
        if limit is not None and value > limit:
            raise MSOCompilationLimitError(
                f"MSO compilation exceeded {name}={limit} (needed {value}); "
                "no partial count was produced. Increase the explicit limit or simplify the sentence.")

    def work(self, amount=1):
        self.transitions += amount
        self.check("max_transitions", self.transitions)


def build_algebra(predicates, identities, forest_add, root, forget, embed,
                  product, close, graph_add, accepts, budget):
    """Saturate all reachable states independently of the requested domain size."""
    predicates = tuple(predicates)
    colors = 1 << len(predicates)
    budget.check("max_colors", colors)
    states, indices = [[] for _ in range(4)], [{} for _ in range(4)]
    pending = deque()
    cache = {}
    callbacks = (forest_add, root, forget, embed, product, close, graph_add)
    outputs = (0, 1, 0, 2, 2, 3, 3)

    def intern(role, value):
        if value not in indices[role]:
            size = sum(map(len, states)) + 1
            budget.check("max_states", size)
            budget.peak_states = max(budget.peak_states, size)
            indices[role][value] = len(states[role])
            states[role].append(value)
            pending.append((role, value))
        return indices[role][value]

    def call(op, *args):
        key = op, args
        if key not in cache:
            budget.work()
            cache[key] = intern(outputs[op], callbacks[op](*args))
        return cache[key]

    fid, gid = intern(0, identities[0]), intern(3, identities[1])
    while pending:
        role, value = pending.popleft()
        if role == 0:
            for other in tuple(states[0]):
                call(0, value, other)
                call(0, other, value)
            for color in range(colors):
                call(1, color, value)
        elif role == 1:
            call(2, value)
            call(3, value)
        elif role == 2:
            call(5, value)
            for other in tuple(states[2]):
                call(4, value, other)
                call(4, other, value)
        else:
            for other in tuple(states[3]):
                call(6, value, other)
                call(6, other, value)
    f, h, d, g = states
    result = MSOSpecification(
        predicates,
        tuple(tuple(call(0, a, b) for b in f) for a in f),
        tuple(tuple(call(1, c, a) for a in f) for c in range(colors)),
        tuple(call(2, a) for a in h), tuple(call(3, a) for a in h),
        tuple(tuple(call(4, a, b) for b in d) for a in d), tuple(call(5, a) for a in d),
        tuple(tuple(call(6, a, b) for b in g) for a in g),
        frozenset(i for i, value in enumerate(g) if accepts(value)), fid, gid)
    budget.algebras += 1
    return minimize(result)


def _classes(signatures):
    ids = {}
    result = []
    for signature in signatures:
        result.append(ids.setdefault(signature, len(ids)))
    return result


def minimize(specification):
    """Partition refinement over every well-typed one-hole constructor context."""
    s = specification
    f, h, d, g = s.state_counts.values()
    blocks = ([0] * f, [0] * h, [0] * d, _classes(i in s.accepting for i in range(g)))
    while True:
        bf, bh, bd, bg = blocks
        refined = (
            _classes((bf[a], tuple(bf[b] for b in s.forest_add[a]),
                      tuple(bh[row[a]] for row in s.root)) for a in range(f)),
            _classes((bh[a], bf[s.forget[a]], bd[s.tree_to_dotted[a]]) for a in range(h)),
            _classes((bd[a], tuple(bd[b] for b in s.dotted_product[a]),
                      tuple(bd[row[a]] for row in s.dotted_product), bg[s.close[a]]) for a in range(d)),
            _classes((bg[a], tuple(bg[b] for b in s.graph_add[a])) for a in range(g)),
        )
        if blocks == refined:
            break
        blocks = refined
    representatives = []
    for partition in blocks:
        representatives.append([partition.index(i) for i in range(max(partition) + 1)])
    rf, rh, rd, rg = representatives
    bf, bh, bd, bg = blocks
    return MSOSpecification(
        s.predicates, tuple(tuple(bf[s.forest_add[a][b]] for b in rf) for a in rf),
        tuple(tuple(bh[row[a]] for a in rf) for row in s.root),
        tuple(bf[s.forget[a]] for a in rh), tuple(bd[s.tree_to_dotted[a]] for a in rh),
        tuple(tuple(bd[s.dotted_product[a][b]] for b in rd) for a in rd),
        tuple(bg[s.close[a]] for a in rd),
        tuple(tuple(bg[s.graph_add[a][b]] for b in rg) for a in rg),
        frozenset(bg[a] for a in s.accepting), bf[s.forest_identity], bg[s.graph_identity])


def complement(s):
    return minimize(MSOSpecification(s.predicates, s.forest_add, s.root, s.forget,
                                    s.tree_to_dotted, s.dotted_product, s.close, s.graph_add,
                                    frozenset(range(len(s.graph_add))) - s.accepting,
                                    s.forest_identity, s.graph_identity))


def combine(left, right, op, budget):
    vocabulary = tuple(sorted(set(left.predicates) | set(right.predicates)))
    positions = {name: i for i, name in enumerate(vocabulary)}

    def color_in(color, names):
        return sum(((color >> positions[name]) & 1) << i for i, name in enumerate(names))

    def accept(state):
        a, b = state[0] in left.accepting, state[1] in right.accepting
        return {"&": a and b, "|": a or b, "->": not a or b, "<->": a == b}[op]

    return build_algebra(
        vocabulary, ((left.forest_identity, right.forest_identity), (left.graph_identity, right.graph_identity)),
        lambda a, b: (left.forest_add[a[0]][b[0]], right.forest_add[a[1]][b[1]]),
        lambda c, a: (left.root[color_in(c, left.predicates)][a[0]], right.root[color_in(c, right.predicates)][a[1]]),
        lambda a: (left.forget[a[0]], right.forget[a[1]]),
        lambda a: (left.tree_to_dotted[a[0]], right.tree_to_dotted[a[1]]),
        lambda a, b: (left.dotted_product[a[0]][b[0]], right.dotted_product[a[1]][b[1]]),
        lambda a: (left.close[a[0]], right.close[a[1]]),
        lambda a, b: (left.graph_add[a[0]][b[0]], right.graph_add[a[1]][b[1]]), accept, budget)


def project(s, predicate, budget):
    """Existentially forget one vertex-color bit, determinizing every sort.

    Elements of a state set are possible witnesses, not multiplicities.
    Independent substructures may choose different witnesses, including
    different copies of the same uncolored isomorphism type.
    """
    if predicate not in s.predicates:
        return s
    index = s.predicates.index(predicate)
    vocabulary = tuple(name for name in s.predicates if name != predicate)

    def lift(operation, a, b):
        budget.work(len(a) * len(b))
        return frozenset(operation[x][y] for x in a for y in b)

    def root(color, forest):
        low = color & ((1 << index) - 1)
        base = low | ((color >> index) << (index + 1))
        return frozenset(s.root[base | (bit << index)][f] for f in forest for bit in (0, 1))

    return build_algebra(
        vocabulary, (frozenset((s.forest_identity,)), frozenset((s.graph_identity,))),
        lambda a, b: lift(s.forest_add, a, b), root,
        lambda a: frozenset(s.forget[x] for x in a),
        lambda a: frozenset(s.tree_to_dotted[x] for x in a),
        lambda a, b: lift(s.dotted_product, a, b),
        lambda a: frozenset(s.close[x] for x in a),
        lambda a, b: lift(s.graph_add, a, b), lambda a: bool(a & s.accepting), budget)


def vertex_condition(vocabulary, event, budget, *, singleton=False):
    cap = 2 if singleton else 1

    def plus(a, b):
        return min(cap, a + b)

    return build_algebra(vocabulary, (0, 0), plus,
                         lambda color, forest: plus(forest, int(event(color))),
                         lambda a: a, lambda a: a, plus, lambda a: a, plus,
                         lambda a: a == 1, budget)


def edge_condition(vocabulary, source_index, target_index, budget):
    # Forest: (found edge, marked orphan). Dotted: add target-mark at dot.
    return build_algebra(
        vocabulary, ((False, False), False),
        lambda a, b: (a[0] or b[0], a[1] or b[1]),
        lambda color, a: (a[0] or (a[1] and bool(color >> target_index & 1)),
                          bool(color >> source_index & 1), bool(color >> target_index & 1)),
        lambda a: a[:2], lambda a: a,
        lambda a, b: (a[0] or b[0] or (a[1] and b[2]), b[1], a[2]),
        lambda a: a[0] or (a[1] and a[2]),
        lambda a, b: a or b, bool, budget)


def constant(vocabulary, value, budget):
    return build_algebra(vocabulary, (0, 0), lambda a, b: 0, lambda c, a: 0,
                         lambda a: 0, lambda a: 0, lambda a, b: 0,
                         lambda a: 0, lambda a, b: 0, lambda a: value, budget)
