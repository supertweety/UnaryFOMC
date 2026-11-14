"""The C1 parser and the MSO parser group the Boolean connectives alike.

A sentence in the common input language of the two counters has one reading.
From the tightest to the loosest, the connectives are ``~``, ``&``, ``|``,
``->`` and ``<->``.  A chain of ``->`` groups to the right, so that
``A -> B -> C`` is ``A -> (B -> C)``.  Chains of ``&``, ``|`` and ``<->``
group to the left.
"""

from itertools import product
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unaryfomc import (
    C1FAtom, C1FBinary, C1FBooleanConstant, C1FEquality, C1FNot,
    C1FQuantifier, MSOAtom, MSOBinary, MSOBooleanConstant, MSOEquality,
    MSONot, MSOQuantifier, c1f_fomc, c1f_ufomc, compile_mso,
    parse_c1f_sentence, parse_mso_sentence,
)


# The binary connectives from the loosest to the tightest, each with the side
# to which a chain of it groups.
CONNECTIVES = (("<->", "left"), ("->", "right"), ("|", "left"), ("&", "left"))
# E is not among the names: in the MSO syntax it denotes the graph of f.
ATOMS = ("A", "B", "C", "D", "G")

# A chain, and the same sentence with the grouping written out.
WRITTEN_OUT = (
    ("A -> B -> C", "A -> (B -> C)"),
    ("A -> B -> C -> D", "A -> (B -> (C -> D))"),
    ("~A -> ~B -> C", "(~A) -> ((~B) -> C)"),
    ("A & B -> C | D -> G", "(A & B) -> ((C | D) -> G)"),
    ("A -> B -> C <-> D -> G", "(A -> (B -> C)) <-> (D -> G)"),
    ("(A -> B) -> C -> D", "(A -> B) -> (C -> D)"),
    ("forall x: P(x) -> Q(x) -> R(x)", "forall x: (P(x) -> (Q(x) -> R(x)))"),
    ("A -> forall x: P(x) -> B", "A -> (forall x: (P(x) -> B))"),
    ("(exists x: P(x)) -> (forall x: Q(x)) -> (exists x: f(x) = x)",
     "(exists x: P(x)) -> ((forall x: Q(x)) -> (exists x: f(x) = x))"),
)

# A chain with its two groupings: to the right, which is how it is read, and
# to the left.  The two groupings have different numbers of models.
FIRST_ORDER_CHAINS = (
    ("forall x: P(x) -> Q(x) -> R(x)",
     "forall x: (P(x) -> (Q(x) -> R(x)))",
     "forall x: ((P(x) -> Q(x)) -> R(x))"),
    ("(exists x: P(x)) -> (forall x: Q(x)) -> (exists x: f(x) = x)",
     "(exists x: P(x)) -> ((forall x: Q(x)) -> (exists x: f(x) = x))",
     "((exists x: P(x)) -> (forall x: Q(x))) -> (exists x: f(x) = x)"),
    ("forall x: (P(x) -> P(f(x)) -> f(f(x)) = x -> Q(x))",
     "forall x: (P(x) -> (P(f(x)) -> (f(f(x)) = x -> Q(x))))",
     "forall x: (((P(x) -> P(f(x))) -> f(f(x)) = x) -> Q(x))"),
)


def shape(node):
    """The syntax tree of a C1 or an MSO formula as nested tuples."""
    if isinstance(node, (C1FBooleanConstant, MSOBooleanConstant)):
        return node.value
    if isinstance(node, (C1FAtom, MSOAtom)):
        return (node.predicate,
                *((term.variable, term.exponent) for term in node.arguments))
    if isinstance(node, (C1FEquality, MSOEquality)):
        return ("=", (node.left.variable, node.left.exponent),
                (node.right.variable, node.right.exponent))
    if isinstance(node, (C1FNot, MSONot)):
        return ("~", shape(node.body))
    if isinstance(node, (C1FBinary, MSOBinary)):
        return (node.op, shape(node.left), shape(node.right))
    if isinstance(node, (C1FQuantifier, MSOQuantifier)):
        return (node.kind, node.variable, shape(node.body))
    raise AssertionError(type(node))


def grouped(atoms, operators, level=0):
    """The intended tree of ``atoms[0] operators[0] atoms[1] ...``."""
    if not operators:
        return (atoms[0],)
    connective, side = CONNECTIVES[level]
    parts, start = [], 0
    for position, operator in enumerate(operators + (connective,)):
        if operator == connective:
            parts.append(grouped(atoms[start:position + 1],
                                 operators[start:position], level + 1))
            start = position + 1
    if side == "right":
        parts.reverse()
    tree = parts[0]
    for part in parts[1:]:
        tree = (connective, part, tree) if side == "right" else (connective, tree, part)
    return tree


def satisfying_assignments(tree, atoms):
    """Number of truth assignments to ``atoms`` that satisfy ``tree``."""

    def holds(node, values):
        if len(node) == 1:
            return values[node[0]]
        left, right = holds(node[1], values), holds(node[2], values)
        return {"&": left and right, "|": left or right,
                "->": not left or right, "<->": left == right}[node[0]]

    return sum(
        holds(tree, dict(zip(atoms, bits)))
        for bits in product((False, True), repeat=len(atoms))
    )


def chains(longest):
    """All chains of at most ``longest`` binary connectives over ATOMS."""
    symbols = tuple(connective for connective, _ in CONNECTIVES)
    for length in range(1, longest + 1):
        for operators in product(symbols, repeat=length):
            atoms = ATOMS[:length + 1]
            text = atoms[0] + "".join(
                f" {operator} {atom}" for operator, atom in zip(operators, atoms[1:]))
            yield text, atoms, operators


def counts(sentence, largest):
    """Labeled and unlabeled counts of both counters for n = 0, ..., largest."""
    mso = compile_mso(sentence)
    sizes = range(largest + 1)
    return (
        [(int(c1f_fomc(sentence, n)), int(c1f_ufomc(sentence, n))) for n in sizes],
        [(int(mso.fomc(n)), int(mso.ufomc(n))) for n in sizes],
    )


class ConnectiveGroupingTests(unittest.TestCase):
    def test_implication_groups_to_the_right(self):
        right = ("->", ("A",), ("->", ("B",), ("C",)))
        for parse in (parse_c1f_sentence, parse_mso_sentence):
            with self.subTest(parser=parse.__name__):
                self.assertEqual(shape(parse("A -> B -> C")), right)
                self.assertEqual(parse("A -> B -> C"), parse("A -> (B -> C)"))
                self.assertNotEqual(parse("A -> B -> C"), parse("(A -> B) -> C"))

    def test_reported_counts(self):
        # A -> (B -> C) fails for one of the eight truth assignments, and
        # (A -> B) -> C fails for three of them.
        self.assertEqual(int(c1f_fomc("A -> B -> C", 0)), 7)
        self.assertEqual(int(compile_mso("A -> B -> C").fomc(0)), 7)
        self.assertEqual(int(c1f_fomc("(A -> B) -> C", 0)), 5)
        self.assertEqual(int(compile_mso("(A -> B) -> C").fomc(0)), 5)

    def test_parsers_agree_on_every_chain(self):
        for text, atoms, operators in chains(4):
            with self.subTest(sentence=text):
                expected = grouped(atoms, operators)
                self.assertEqual(shape(parse_c1f_sentence(text)), expected)
                self.assertEqual(shape(parse_mso_sentence(text)), expected)

    def test_parsers_agree_on_written_out_groupings(self):
        for chain, written_out in WRITTEN_OUT:
            with self.subTest(sentence=chain):
                c1 = parse_c1f_sentence(chain)
                mso = parse_mso_sentence(chain)
                self.assertEqual(c1, parse_c1f_sentence(written_out))
                self.assertEqual(mso, parse_mso_sentence(written_out))
                self.assertEqual(shape(c1), shape(mso))

    def test_counters_agree_on_propositional_chains(self):
        # The structures on n elements are the functions on them together
        # with a truth assignment: 1, 1, 4 labeled and 1, 1, 3 unlabeled
        # functions for n = 0, 1, 2.
        for text, atoms, operators in chains(3):
            satisfying = satisfying_assignments(grouped(atoms, operators), atoms)
            expected = [(satisfying, satisfying), (satisfying, satisfying),
                        (4 * satisfying, 3 * satisfying)]
            with self.subTest(sentence=text):
                c1, mso = counts(text, 2)
                self.assertEqual(c1, expected)
                self.assertEqual(mso, expected)

    def test_counters_agree_on_first_order_chains(self):
        for chain, to_the_right, to_the_left in FIRST_ORDER_CHAINS:
            with self.subTest(sentence=chain):
                c1, mso = counts(chain, 4)
                self.assertEqual(c1, mso)
                self.assertEqual((c1, mso), counts(to_the_right, 4))
                self.assertNotEqual(c1, counts(to_the_left, 4)[0])

    def test_counts_of_an_implication_chain_under_a_quantifier(self):
        # Every element avoids one of the eight colours given by P, Q and R,
        # so the models are the functions with a colouring by seven colours.
        # Their numbers up to isomorphism come from Burnside's lemma.  With
        # the grouping to the left there would be five colours.
        sentence = "forall x: P(x) -> Q(x) -> R(x)"
        c1, mso = counts(sentence, 4)
        self.assertEqual([labeled for labeled, _ in c1],
                         [7**n * n**n for n in range(5)])
        self.assertEqual([unlabeled for _, unlabeled in c1],
                         [1, 7, 105, 1624, 27153])
        self.assertEqual(mso, c1)


if __name__ == "__main__":
    unittest.main()
