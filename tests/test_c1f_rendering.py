"""A parsed C1 sentence must render to text that parses back to the same tree.

``render_c1f_formula`` writes a quantifier without parentheses of its own,
and a quantifier extends as far to the right as possible.  A quantified left
operand of a binary connective must therefore be closed by parentheses.
Without them ``(exists x: P(x)) -> A`` is printed as text that the parser
reads as ``exists x: (P(x) -> A)``, which is a different sentence.
"""

from itertools import product
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unaryfomc import (
    C1FAtom, C1FBinary, C1FBooleanConstant, C1FEquality, C1FNot,
    C1FPredecessorCount, C1FQuantifier, C1FTerm, c1f_fomc, parse_c1f_sentence,
    render_c1f_formula,
)


CONNECTIVES = ("&", "|", "->", "<->")
X = C1FTerm("x")
Y = C1FTerm("y")
FX = C1FTerm("x", 1)
P_X = C1FAtom("P", (X,))

# Closed formulas of every syntactic kind that can be an operand.
CLOSED_OPERANDS = {
    "nullary atom": C1FAtom("A", ()),
    "constant": C1FBooleanConstant(True),
    "negated atom": C1FNot(C1FAtom("B", ())),
    "forall": C1FQuantifier("forall", "x", P_X),
    "exists": C1FQuantifier("exists", "x", P_X),
    "counting": C1FQuantifier("exists", "x", P_X, ">=", 2),
    "negated quantifier": C1FNot(C1FQuantifier("exists", "x", P_X)),
    "quantified connective": C1FQuantifier(
        "forall", "x", C1FBinary("->", P_X, C1FAtom("Q", (FX,)))
    ),
    "connective": C1FBinary("|", C1FAtom("A", ()), C1FAtom("B", ())),
}

# Formulas with the free variable x, for the body of a quantifier.
OPEN_OPERANDS = {
    "atom": P_X,
    "deep atom": C1FAtom("Q", (FX,)),
    "equality": C1FEquality(FX, X),
    "inequality": C1FNot(C1FEquality(C1FTerm("x", 2), X)),
    "predecessor count": C1FPredecessorCount(
        "x", "y", C1FAtom("C", (Y,)), "==", 2
    ),
    "predecessor bound": C1FPredecessorCount(
        "x", "y", C1FBooleanConstant(True), "<=", 1
    ),
    "negated predecessor count": C1FNot(
        C1FPredecessorCount("x", "y", C1FAtom("C", (Y,)), ">=", 1)
    ),
    "connective": C1FBinary("&", P_X, C1FAtom("Q", (X,))),
}

QUANTIFIED = ("forall", "exists", "counting", "quantified connective")


class RenderRoundTripTests(unittest.TestCase):
    def assert_round_trip(self, tree):
        text = render_c1f_formula(tree)
        self.assertEqual(parse_c1f_sentence(text), tree, text)

    def test_quantified_left_operand_is_parenthesized(self):
        tree = parse_c1f_sentence("(exists x: P(x)) -> A")
        self.assertEqual(render_c1f_formula(tree), "((exists x: P(x)) -> A)")
        self.assert_round_trip(tree)
        # The two readings are different sentences.  They agree on one
        # element, where both fail only if P holds and A does not.  They
        # differ on the empty domain and on two elements.
        self.assertEqual([c1f_fomc(tree, n) for n in range(3)], [2, 3, 20])
        self.assertEqual(
            [c1f_fomc(render_c1f_formula(tree), n) for n in range(3)],
            [2, 3, 20],
        )
        self.assertEqual(
            [c1f_fomc("exists x: (P(x) -> A)", n) for n in range(3)],
            [0, 3, 28],
        )

    def test_quantified_right_operand_is_left_alone(self):
        tree = parse_c1f_sentence("A -> (exists x: P(x))")
        self.assertEqual(render_c1f_formula(tree), "(A -> exists x: P(x))")
        self.assert_round_trip(tree)

    def test_predecessor_count_as_left_operand(self):
        text = "forall x: ((exists=2 y: (f(y) = x & C(y))) & P(x))"
        tree = parse_c1f_sentence(text)
        self.assertEqual(render_c1f_formula(tree), text)
        self.assert_round_trip(tree)

    def test_predecessor_count_with_a_closed_conjunct(self):
        # The previous renderer wrote this tree as ``old_text``.  The parser
        # accepts that text but reads A as part of the filler of the count,
        # which is a different sentence.
        text = "forall x: ((exists<=1 y: (f(y) = x & C(y))) & A)"
        old_text = "forall x: (exists<=1 y: (f(y) = x & C(y)) & A)"
        tree = parse_c1f_sentence(text)
        self.assertEqual(render_c1f_formula(tree), text)
        self.assert_round_trip(tree)
        self.assertEqual(
            [c1f_fomc(tree, n) for n in range(4)], [2, 2, 14, 168]
        )
        self.assertEqual(
            [c1f_fomc(old_text, n) for n in range(4)], [2, 4, 30, 384]
        )

    def test_every_pair_of_closed_operands(self):
        for op, (left_name, left), (right_name, right) in product(
            CONNECTIVES, CLOSED_OPERANDS.items(), CLOSED_OPERANDS.items()
        ):
            with self.subTest(op=op, left=left_name, right=right_name):
                self.assert_round_trip(C1FBinary(op, left, right))

    def test_every_pair_of_open_operands_under_a_quantifier(self):
        for op, (left_name, left), (right_name, right) in product(
            CONNECTIVES, OPEN_OPERANDS.items(), OPEN_OPERANDS.items()
        ):
            with self.subTest(op=op, left=left_name, right=right_name):
                self.assert_round_trip(
                    C1FQuantifier("forall", "x", C1FBinary(op, left, right))
                )

    def test_nested_connectives(self):
        names = ("nullary atom", "exists", "counting", "negated quantifier")
        operands = [CLOSED_OPERANDS[name] for name in names]
        for inner, outer in product(CONNECTIVES, repeat=2):
            for left, right, other in product(operands, repeat=3):
                nested = C1FBinary(inner, left, right)
                with self.subTest(inner=inner, outer=outer):
                    self.assert_round_trip(C1FBinary(outer, nested, other))
                    self.assert_round_trip(C1FBinary(outer, other, nested))

    def test_rendered_text_has_the_counts_of_the_tree(self):
        for op, left_name, right_name in product(
            CONNECTIVES, QUANTIFIED, ("nullary atom", "exists")
        ):
            tree = C1FBinary(
                op, CLOSED_OPERANDS[left_name], CLOSED_OPERANDS[right_name]
            )
            text = render_c1f_formula(tree)
            with self.subTest(text=text):
                for domain_size in range(3):
                    self.assertEqual(
                        c1f_fomc(text, domain_size),
                        c1f_fomc(tree, domain_size),
                    )


if __name__ == "__main__":
    unittest.main()
