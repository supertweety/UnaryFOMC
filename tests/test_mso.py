from __future__ import annotations

from itertools import permutations, product
from math import factorial
from pathlib import Path
import json
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unaryfomc.mso import compile_mso, mso_fomc, mso_ufomc, MSOCompilerLimits, MSOCompilationLimitError
from unaryfomc.mso_specification import MSOSpecification, evaluate_mso_specification
from unaryfomc.mso_syntax import (
    MSOAtom, MSOBooleanConstant, MSOBinary, MSOEquality, MSONot, MSOQuantifier,
    parse_mso_sentence, render_mso_formula, mso_predicate_arities, evaluate_mso_formula,
)


EVEN_CYCLES = "exists X: forall x: (X(f(x)) <-> ~X(x))"
CONNECTED = ("forall X: (((forall x: (X(x) <-> X(f(x)))) & "
             "(exists x: X(x))) -> (forall x: X(x)))")
ONE_CYCLE = ("forall x: forall y: forall X: "
             "((X(x) & (forall z: (X(z) -> X(f(z))))) -> X(y))")
ROOTED_TREES = ("exists r: (f(r)=r & (forall x: forall X: "
                "((X(x) & (forall y: (X(y) -> X(f(y))))) -> X(r))))")


def direct_truth(node, function, predicates, nodes=None, sets=None):
    """Independent truth definition; deliberately not the library checker."""
    nodes, sets = dict(nodes or {}), dict(sets or {})

    def value(t):
        result = nodes[t.variable]
        for _ in range(t.exponent):
            result = function[result]
        return result

    if isinstance(node, MSOBooleanConstant):
        return node.value
    if isinstance(node, MSOEquality):
        return value(node.left) == value(node.right)
    if isinstance(node, MSOAtom):
        if node.predicate in sets:
            return bool(sets[node.predicate] & (1 << value(node.arguments[0])))
        if node.predicate == "E":
            return function[value(node.arguments[0])] == value(node.arguments[1])
        return (bool(predicates[node.predicate]) if not node.arguments
                else bool(predicates[node.predicate] & (1 << value(node.arguments[0]))))
    if isinstance(node, MSONot):
        return not direct_truth(node.body, function, predicates, nodes, sets)
    if isinstance(node, MSOBinary):
        a = direct_truth(node.left, function, predicates, nodes, sets)
        b = direct_truth(node.right, function, predicates, nodes, sets)
        return {"&": a and b, "|": a or b, "->": not a or b, "<->": a == b}[node.op]
    if isinstance(node, MSOQuantifier):
        answers = []
        scope = nodes if node.sort == "node" else sets
        for v in range(len(function) if node.sort == "node" else 1 << len(function)):
            scope[node.variable] = v
            answers.append(direct_truth(node.body, function, predicates, nodes, sets))
        return all(answers) if node.kind == "forall" else any(answers)
    raise AssertionError(type(node))


def brute_counts(formula, n):
    node = parse_mso_sentence(formula)
    arities = mso_predicate_arities(node)
    names = tuple(sorted(arities))
    labelings = tuple(permutations(range(n)))
    count, orbits = 0, set()
    for function in product(range(n), repeat=n):
        for values in product(*(range(1 << n) if arities[p] else range(2) for p in names)):
            interpretation = dict(zip(names, values))
            if not direct_truth(node, function, interpretation):
                continue
            count += 1
            candidates = []
            for perm in labelings:
                renamed = [0] * n
                for v, target in enumerate(function):
                    renamed[perm[v]] = perm[target]
                colors = tuple(sum(((interpretation[p] >> v) & 1) << perm[v] for v in range(n))
                               if arities[p] else interpretation[p] for p in names)
                candidates.append((tuple(renamed), colors))
            orbits.add(min(candidates))
    return count, len(orbits)


class MSOSyntaxTests(unittest.TestCase):
    def test_round_trip(self):
        for text in (EVEN_CYCLES, CONNECTED, ONE_CYCLE,
                     "forall x,y: (f^2(x)=y -> P(f(y)))",
                     "exists set s: forall node X: X in s",
                     "Q() & (exists X: forall x: (X(x) | P(x)))",
                     "exists X: ((forall x: X(x)) | (exists x: ~X(x)))"):
            with self.subTest(text=text):
                parsed = parse_mso_sentence(text)
                self.assertEqual(parsed, parse_mso_sentence(render_mso_formula(parsed)))

    def test_rejects_outside_signature(self):
        for text in ("P(x)", "forall x,y: R(x,y)", "forall x: g(x)=x",
                     "forall x: E(x)", "exists X: X()", "exists=2 x: P(x)",
                     "forall f: TRUE", "forall E: TRUE", "forall x: (P(x) & P())"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_mso_sentence(text)

    def test_shadowing_and_explicit_sorts(self):
        for text in ("exists X: ((exists X: forall x: X(x)) & (forall y: ~X(y)))",
                     "forall x: (P(x) | (exists x: ~P(x)))",
                     "exists set s: forall node X: X in s"):
            c = compile_mso(text)
            for n in range(4):
                self.assertEqual((c.fomc(n), c.ufomc(n)), brute_counts(text, n))

    def test_direct_model_checker(self):
        node = parse_mso_sentence(EVEN_CYCLES)
        self.assertTrue(evaluate_mso_formula(node, (1, 0, 0)))
        self.assertFalse(evaluate_mso_formula(node, (1, 2, 0)))


class MSOCountingTests(unittest.TestCase):
    def test_public_api(self):
        from unaryfomc import compile_mso_fomc, compile_mso_ufomc, evaluate_mso_fomc, evaluate_mso_ufomc

        self.assertEqual(evaluate_mso_fomc(compile_mso_fomc(EVEN_CYCLES), 5), 680)
        self.assertEqual(evaluate_mso_ufomc(compile_mso_ufomc(EVEN_CYCLES), 5), 8)

    def test_classical_sequences(self):
        c = compile_mso("TRUE")
        self.assertEqual(c.sequence(8), tuple(n ** n for n in range(9)))
        self.assertEqual(c.sequence(8, labeled=False), (1, 1, 3, 7, 19, 47, 130, 343, 951))
        c = compile_mso("forall y: exists x: f(x)=y")
        self.assertEqual(c.sequence(12), tuple(factorial(n) for n in range(13)))
        self.assertEqual(c.sequence(12, labeled=False), (1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77))
        c = compile_mso("forall x: f(x)=x")
        self.assertEqual(c.sequence(12), (1,) * 13)
        self.assertEqual(c.sequence(12, labeled=False), (1,) * 13)

    def test_full_mso_against_exhaustive_orbits(self):
        formulas = (
            EVEN_CYCLES, CONNECTED, ONE_CYCLE,
            "exists X: exists x: X(x)",
            "forall X: exists x: X(x)",
            "exists X: forall x: (P(x) <-> X(x))",
            "forall X: exists Y: forall x: (X(x) <-> ~Y(x))",
            "exists x: forall y: f(y)=x",
            "forall x: exists y: (x!=y & f(y)=f(x))",
            "forall x: f(f(x))=x",
            "exists x: (f^2(x)=x & f(x)!=x)",
            "forall x: (P(x) -> P(f(x)))",
            "exists x: exists y: (P(x) & ~P(y) & f(x)=f(y))",
            "exists X: ((exists x: X(x)) & (forall y: (X(y) -> (P(y) & X(f(y))))))",
            "forall x: exists X: (X(x) & (forall y: (X(y) <-> y=x)))",
            "Q() | (exists x: P(x))",
        )
        for text in formulas:
            c = compile_mso(text)
            for n in range(4):
                with self.subTest(formula=text, n=n):
                    self.assertEqual((c.fomc(n), c.ufomc(n)), brute_counts(text, n))

    def test_quantified_sets_are_not_counted_as_witnesses(self):
        c = compile_mso("exists X: TRUE")
        self.assertEqual(c.sequence(7), tuple(n ** n for n in range(8)))
        # A two-cycle has two alternating sets, neither rotation-invariant.
        c = compile_mso(EVEN_CYCLES)
        self.assertEqual((c.fomc(2), c.ufomc(2)), (1, 1))
        self.assertEqual((c.fomc(4), c.ufomc(4)), brute_counts(EVEN_CYCLES, 4))

    def test_component_multiplicity_and_cancellation(self):
        text = ("(forall x: f(f(x))=x) & (exists x,y: (x!=y & f(x)=x & f(y)=y & "
                "(forall z: (f(z)=z -> (z=x | z=y)))))")
        c = compile_mso(text)
        expected = tuple(factorial(n) // (2 * 2 ** ((n - 2) // 2) * factorial((n - 2) // 2))
                         if n >= 2 and n % 2 == 0 else 0 for n in range(13))
        self.assertEqual(c.sequence(12), expected)
        self.assertEqual(c.sequence(12, labeled=False), tuple(int(n >= 2 and n % 2 == 0) for n in range(13)))

    def test_child_multiset_repetition(self):
        # Every nonroot is a leaf of the sole root: colored rooted stars.
        text = ("exists r: (f(r)=r & (forall x: f(x)=r) & "
                "(exists x,y: (x!=y & x!=r & y!=r & P(x) & P(y) & "
                "(forall z: ((z!=r & P(z)) -> (z=x | z=y))))))")
        c = compile_mso(text)
        self.assertEqual(c.sequence(8), tuple(n * (n - 1) * (n - 2) if n >= 3 else 0 for n in range(9)))
        self.assertEqual(c.sequence(8, labeled=False), (0, 0, 0, 2, 2, 2, 2, 2, 2))

    def test_random_quantifier_alternation(self):
        rng = random.Random(260830)
        atoms = ("X(x)", "X(y)", "Y(x)", "Y(y)", "E(x,y)", "E(y,x)", "x=y", "P(x)")
        for _ in range(24):
            a, b, c = (rng.choice(atoms) for _ in range(3))
            op = rng.choice(("&", "|", "->", "<->"))
            prefix = " ".join(f"{rng.choice(('exists', 'forall'))} {v}:" for v in ("X", "x", "Y", "y"))
            text = f"{prefix} (({a} {op} ~{b}) <-> {c})"
            compiled = compile_mso(text)
            for n in range(4):
                with self.subTest(formula=text, n=n):
                    self.assertEqual((compiled.fomc(n), compiled.ufomc(n)), brute_counts(text, n))

    def test_connectedness_and_reachability(self):
        c = compile_mso(ONE_CYCLE)
        self.assertEqual(c.sequence(9), (1,) + tuple(factorial(n - 1) for n in range(1, 10)))
        self.assertEqual(c.sequence(9, labeled=False), (1,) * 10)
        # Force nonemptiness separately; universal connectedness accepts empty.
        c = compile_mso(f"({CONNECTED}) & (exists x: x=x)")
        self.assertEqual(c.sequence(6), (0, 1, 3, 17, 142, 1569, 21576))

    def test_fixed_sentence_rooted_trees(self):
        c = compile_mso(ROOTED_TREES)
        self.assertEqual(c.sequence(20), (0,) + tuple(n ** (n - 1) for n in range(1, 21)))
        self.assertEqual(c.sequence(12, labeled=False), (0, 1, 1, 2, 4, 9, 20, 48, 115, 286, 719, 1842, 4766))
        for n in range(4):
            self.assertEqual((c.fomc(n), c.ufomc(n)), brute_counts(ROOTED_TREES, n))

    def test_nullaries_and_explicit_vocabulary(self):
        c = compile_mso("Q() | ~Q()", predicates=("P",))
        self.assertEqual(c.sequence(5), tuple(2 * 2 ** n * n ** n for n in range(6)))
        self.assertEqual(c.ufomc(0), 2)
        self.assertEqual(len(c.branches), 2)
        with self.assertRaises(ValueError):
            _ = c.specification
        self.assertEqual(mso_fomc("exists P: forall x: P(x)", 3), 27)
        self.assertEqual(mso_fomc("exists P: forall x: P(x)", 3, predicates=("P",)), 216)

    def test_custom_function_and_relational_input(self):
        self.assertEqual(mso_fomc("forall x: g(x)=x", 6, function_symbol="g"), 1)
        self.assertEqual(mso_ufomc("forall y: exists x: Edge(x,y)", 6, edge_predicate="Edge"), 11)
        self.assertEqual(mso_fomc("forall x: E(x,x)", 5), 1)

    def test_serialization_and_algebra_laws(self):
        for text in (EVEN_CYCLES, CONNECTED, "forall x: f(f(x))=x", "exists x: P(x)"):
            c = compile_mso(text)
            specification = MSOSpecification.from_dict(json.loads(json.dumps(c.specification.to_dict())))
            self.assertEqual(specification, c.specification)
            self.assertEqual(evaluate_mso_specification(specification, 7), c.sequence(7))
            self.assertEqual(evaluate_mso_specification(specification, 7, labeled=False), c.sequence(7, labeled=False))
        invalid = compile_mso("TRUE").specification.to_dict()
        invalid["close"] = [2]
        with self.assertRaises(ValueError):
            MSOSpecification.from_dict(invalid)

    def test_emptiness_and_identity_nonempty_realizations(self):
        self.assertTrue(compile_mso("TRUE").has_model())
        self.assertFalse(compile_mso("FALSE").has_model(nonempty=False))
        c = compile_mso("forall x: x!=x")
        self.assertFalse(c.has_model())
        self.assertTrue(c.has_model(nonempty=False))
        self.assertEqual((c.fomc(0), c.ufomc(0)), (1, 1))

    def test_limits_and_invalid_sizes(self):
        for limits in (MSOCompilerLimits(max_states=3), MSOCompilerLimits(max_transitions=1),
                       MSOCompilerLimits(max_colors=1), MSOCompilerLimits(max_formula_nodes=1)):
            with self.subTest(limits=limits), self.assertRaises(MSOCompilationLimitError):
                compile_mso("forall x: P(x)", limits=limits)
        with self.assertRaises(MSOCompilationLimitError):
            compile_mso("forall x: f^1000000000(x)=x")
        with self.assertRaises(ValueError):
            MSOCompilerLimits(max_states=0)
        with self.assertRaises(TypeError):
            compile_mso("TRUE", limits={})
        c = compile_mso("TRUE")
        for size in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                c.fomc(size)


if __name__ == "__main__":
    unittest.main()
