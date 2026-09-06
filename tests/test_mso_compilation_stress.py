from itertools import permutations, product
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from mso_compilation_stress import (
    branch_equivalence, duplicate_branches_sentence, duplicate_branches_specification,
    graph_state_lower_bound,
)
from unaryfomc import compile_mso, evaluate_mso_formula, evaluate_mso_specification, parse_mso_sentence
from unaryfomc.mso_syntax import MSOBinary, MSONot, MSOQuantifier


def signatures(function, depth):
    values = [None] * len(function)
    for _ in range(depth):
        values = [frozenset(values[u] for u in range(len(function)) if u != x and function[u] == x)
                  for x in range(len(function))]
    return values


def duplicate_truth(function, depth):
    values = signatures(function, depth)
    roots = [values[x] for x in range(len(function)) if function[x] == x]
    return len(roots) != len(set(roots))


def code_trees(depth):
    if depth == 0:
        return ((),)
    previous = code_trees(depth - 1)
    return tuple(tuple(tree for i, tree in enumerate(previous) if mask >> i & 1)
                 for mask in range(1 << len(previous)))


def as_function(forest):
    function = []

    def add(tree, parent=None):
        root = len(function)
        function.append(root if parent is None else parent)
        for child in tree:
            add(child, root)

    for tree in forest:
        add(tree)
    return tuple(function)


class MSOStressTests(unittest.TestCase):
    def test_linear_size_and_fixed_signature(self):
        for depth in range(8):
            text = duplicate_branches_sentence(depth)
            self.assertLessEqual(len(text), 145 * depth + 43)
            formula = parse_mso_sentence(text)
            variables = set()

            def visit(node):
                if isinstance(node, MSOQuantifier):
                    self.assertEqual(node.sort, "node")
                    variables.add(node.variable)
                    visit(node.body)
                elif isinstance(node, MSOBinary):
                    visit(node.left)
                    visit(node.right)
                elif isinstance(node, MSONot):
                    visit(node.body)

            visit(formula)
            self.assertLessEqual(variables, {"x", "y", "u", "v"})

    def test_formula_against_independent_branch_signatures(self):
        for depth in range(3):
            formula = parse_mso_sentence(duplicate_branches_sentence(depth))
            for n in range(4):
                for function in product(range(n), repeat=n):
                    self.assertEqual(evaluate_mso_formula(formula, function), duplicate_truth(function, depth))

    def test_distinguishing_contexts(self):
        for depth in range(3):
            trees = code_trees(depth)
            acceptance_vectors = set()
            for mask in range(1 << len(trees)):
                forest = tuple(tree for i, tree in enumerate(trees) if mask >> i & 1)
                self.assertFalse(duplicate_truth(as_function(forest), depth))
                answers = tuple(duplicate_truth(as_function((*forest, tree)), depth) for tree in trees)
                self.assertEqual(answers, tuple(bool(mask >> i & 1) for i in range(len(trees))))
                acceptance_vectors.add(answers)
            self.assertEqual(len(acceptance_vectors), 1 << len(trees))
            # An already accepting state is distinct from all these rejecting ones.
            self.assertTrue(duplicate_truth(as_function((trees[0], trees[0])), depth))
        trees = code_trees(3)
        self.assertEqual(len(trees), 16)
        self.assertEqual(len({signatures(as_function((t,)), 3)[0] for t in trees}), 16)

    def test_generic_and_specialized_specifications(self):
        for depth in range(3):
            s = duplicate_branches_specification(depth).validate(check_laws=True)
            self.assertEqual(s.state_counts["graphs"], int(graph_state_lower_bound(depth)))
            expected = tuple(sum(duplicate_truth(f, depth) for f in product(range(n), repeat=n))
                             for n in range(6))
            self.assertEqual(evaluate_mso_specification(s, 5), expected)
            for n in range(5):
                orbits = set()
                for function in product(range(n), repeat=n):
                    if not duplicate_truth(function, depth):
                        continue
                    candidates = []
                    for permutation in permutations(range(n)):
                        renamed = [0] * n
                        for x, target in enumerate(function):
                            renamed[permutation[x]] = permutation[target]
                        candidates.append(tuple(renamed))
                    orbits.add(min(candidates))
                self.assertEqual(evaluate_mso_specification(s, n, labeled=False)[-1], len(orbits))
            if depth < 2:
                c = compile_mso(duplicate_branches_sentence(depth))
                self.assertEqual(c.specification.state_counts["graphs"], s.state_counts["graphs"])
                for labeled in (True, False):
                    self.assertEqual(c.sequence(8, labeled=labeled),
                                     evaluate_mso_specification(s, 8, labeled=labeled))

    def test_bounds_and_resource_guards(self):
        self.assertEqual([graph_state_lower_bound(h) for h in range(5)],
                         ["3", "5", "17", "65537", "2^(65536)+1"])
        for depth in (3, 4, 100):
            with self.assertRaises(ValueError):
                duplicate_branches_specification(depth)
        for depth in (-1, True, 0.5):
            with self.assertRaises(ValueError):
                branch_equivalence(depth)


if __name__ == "__main__":
    unittest.main()
