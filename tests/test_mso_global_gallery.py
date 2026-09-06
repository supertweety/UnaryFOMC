from itertools import permutations, product
from math import comb, factorial
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from mso_global_gallery import EXAMPLES
from unaryfomc import compile_mso, evaluate_mso_formula, parse_mso_sentence


def directed_cycles(function):
    seen, result = set(), []
    for start in range(len(function)):
        path, positions, x = [], {}, start
        while x not in seen and x not in positions:
            positions[x] = len(path)
            path.append(x)
            x = function[x]
        if x in positions:
            result.append(tuple(path[positions[x]:]))
        seen.update(path)
    return result


def accepts(name, function, marked=0):
    cycles = directed_cycles(function)
    if name == "exactly_two_components":
        return len(cycles) == 2
    if name == "cycle_lengths_divisible_by_three":
        return all(len(c) % 3 == 0 for c in cycles)
    if name == "one_marked_cycle_vertex_per_component":
        cyclic_mask = sum(1 << x for c in cycles for x in c)
        return (marked & ~cyclic_mask == 0
                and all(sum((marked >> x) & 1 for x in c) == 1 for c in cycles))
    if len(cycles) != 1 or len(cycles[0]) != 1:
        return False
    if name == "rooted_trees":
        return True
    children = [0] * len(function)
    for x, parent in enumerate(function):
        if x != parent:
            children[parent] += 1
    root = cycles[0][0]
    for leaf, count in enumerate(children):
        if not count:
            distance, x = 0, leaf
            while x != root:
                x = function[x]
                distance += 1
            if distance % 2:
                return False
    return True


def brute_counts(name, n):
    labeled, orbits = 0, set()
    marked_example = name == "one_marked_cycle_vertex_per_component"
    relabelings = tuple(permutations(range(n)))
    for function in product(range(n), repeat=n):
        for marked in range((1 << n) if marked_example else 1):
            if not accepts(name, function, marked):
                continue
            labeled += 1
            representatives = []
            for permutation in relabelings:
                renamed = [0] * n
                for x, target in enumerate(function):
                    renamed[permutation[x]] = permutation[target]
                new_marked = sum(((marked >> x) & 1) << permutation[x] for x in range(n))
                representatives.append((tuple(renamed), new_marked))
            orbits.add(min(representatives))
    return labeled, len(orbits)


def component_counts(n, weight):
    return sum(weight(length) * (factorial(n - 1) if length == n else
                                factorial(n) // factorial(n - length) * n ** (n - length - 1))
               for length in range(1, n + 1))


def assemblies(components):
    result = [1]
    for n in range(1, len(components)):
        result.append(sum(comb(n - 1, k - 1) * components[k] * result[n - k]
                          for k in range(1, n + 1)))
    return tuple(result)


def even_leaf_counts(bound):
    even, odd = [0], [0]
    even_forests, odd_forests = [1], [1]
    for n in range(1, bound + 1):
        even.append(n * odd_forests[n - 1])
        odd.append(n * even_forests[n - 1] if n > 1 else 0)
        for trees, forests in ((even, even_forests), (odd, odd_forests)):
            forests.append(sum(comb(n - 1, k - 1) * trees[k] * forests[n - k]
                               for k in range(1, n + 1)))
    return tuple(even)


class MSOGlobalGalleryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = {name: compile_mso(sentence) for name, sentence in EXAMPLES.items()}

    def test_small_algebras_and_exact_orbits(self):
        for name, compiled in self.compiled.items():
            self.assertLessEqual(sum(compiled.specification.state_counts.values()), 12)
            self.assertLessEqual(compiled.statistics.peak_intermediate_states, 300)
            for n in range(5):
                with self.subTest(name=name, n=n):
                    self.assertEqual((compiled.fomc(n), compiled.ufomc(n)), brute_counts(name, n))

    def test_formula_semantics_against_graph_algorithms(self):
        for name, sentence in EXAMPLES.items():
            formula = parse_mso_sentence(sentence)
            for n in range(4):
                for function in product(range(n), repeat=n):
                    for marked in range((1 << n) if name == "one_marked_cycle_vertex_per_component" else 1):
                        predicates = {"P": {x for x in range(n) if marked >> x & 1}}
                        self.assertEqual(evaluate_mso_formula(formula, function, predicates),
                                         accepts(name, function, marked), (name, function, marked))

    def test_labeled_combinatorial_recurrences(self):
        bound = 10
        self.assertEqual(self.compiled["rooted_trees"].sequence(bound),
                         (0,) + tuple(n ** (n - 1) for n in range(1, bound + 1)))
        self.assertEqual(self.compiled["rooted_trees_even_depth_leaves"].sequence(bound), even_leaf_counts(bound))
        connected = [0] + [component_counts(n, lambda length: 1) for n in range(1, bound + 1)]
        two = tuple(sum(comb(n, k) * connected[k] * connected[n - k] for k in range(1, n)) // 2
                    for n in range(bound + 1))
        self.assertEqual(self.compiled["exactly_two_components"].sequence(bound), two)
        for name, weight in (("cycle_lengths_divisible_by_three", lambda length: int(length % 3 == 0)),
                             ("one_marked_cycle_vertex_per_component", lambda length: length)):
            components = [0] + [component_counts(n, weight) for n in range(1, bound + 1)]
            self.assertEqual(self.compiled[name].sequence(bound), assemblies(components))


if __name__ == "__main__":
    unittest.main()
