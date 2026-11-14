"""The numerical examples of Part II.

Part II is the companion paper on unlabeled counting.  Every test below pins
numbers that the paper prints, so that the paper and the code cannot drift
apart unnoticed.  The examples are named by their titles in the paper, because
the numbering of the manuscript may still change.

Every conjunct of a sentence below is in parentheses.  A quantifier extends as
far to the right as possible, and without the parentheses the first quantifier
would extend over the whole sentence.  For the running example this would
change the count on the empty domain from 0 to 1.
"""

from itertools import permutations, product
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unaryfomc import c1f_fomc, c1f_ufomc, compile_c1f_ufomc, compile_mso


# "All unlabeled endofunctions": the number of functions on n elements up to
# conjugation, for n = 0..7 (OEIS A001372).  The example "Endofunctions"
# prints the values for n = 1..6.
ENDOFUNCTIONS = (1, 1, 3, 7, 19, 47, 130, 343)

# "Running example": every fixed point of f is marked by P, no directed
# 2-cycle is monochromatic, and exactly two elements are marked.
RUNNING = (
    "(forall x: (f(x) = x -> P(x))) & "
    "(forall x: ((f(f(x)) = x & ~(f(x) = x)) -> ~(P(x) <-> P(f(x))))) & "
    "(exists=2 x: P(x))"
)
# The same sentence for the MSO counter.  Its input language has no counting
# quantifiers, so "exactly two" is written out with ordinary quantifiers.
RUNNING_MSO = (
    "(forall x: (f(x) = x -> P(x))) & "
    "(forall x: ((f(f(x)) = x & ~(f(x) = x)) -> ~(P(x) <-> P(f(x))))) & "
    "(exists x, y: (~(x = y) & P(x) & P(y) & "
    "forall z: (P(z) -> (z = x | z = y))))"
)
# The table of the running example, for n = 1..8.  The sentence has no model
# on the empty domain.
RUNNING_TABLE = (0, 2, 8, 33, 119, 434, 1513, 5239)

# "Connected mappings": the domain is nonempty and the functional digraph is
# weakly connected.  The values are for n = 1..6 (OEIS A002861).
CONNECTED = (
    "(exists x: x = x) & "
    "(forall X: (((exists x: X(x)) & (exists x: ~X(x))) -> "
    "(exists x: ~(X(x) <-> X(f(x))))))"
)
CONNECTED_MAPPINGS = (1, 2, 4, 9, 20, 51)

# "Loopless directed graphs": f is the identity and the binary relation R is
# irreflexive.  The paper gives the number of labeled models, 2^(n(n-1)).
LOOPLESS_DIGRAPHS = "(forall x: f(x) = x) & (forall x: ~R(x, x))"


def _running_example_holds(function, marked):
    if sum(marked) != 2:
        return False
    for vertex, successor in enumerate(function):
        if successor == vertex:
            if not marked[vertex]:
                return False
        elif function[successor] == vertex and marked[vertex] == marked[successor]:
            return False
    return True


def _running_example_classes(domain_size):
    """Count the models of the running example up to isomorphism, literally."""
    relabelings = tuple(permutations(range(domain_size)))
    seen = set()
    classes = 0
    for function in product(range(domain_size), repeat=domain_size):
        for marked in product((False, True), repeat=domain_size):
            if not _running_example_holds(function, marked):
                continue
            if (function, marked) in seen:
                continue
            classes += 1
            for relabeling in relabelings:
                image_function = [0] * domain_size
                image_marked = [False] * domain_size
                for vertex in range(domain_size):
                    image_function[relabeling[vertex]] = relabeling[function[vertex]]
                    image_marked[relabeling[vertex]] = marked[vertex]
                seen.add((tuple(image_function), tuple(image_marked)))
    return classes


class PartTwoExampleTests(unittest.TestCase):
    def test_endofunctions(self):
        self.assertEqual(
            tuple(c1f_ufomc("TRUE", n) for n in range(8)), ENDOFUNCTIONS
        )
        self.assertEqual(
            compile_mso("TRUE").sequence(7, labeled=False), ENDOFUNCTIONS
        )

    def test_running_example_table(self):
        compiled = compile_c1f_ufomc(RUNNING)
        self.assertEqual(compiled.ufomc(0), 0)
        self.assertEqual(
            tuple(compiled.ufomc(n) for n in range(1, 9)), RUNNING_TABLE
        )

    def test_running_example_table_under_the_other_options(self):
        # Pruning and twin merging must not change a count.  The first five
        # columns keep this test fast.
        for profile_pruning, twin_compression in (
            (True, False), (False, True), (False, False)
        ):
            compiled = compile_c1f_ufomc(
                RUNNING,
                profile_pruning=profile_pruning,
                twin_compression=twin_compression,
            )
            with self.subTest(
                profile_pruning=profile_pruning,
                twin_compression=twin_compression,
            ):
                self.assertEqual(
                    tuple(compiled.ufomc(n) for n in range(6)),
                    (0,) + RUNNING_TABLE[:5],
                )

    def test_running_example_by_enumeration(self):
        # "Exhaustive enumeration gives" the table.  Enumeration is feasible
        # here up to five elements.
        for domain_size in range(6):
            with self.subTest(domain_size=domain_size):
                self.assertEqual(
                    _running_example_classes(domain_size),
                    ((0,) + RUNNING_TABLE)[domain_size],
                )

    def test_running_example_profiles(self):
        # The window (x, f(x), f(f(x))) has 8 + 2 + 4 + 4 = 18 profiles.  The
        # profile of an unmarked fixed point and the two profiles of a
        # monochromatic 2-cycle are deleted, which leaves 15 profiles.  These
        # two numbers describe the compilation and not a count: if the
        # pruning is ever strengthened, they change here and in the paper.
        (pruned,) = compile_c1f_ufomc(RUNNING).profile_statistics()
        self.assertEqual(pruned.raw_profiles, 18)
        self.assertEqual(pruned.pruned_profiles, 15)
        (unpruned,) = compile_c1f_ufomc(
            RUNNING, profile_pruning=False
        ).profile_statistics()
        self.assertEqual(unpruned.raw_profiles, 18)
        self.assertEqual(unpruned.pruned_profiles, 18)

    def test_running_example_with_the_mso_counter(self):
        sequence = compile_mso(RUNNING_MSO).sequence(8, labeled=False)
        self.assertEqual(sequence, (0,) + RUNNING_TABLE)

    def test_connected_mappings(self):
        sequence = compile_mso(CONNECTED).sequence(6, labeled=False)
        # The sentence requires a nonempty domain.
        self.assertEqual(sequence, (0,) + CONNECTED_MAPPINGS)

    def test_loopless_directed_graphs(self):
        for n in range(6):
            with self.subTest(n=n):
                self.assertEqual(
                    c1f_fomc(LOOPLESS_DIGRAPHS, n), 2 ** (n * (n - 1))
                )
        # Counting up to isomorphism is restricted to monadic vocabularies.
        with self.assertRaises(ValueError):
            c1f_ufomc(LOOPLESS_DIGRAPHS, 3)


if __name__ == "__main__":
    unittest.main()
