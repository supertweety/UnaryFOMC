"""Simplification must not change the counted vocabulary.

The models of a sentence are structures over every relation symbol of the
sentence as written.  A Boolean constant, or the truth value that the nullary
case split substitutes for a nullary symbol, can remove all atoms of a
relation symbol from a branch of the sentence.  The symbol is then still
interpreted freely in the structures of that branch.
"""

from itertools import permutations, product
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unaryfomc import (
    C1FAtom, C1FBinary, C1FBooleanConstant, C1FEquality, C1FNot,
    C1FQuantifier, c1f_fomc, c1f_ufomc, compile_c1f_fomc, compile_c1f_ufomc,
    compile_mso, parse_c1f_sentence,
)


REMOVED_BY_NULLARY = "Z | (exists x: P(x))"
REMOVED_BY_CONSTANT = "(exists x: P(x)) | TRUE"

# Sentences in the common input language of the C1 and MSO counters, each
# with its vocabulary besides f.  Simplification removes a unary predicate
# from at least one nullary branch of every sentence.
MONADIC_CASES = (
    (REMOVED_BY_NULLARY, {"P": 1, "Z": 0}),
    (REMOVED_BY_CONSTANT, {"P": 1}),
    # No nullary symbol: P vanishes inside a quantifier, the rest has depth 2.
    ("(forall x: (P(x) | TRUE)) & (forall x: f(f(x)) != x)", {"P": 1}),
    # No nullary symbol: Q vanishes, P stays and is read at depth 1.
    ("(forall x: (Q(x) -> TRUE)) & (exists x: (P(x) & ~P(f(x))))",
     {"P": 1, "Q": 1}),
    # The two nullary branches keep different predicates.
    ("(Z & (forall x: P(x))) | (~Z & (exists x: Q(f(x))))",
     {"P": 1, "Q": 1, "Z": 0}),
)

OPTIONS = tuple(
    dict(profile_pruning=pruning, twin_compression=twins)
    for pruning in (True, False) for twins in (True, False)
)


def holds(node, assignment, function, relations):
    """Truth of the parse tree, which is never simplified."""

    def value(term):
        result = assignment[term.variable]
        for _ in range(term.exponent):
            result = function[result]
        return result

    if isinstance(node, C1FBooleanConstant):
        return node.value
    if isinstance(node, C1FAtom):
        return tuple(map(value, node.arguments)) in relations[node.predicate]
    if isinstance(node, C1FEquality):
        return value(node.left) == value(node.right)
    if isinstance(node, C1FNot):
        return not holds(node.body, assignment, function, relations)
    if isinstance(node, C1FBinary):
        left = holds(node.left, assignment, function, relations)
        right = holds(node.right, assignment, function, relations)
        return {"&": left and right, "|": left or right,
                "->": not left or right, "<->": left == right}[node.op]
    if isinstance(node, C1FQuantifier) and node.comparator is None:
        witnesses = (
            holds(node.body, {**assignment, node.variable: element},
                  function, relations)
            for element in range(len(function))
        )
        return all(witnesses) if node.kind == "forall" else any(witnesses)
    raise AssertionError(type(node))


def brute_force_counts(sentence, vocabulary, n):
    """Count the n-element models over ``vocabulary`` and their orbits."""
    node = parse_c1f_sentence(sentence)
    names = sorted(vocabulary)
    interpretations = []
    for name in names:
        tuples = tuple(product(range(n), repeat=vocabulary[name]))
        interpretations.append([
            frozenset(t for t, chosen in zip(tuples, bits) if chosen)
            for bits in product((False, True), repeat=len(tuples))
        ])
    relabelings = tuple(permutations(range(n)))
    labeled, unlabeled, seen = 0, 0, set()
    for function in product(range(n), repeat=n):
        for chosen in product(*interpretations):
            if not holds(node, {}, function, dict(zip(names, chosen))):
                continue
            labeled += 1
            if (function, chosen) in seen:
                continue
            unlabeled += 1
            for relabel in relabelings:
                image = [0] * n
                for element in range(n):
                    image[relabel[element]] = relabel[function[element]]
                seen.add((tuple(image), tuple(
                    frozenset(tuple(relabel[e] for e in t) for t in relation)
                    for relation in chosen
                )))
    return labeled, unlabeled


class C1FVocabularyTests(unittest.TestCase):
    def test_reported_counts(self):
        # Z true: all structures over {f, P}.  Z false: those with P nonempty.
        self.assertEqual(
            [int(c1f_ufomc(REMOVED_BY_NULLARY, n)) for n in range(5)],
            [1 + 0, 2 + 1, 10 + 7, 44 + 37, 218 + 199],
        )
        self.assertEqual(
            [int(c1f_fomc(REMOVED_BY_NULLARY, n)) for n in range(5)],
            [1, 3, 28, 405, 7936],
        )
        # All structures over {f, P}, and not only those over {f}.
        self.assertEqual(
            [int(c1f_ufomc(REMOVED_BY_CONSTANT, n)) for n in range(5)],
            [1, 2, 10, 44, 218],
        )
        self.assertEqual(
            [int(c1f_fomc(REMOVED_BY_CONSTANT, n)) for n in range(5)],
            [1, 2, 16, 216, 4096],
        )

    def test_agrees_with_mso_counter_and_exhaustive_enumeration(self):
        for sentence, vocabulary in MONADIC_CASES:
            mso = compile_mso(sentence)
            self.assertEqual(
                set(mso.predicates) | set(mso.nullary_predicates),
                set(vocabulary),
            )
            expected = [brute_force_counts(sentence, vocabulary, n)
                        for n in range(5)]
            with self.subTest(sentence=sentence, counter="mso"):
                self.assertEqual(
                    list(zip(mso.sequence(4), mso.sequence(4, labeled=False))),
                    expected,
                )
            for options in OPTIONS:
                with self.subTest(sentence=sentence, **options):
                    labeled = compile_c1f_fomc(sentence, **options)
                    unlabeled = compile_c1f_ufomc(sentence, **options)
                    self.assertEqual(
                        [(int(labeled.fomc(n)), int(unlabeled.ufomc(n)))
                         for n in range(5)],
                        expected,
                    )

    def test_removed_predicate_counts_like_a_declared_one(self):
        removed = compile_c1f_ufomc(REMOVED_BY_CONSTANT)
        tautology = compile_c1f_ufomc("forall x: (P(x) | ~P(x))")
        declared = compile_mso("TRUE", predicates=("P",))
        for n in range(9):
            with self.subTest(n=n):
                self.assertEqual(int(removed.ufomc(n)), int(tautology.ufomc(n)))
                self.assertEqual(int(removed.ufomc(n)), declared.ufomc(n))
                self.assertEqual(int(removed.fomc(n)), 2**n * n**n)

    def test_every_nullary_branch_keeps_the_vocabulary(self):
        compiled = compile_c1f_ufomc(REMOVED_BY_NULLARY)
        z_false, z_true = compiled.branches
        self.assertEqual(z_false.monadic.unary_predicates, ("P",))
        self.assertEqual(z_true.monadic.unary_predicates, ("P",))
        # The branch TRUE observes nothing, so its two profiles are twins.
        statistics = z_true.profile_statistics()
        self.assertEqual(statistics.raw_profiles, 2)
        self.assertEqual(statistics.twin_multiplicities, (2,))
        self.assertEqual(
            z_false.profile_statistics().twin_multiplicities, (1, 1))

    def test_pruning_keeps_every_colouring_of_a_removed_predicate(self):
        sentence = "(forall x: (P(x) | TRUE)) & (forall x: f(f(x)) != x)"
        statistics = compile_c1f_ufomc(sentence).profile_statistics()[0]
        # Depth 2 and one predicate: 2^3 + 1*2 + 2*4 profiles.  The filter
        # keeps the windows x, f(x), f(f(x)) with three and with two distinct
        # elements whose last two coincide, in all 2^3 + 2^2 colourings.
        self.assertEqual(statistics.raw_profiles, 18)
        self.assertEqual(statistics.pruned_profiles, 12)
        self.assertEqual(sum(statistics.twin_multiplicities), 12)

    def test_removed_relation_stays_in_the_labeled_vocabulary(self):
        cases = (
            ("Z | (forall x: R(x,f(x)))", {"R": 2, "Z": 0}, 3,
             lambda n: n**n * 2**(n * n - n) * (2**n + 1)),
            ("forall x: (R(x,f(x)) | TRUE)", {"R": 2}, 3,
             lambda n: n**n * 2**(n * n)),
            # R is removed, S is still compressed to its trace on f.
            ("(forall x: (R(x,f(x)) | TRUE)) & "
             "(forall x: (S(f(x),x) <-> P(x)))", {"P": 1, "R": 2, "S": 2}, 2,
             lambda n: n**n * 2**(2 * n * n)),
        )
        for sentence, vocabulary, bruteforce_bound, closed_form in cases:
            for options in OPTIONS:
                compiled = compile_c1f_fomc(sentence, **options)
                for n in range(6):
                    with self.subTest(sentence=sentence, n=n, **options):
                        self.assertEqual(int(compiled.fomc(n)), closed_form(n))
            for n in range(bruteforce_bound + 1):
                with self.subTest(sentence=sentence, bruteforce_n=n):
                    self.assertEqual(
                        brute_force_counts(sentence, vocabulary, n)[0],
                        closed_form(n),
                    )

    def test_trace_predicates_avoid_removed_symbols(self):
        # C1F_R_trace_0 is the name that the compression of R would pick if
        # the removed unary predicate of that name were not reserved.
        sentence = ("(forall x: (C1F_R_trace_0(x) | TRUE)) & "
                    "(forall x: R(x,f(x)))")
        vocabulary = {"C1F_R_trace_0": 1, "R": 2}
        compiled = compile_c1f_fomc(sentence)
        self.assertEqual(len(compiled.branches[0].monadic.unary_predicates), 2)
        for n in range(6):
            with self.subTest(n=n):
                self.assertEqual(int(compiled.fomc(n)), n**n * 2**(n * n))
        for n in range(3):
            with self.subTest(bruteforce_n=n):
                self.assertEqual(
                    brute_force_counts(sentence, vocabulary, n)[0],
                    n**n * 2**(n * n),
                )


if __name__ == "__main__":
    unittest.main()
