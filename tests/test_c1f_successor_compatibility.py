"""Successor compatibility must preserve the whole colored overlap."""

from itertools import permutations, product
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unaryfomc.c1f import (
    _Profile, _build_successor_matrix, _profile_overlap_key, _profiles_for,
    _quotient_twin_profiles, _successor_compatible, parse_c1f_formula,
    parse_c1f_sentence,
)
from unaryfomc import c1f_fomc, c1f_ufomc


def semantic_compatibility(left, right, depth):
    """The defining conditions, independent of canonical keys and bucketing."""
    return (
        all(left.unary_by_position[i + 1] == right.unary_by_position[i]
            for i in range(depth))
        and all(
            (left.equivalence[i + 1] == left.equivalence[j + 1])
            == (right.equivalence[i] == right.equivalence[j])
            for i in range(depth) for j in range(depth)
        )
    )


def brute_alternating_counts(n, bijective):
    labeled = 0
    orbits = set()
    for function in product(range(n), repeat=n):
        if bijective and len(set(function)) != n:
            continue
        for colors in product((False, True), repeat=n):
            if any(colors[x] == colors[function[x]] for x in range(n)):
                continue
            labeled += 1
            images = []
            for relabel in permutations(range(n)):
                new_function, new_colors = [0] * n, [False] * n
                for x in range(n):
                    new_function[relabel[x]] = relabel[function[x]]
                    new_colors[relabel[x]] = colors[x]
                images.append((tuple(new_function), tuple(new_colors)))
            orbits.add(min(images))
    return labeled, len(orbits)


class TestSuccessorCompatibility(unittest.TestCase):
    def test_equal_shape_conflicting_unary_overlap_is_rejected(self):
        left = _Profile(0, ((False,), (True,)), (0, 1), None)
        conflict = _Profile(1, ((False,), (False,)), (0, 1), None)
        matching = _Profile(2, ((True,), (False,)), (0, 1), None)
        self.assertEqual(
            _profile_overlap_key(left, 1, 1)[0],
            _profile_overlap_key(conflict, 0, 1)[0],
        )
        self.assertFalse(_successor_compatible(left, conflict, 1))
        self.assertTrue(_successor_compatible(left, matching, 1))
        matrix = _build_successor_matrix((left, conflict, matching), 1)
        self.assertFalse(matrix[0][1])
        self.assertTrue(matrix[0][2])

    def test_every_unary_predicate_is_in_the_key(self):
        left = _Profile(0, ((False, False), (True, True)), (0, 1), None)
        right = _Profile(1, ((True, False), (False, False)), (0, 1), None)
        self.assertFalse(_successor_compatible(left, right, 1))

    def test_equal_colors_do_not_hide_equality_conflict(self):
        left = _Profile(0, ((True,),) * 3, (0, 1, 2), None)
        right = _Profile(1, ((True,),) * 3, (0, 0, 0), 1)
        self.assertFalse(_successor_compatible(left, right, 2))

    def test_class_names_are_canonicalized_but_colors_are_preserved(self):
        left = _Profile(0, ((False,), (True,), (False,), (True,)), (0, 1, 2, 1), None)
        right = _Profile(1, ((True,), (False,), (True,), (False,)), (0, 1, 0, 1), 2)
        self.assertEqual(
            _profile_overlap_key(left, 1, 3),
            ((0, 1, 0), ((True,), (False,), (True,))),
        )
        self.assertTrue(_successor_compatible(left, right, 3))

    def test_bucket_matrix_matches_definition_exhaustively(self):
        for predicate_count, depth in [
            *product(range(3), range(4)), (1, 4),
        ]:
            with self.subTest(predicate_count=predicate_count, depth=depth):
                predicates = tuple(f"P{i}" for i in range(predicate_count))
                profiles = _profiles_for(predicates, depth)
                matrix = _build_successor_matrix(profiles, depth)
                expected = tuple(tuple(semantic_compatibility(a, b, depth)
                                       for b in profiles) for a in profiles)
                self.assertEqual(matrix, expected)

    def test_pruning_and_twins_keep_semantic_transition_signatures(self):
        depth = 2
        profiles = _profiles_for(
            ("P", "Q"), depth,
            (parse_c1f_formula("P(x) | Q(f(x))"),),
        )
        matrix = _build_successor_matrix(profiles, depth)
        expected = tuple(tuple(semantic_compatibility(a, b, depth)
                               for b in profiles) for a in profiles)
        self.assertEqual(matrix, expected)
        representatives, quotient, multiplicities = _quotient_twin_profiles(
            profiles, matrix, parse_c1f_sentence("TRUE"), {"P": 0, "Q": 1},
        )
        self.assertEqual(quotient, tuple(
            tuple(semantic_compatibility(a, b, depth) for b in representatives)
            for a in representatives
        ))
        # Even with no formula observations, every member of a twin class
        # must have exactly the same colored incoming and outgoing edges.
        def signature(profile):
            return (
                profile.visible_cycle_length,
                tuple(semantic_compatibility(profile, b, depth) for b in profiles),
                tuple(semantic_compatibility(a, profile, depth) for a in profiles),
            )
        for representative, multiplicity in zip(representatives, multiplicities):
            self.assertEqual(multiplicity, sum(
                signature(p) == signature(representative) for p in profiles))
        self.assertEqual(sum(multiplicities), len(profiles))

    def test_alternating_color_counts_match_independent_enumeration(self):
        for bijective in (False, True):
            sentence = "(forall x: (P(x) <-> ~P(f(x))))"
            if bijective:
                sentence += " & (forall x: (exists=1 y: (f(y)=x & TRUE)))"
            for n in range(4):
                labeled, unlabeled = brute_alternating_counts(n, bijective)
                for pruning, twins in product((False, True), repeat=2):
                    with self.subTest(n=n, bijective=bijective, pruning=pruning, twins=twins):
                        options = dict(profile_pruning=pruning, twin_compression=twins)
                        self.assertEqual(c1f_fomc(sentence, n, **options), labeled)
                        self.assertEqual(c1f_ufomc(sentence, n, **options), unlabeled)


if __name__ == "__main__":
    unittest.main()
