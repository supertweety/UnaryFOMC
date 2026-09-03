from __future__ import annotations

from itertools import permutations, product
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from unaryfomc import (
    C1FAtom,
    C1FEquality,
    C1FFormula,
    C1FNot,
    C1FBinary,
    C1FBooleanConstant,
    C1FPredecessorCount,
    C1FQuantifier,
    C1FTerm,
    c1f_fomc,
    c1f_ufomc,
    c1f_predicate_arities,
    compile_c1f_fomc,
    compile_c1f_ufomc,
    parse_c1f_sentence,
    render_c1f_formula,
)


def _all_tuples(arity: int, domain_size: int):
    if arity == 0:
        return [tuple()]
    return list(product(range(domain_size), repeat=arity))


def _iterated_function_term(depth: int, variable: str = "x") -> str:
    return "f(" * depth + variable + ")" * depth


def _eval_term(term: C1FTerm, env: dict[str, int], function: tuple[int, ...]) -> int:
    value = env[term.variable]
    for _ in range(term.exponent):
        value = function[value]
    return value


def _eval_c1f(
    node: C1FFormula,
    env: dict[str, int],
    function: tuple[int, ...],
    interpretation: dict[str, set[tuple[int, ...]]],
    domain_size: int,
) -> bool:
    if isinstance(node, C1FBooleanConstant):
        return node.value
    if isinstance(node, C1FAtom):
        args = tuple(_eval_term(term, env, function) for term in node.arguments)
        return args in interpretation[node.predicate]
    if isinstance(node, C1FEquality):
        return _eval_term(node.left, env, function) == _eval_term(node.right, env, function)
    if isinstance(node, C1FNot):
        return not _eval_c1f(node.body, env, function, interpretation, domain_size)
    if isinstance(node, C1FBinary):
        left = _eval_c1f(node.left, env, function, interpretation, domain_size)
        right = _eval_c1f(node.right, env, function, interpretation, domain_size)
        if node.op == "&":
            return left and right
        if node.op == "|":
            return left or right
        if node.op == "->":
            return (not left) or right
        if node.op == "<->":
            return left == right
        raise ValueError(f"Unsupported boolean connective {node.op!r}")
    if isinstance(node, C1FPredecessorCount):
        anchor = env[node.anchor]
        satisfied = 0
        previous = env.get(node.witness)
        had_previous = node.witness in env
        try:
            for value in range(domain_size):
                if function[value] != anchor:
                    continue
                env[node.witness] = value
                if _eval_c1f(node.body, env, function, interpretation, domain_size):
                    satisfied += 1
        finally:
            if had_previous:
                env[node.witness] = previous  # type: ignore[assignment]
            else:
                env.pop(node.witness, None)
        if node.comparator == "==":
            return satisfied == node.count
        if node.comparator == "<":
            return satisfied < node.count
        if node.comparator == "<=":
            return satisfied <= node.count
        if node.comparator == ">":
            return satisfied > node.count
        if node.comparator == ">=":
            return satisfied >= node.count
        raise ValueError(f"Unsupported comparator {node.comparator!r}")
    if isinstance(node, C1FQuantifier):
        satisfied = 0
        previous = env.get(node.variable)
        had_previous = node.variable in env
        try:
            for value in range(domain_size):
                env[node.variable] = value
                if _eval_c1f(node.body, env, function, interpretation, domain_size):
                    satisfied += 1
        finally:
            if had_previous:
                env[node.variable] = previous  # type: ignore[assignment]
            else:
                env.pop(node.variable, None)
        if node.kind == "forall":
            return satisfied == domain_size
        if node.comparator is None:
            return satisfied >= 1
        if node.comparator == "==":
            return satisfied == node.count
        if node.comparator == "<":
            return satisfied < node.count
        if node.comparator == "<=":
            return satisfied <= node.count
        if node.comparator == ">":
            return satisfied > node.count
        if node.comparator == ">=":
            return satisfied >= node.count
        raise ValueError(f"Unsupported comparator {node.comparator!r}")
    raise TypeError(f"Unsupported C1F node {type(node)!r}")


def _bruteforce_c1f_model_count(sentence: str, domain_size: int) -> int:
    parsed = parse_c1f_sentence(sentence)
    arities = c1f_predicate_arities(parsed)
    predicate_names = sorted(arities)
    tuple_spaces = {
        name: _all_tuples(arity, domain_size)
        for name, arity in arities.items()
    }
    atom_count = sum(len(tuple_spaces[name]) for name in predicate_names)

    count = 0
    for function in product(range(domain_size), repeat=domain_size):
        for bits in product((False, True), repeat=atom_count):
            interpretation: dict[str, set[tuple[int, ...]]] = {}
            offset = 0
            for name in predicate_names:
                tuples = tuple_spaces[name]
                chosen: set[tuple[int, ...]] = set()
                for local_index, ground_tuple in enumerate(tuples):
                    if bits[offset + local_index]:
                        chosen.add(ground_tuple)
                interpretation[name] = chosen
                offset += len(tuples)
            if _eval_c1f(parsed, {}, function, interpretation, domain_size):
                count += 1
    return count


def _canonical_monadic_key(
    function: tuple[int, ...],
    interpretation: dict[str, set[tuple[int, ...]]],
    arities: dict[str, int],
    domain_size: int,
):
    if any(arity >= 2 for arity in arities.values()):
        raise ValueError("Unlabeled brute-force helper supports only monadic vocabularies")
    predicate_names = sorted(arities)
    candidates = []
    for permutation in permutations(range(domain_size)):
        transformed_function = [0] * domain_size
        for old_value, new_value in enumerate(permutation):
            transformed_function[new_value] = permutation[function[old_value]]

        transformed_predicates = []
        for predicate in predicate_names:
            arity = arities[predicate]
            if arity == 0:
                bits = (tuple() in interpretation[predicate],)
            else:
                transformed = [False] * domain_size
                for (old_value,) in interpretation[predicate]:
                    transformed[permutation[old_value]] = True
                bits = tuple(transformed)
            transformed_predicates.append((arity, bits))
        candidates.append((tuple(transformed_function), tuple(transformed_predicates)))
    return min(candidates)


def _bruteforce_c1f_unlabeled_count(sentence: str, domain_size: int) -> int:
    parsed = parse_c1f_sentence(sentence)
    arities = c1f_predicate_arities(parsed)
    if any(arity >= 2 for arity in arities.values()):
        raise ValueError("Unlabeled brute-force helper supports only monadic vocabularies")
    predicate_names = sorted(arities)
    tuple_spaces = {
        name: _all_tuples(arity, domain_size)
        for name, arity in arities.items()
    }
    atom_count = sum(len(tuple_spaces[name]) for name in predicate_names)

    isomorphism_types = set()
    for function in product(range(domain_size), repeat=domain_size):
        for bits in product((False, True), repeat=atom_count):
            interpretation: dict[str, set[tuple[int, ...]]] = {}
            offset = 0
            for name in predicate_names:
                tuples = tuple_spaces[name]
                chosen = {
                    ground_tuple
                    for local_index, ground_tuple in enumerate(tuples)
                    if bits[offset + local_index]
                }
                interpretation[name] = chosen
                offset += len(tuples)
            if _eval_c1f(parsed, {}, function, interpretation, domain_size):
                isomorphism_types.add(
                    _canonical_monadic_key(
                        function,
                        interpretation,
                        arities,
                        domain_size,
                    )
                )
    return len(isomorphism_types)


class C1FTests(unittest.TestCase):
    def test_parser_handles_nested_function_terms(self):
        parsed = parse_c1f_sentence("forall x: P(f(f(x))) -> x != f(x)")
        self.assertEqual(c1f_predicate_arities(parsed), {"P": 1})
        self.assertEqual(
            render_c1f_formula(parsed),
            "forall x: (P(f(f(x))) -> x != f(x))",
        )

    def test_parser_normalizes_guarded_predecessor_counts(self):
        parsed = parse_c1f_sentence(
            "forall x: exists=2 y: (Q(y) & f(y) = x)"
        )
        self.assertIsInstance(parsed, C1FQuantifier)
        self.assertIsInstance(parsed.body, C1FPredecessorCount)
        self.assertEqual(
            render_c1f_formula(parsed),
            "forall x: exists=2 y: (f(y) = x & Q(y))",
        )

    def test_rejects_unguarded_two_variable_formulas(self):
        with self.assertRaisesRegex(ValueError, "at most one free variable"):
            parse_c1f_sentence("forall x: exists y: R(x,y)")
        with self.assertRaisesRegex(ValueError, "at most one free variable"):
            parse_c1f_sentence(
                "forall x: exists y: (f(y) = x & R(x,y))"
            )
        with self.assertRaisesRegex(ValueError, "ordinary quantifiers"):
            parse_c1f_sentence(
                "forall x: exists>=1 y: "
                "(f(y) = x & exists z: P(z))"
            )

    def test_guarded_surjectivity_counts_permutations(self):
        sentence = "forall x: exists>=1 y: (f(y) = x & TRUE)"
        compiled = compile_c1f_fomc(sentence)
        expected = [1, 1, 2, 6, 24, 120]
        self.assertEqual(
            [int(compiled.fomc(n)) for n in range(len(expected))],
            expected,
        )
        self.assertEqual(compiled.profile_statistics()[0].guarded_types, 2)

    def test_guarded_surjectivity_unlabeled_counts_cycle_partitions(self):
        sentence = "forall x: exists>=1 y: (f(y) = x & TRUE)"
        expected = [1, 1, 2, 3, 5, 7, 11]
        self.assertEqual(
            [int(c1f_ufomc(sentence, n)) for n in range(len(expected))],
            expected,
        )

    def test_guarded_colours_and_nested_counts_match_bruteforce(self):
        sentences = [
            (
                "forall x: (P(x) -> "
                "exists=2 y: (f(y) = x & Q(y)))"
            ),
            (
                "forall x: (P(x) <-> exists>=1 y: (f(y) = x & "
                "exists=1 z: (f(z) = y & P(z))))"
            ),
            (
                "exists=1 x: (P(x) & "
                "exists>=1 y: (f(y) = x & Q(y)))"
            ),
        ]
        for sentence in sentences:
            labeled = compile_c1f_fomc(sentence)
            unlabeled = compile_c1f_ufomc(sentence)
            for n in range(4):
                with self.subTest(sentence=sentence, n=n):
                    self.assertEqual(
                        int(labeled.fomc(n)),
                        _bruteforce_c1f_model_count(sentence, n),
                    )
                    self.assertEqual(
                        int(unlabeled.ufomc(n)),
                        _bruteforce_c1f_unlabeled_count(sentence, n),
                    )

    def test_guarded_higher_arity_trace_compression_matches_bruteforce(self):
        sentence = (
            "forall x: exists>=1 y: "
            "(f(y) = x & R(y,f(y)))"
        )
        compiled = compile_c1f_fomc(sentence)
        for n in range(3):
            with self.subTest(n=n):
                self.assertEqual(
                    int(compiled.fomc(n)),
                    _bruteforce_c1f_model_count(sentence, n),
                )

    def test_guarded_optimizations_preserve_counts(self):
        sentence = (
            "forall x: (P(x) <-> "
            "exists>=1 y: (f(y) = x & P(y)))"
        )
        optimized = compile_c1f_fomc(sentence)
        raw = compile_c1f_fomc(
            sentence,
            profile_pruning=False,
            twin_compression=False,
        )
        for n in range(5):
            with self.subTest(n=n):
                self.assertEqual(int(optimized.fomc(n)), int(raw.fomc(n)))

    def test_true_counts_all_endofunctions(self):
        for n in range(5):
            with self.subTest(n=n):
                self.assertEqual(int(c1f_fomc("TRUE", n)), n**n if n else 1)

    def test_forall_fixed_point_counts_identity_function(self):
        sentence = "forall x: x = f(x)"
        for n in range(5):
            with self.subTest(n=n):
                self.assertEqual(int(c1f_fomc(sentence, n)), 1)

    def test_unary_tautology_keeps_function_symbol_in_vocabulary(self):
        sentence = "forall x: (P(x) | ~P(x))"
        for n in range(5):
            with self.subTest(n=n):
                expected = (n**n if n else 1) * (2**n)
                self.assertEqual(int(c1f_fomc(sentence, n)), expected)

    def test_prescribed_single_three_cycle_matches_bruteforce(self):
        sentence = "exists=3 x: (f(f(f(x))) = x & f(x) != x)"
        compiled = compile_c1f_fomc(sentence)
        for n in range(5):
            with self.subTest(n=n):
                self.assertEqual(int(compiled.fomc(n)), _bruteforce_c1f_model_count(sentence, n))

    def test_quasi_kernel_with_forbidden_short_cycles_matches_bruteforce(self):
        sentence = (
            "exists=1 x: Q(x) & "
            "forall x: (Q(x) -> ~Q(f(x))) & "
            "forall x: (~Q(x) -> (Q(f(x)) | Q(f(f(x))))) & "
            "forall x: f(f(x)) != x & "
            "forall x: f(f(f(x))) != x"
        )
        compiled = compile_c1f_fomc(sentence)
        for n in range(4):
            with self.subTest(n=n):
                self.assertEqual(int(compiled.fomc(n)), _bruteforce_c1f_model_count(sentence, n))

    def test_bounded_height_colour_family_matches_bruteforce(self):
        sentence = (
            "forall x: f(f(f(f(f(x))))) = f(f(f(f(x)))) & "
            "forall x: (f(f(x)) != f(x) -> "
            "(B(x) <-> (B(f(x)) | B(f(f(x))))))"
        )
        compiled = compile_c1f_fomc(sentence)
        for n in range(4):
            with self.subTest(n=n):
                expected = _bruteforce_c1f_model_count(sentence, n)
                self.assertEqual(int(compiled.fomc(n)), expected)
        self.assertEqual(int(compiled.fomc(4)), 1184)

    def test_profile_pruning_and_twin_compression_preserve_quasi_kernel_count(self):
        sentence = (
            "exists=1 x: Q(x) & "
            "forall x: (Q(x) -> ~Q(f(x))) & "
            "forall x: (~Q(x) -> (Q(f(x)) | Q(f(f(x))))) & "
            "forall x: f(f(x)) != x & "
            "forall x: f(f(f(x))) != x"
        )
        optimized = compile_c1f_fomc(sentence)
        raw = compile_c1f_fomc(
            sentence,
            profile_pruning=False,
            twin_compression=False,
        )
        stats = optimized.profile_statistics()[0]
        self.assertEqual(stats.raw_profiles, 50)
        self.assertEqual(stats.pruned_profiles, 22)
        self.assertEqual(stats.profile_classes, 17)
        for n in range(4):
            with self.subTest(n=n):
                self.assertEqual(int(optimized.fomc(n)), int(raw.fomc(n)))

    def test_direct_profile_counts_match_closed_formula(self):
        for depth in range(6):
            term = _iterated_function_term(depth)
            equality_only = compile_c1f_fomc(
                f"forall x: {term} = {term}",
                profile_pruning=False,
                twin_compression=False,
            ).profile_statistics()[0]
            self.assertEqual(
                equality_only.raw_profiles,
                1 + depth * (depth + 1) // 2,
            )

            one_predicate = compile_c1f_fomc(
                f"forall x: (P({term}) | ~P({term}))",
                profile_pruning=False,
                twin_compression=False,
            ).profile_statistics()[0]
            expected = 2 ** (depth + 1) + sum(
                first_repeat * 2**first_repeat
                for first_repeat in range(1, depth + 1)
            )
            self.assertEqual(one_predicate.raw_profiles, expected)

    def test_deep_stabilization_avoids_partition_enumeration(self):
        stabilization_depth = 12
        sentence = (
            "exists=1 x: x = f(x) & "
            f"forall x: {_iterated_function_term(stabilization_depth)} = "
            f"{_iterated_function_term(stabilization_depth + 1)}"
        )
        stats = compile_c1f_fomc(
            sentence,
            twin_compression=False,
        ).profile_statistics()[0]
        depth = stabilization_depth + 1
        self.assertEqual(stats.raw_profiles, 1 + depth * (depth + 1) // 2)
        self.assertEqual(stats.pruned_profiles, depth)
        self.assertEqual(stats.profile_classes, depth)

    def test_twin_compression_preserves_tautological_depth_count(self):
        sentence = "forall x: f(f(x)) = f(f(x))"
        optimized = compile_c1f_fomc(sentence)
        raw = compile_c1f_fomc(
            sentence,
            profile_pruning=False,
            twin_compression=False,
        )
        stats = optimized.profile_statistics()[0]
        self.assertEqual(stats.raw_profiles, stats.pruned_profiles)
        self.assertEqual(stats.profile_classes, stats.raw_profiles)
        for n in range(5):
            with self.subTest(n=n):
                self.assertEqual(int(optimized.fomc(n)), int(raw.fomc(n)))
                self.assertEqual(int(optimized.fomc(n)), n**n if n else 1)

    def test_zero_witness_count_does_not_become_a_profile_filter(self):
        sentence = "exists=0 x: (P(x) & forall x: f(x) = x)"
        optimized = compile_c1f_fomc(sentence)
        raw = compile_c1f_fomc(
            sentence,
            profile_pruning=False,
            twin_compression=False,
        )
        self.assertEqual(optimized.profile_statistics()[0].pruning_filters, 0)
        for n in range(4):
            with self.subTest(n=n):
                self.assertEqual(int(optimized.fomc(n)), int(raw.fomc(n)))

    def test_nullary_predicate_uses_shannon_expansion(self):
        for n in range(5):
            with self.subTest(n=n):
                endofunctions = n**n if n else 1
                self.assertEqual(int(c1f_fomc("Z", n)), endofunctions)
                self.assertEqual(int(c1f_fomc("Z | ~Z", n)), 2 * endofunctions)

    def test_binary_orbit_trace_compression_matches_bruteforce(self):
        sentences = [
            "forall x: (R(x,f(x)) <-> R(f(x),x))",
            "exists=1 x: R(x,f(x))",
            "forall x: (R(x,f(x)) -> R(f(x),f(f(x))))",
        ]
        for sentence in sentences:
            compiled = compile_c1f_fomc(sentence)
            for n in range(3):
                with self.subTest(sentence=sentence, n=n):
                    self.assertEqual(
                        int(compiled.fomc(n)),
                        _bruteforce_c1f_model_count(sentence, n),
                    )

    def test_all_quantifiers_match_bruteforce(self):
        sentences = [
            "forall x: (P(f(x)) -> f(f(x)) != x)",
            "exists x: (P(x) & x = f(x))",
            "exists=1 x: (P(x) | Q(f(x)))",
            "exists<2 x: (P(x) & f(x) != x)",
            "exists<=1 x: (P(f(x)) <-> Q(x))",
            "exists>1 x: (P(x) & Q(f(x)))",
            "exists>=2 x: (P(f(f(x))) | x = f(x))",
            "(exists=1 x: P(x)) | (forall x: Q(f(x)))",
            "forall x: (P(x) | (exists=1 x: Q(x)))",
            "exists x: (P(x) & (forall x: (Q(x) -> x = f(x))))",
        ]
        for sentence in sentences:
            compiled = compile_c1f_fomc(sentence)
            for n in range(4):
                with self.subTest(sentence=sentence, n=n):
                    actual = int(compiled.fomc(n))
                    brute_force = _bruteforce_c1f_model_count(sentence, n)
                    self.assertEqual(actual, brute_force)

    def test_nullary_and_higher_arity_compression_matches_bruteforce(self):
        sentences = [
            "Z & (exists=1 x: P(x))",
            "forall x: (R(x,f(x)) <-> R(f(x),x))",
        ]
        for sentence in sentences:
            compiled = compile_c1f_fomc(sentence)
            for n in range(3):
                with self.subTest(sentence=sentence, n=n):
                    self.assertEqual(
                        int(compiled.fomc(n)),
                        _bruteforce_c1f_model_count(sentence, n),
                    )

    def test_unlabeled_endofunction_sequence(self):
        expected = [1, 1, 3, 7, 19, 47, 130, 343, 951, 2615, 7318]
        self.assertEqual(
            [int(c1f_ufomc("TRUE", n)) for n in range(len(expected))],
            expected,
        )
        for n in range(6):
            with self.subTest(bruteforce_n=n):
                self.assertEqual(
                    expected[n],
                    _bruteforce_c1f_unlabeled_count("TRUE", n),
                )

    def test_unlabeled_monadic_formulas_match_literal_isomorphism_classes(self):
        sentences = [
            "forall x: x = f(x)",
            "forall x: (P(x) | ~P(x))",
            "exists=1 x: (P(x) & x = f(x))",
            "exists<2 x: (P(x) & f(x) != x)",
            "exists<=1 x: (P(f(x)) <-> Q(x))",
            "exists>1 x: (P(x) & Q(f(x)))",
            "exists>=2 x: (P(f(f(x))) | x = f(x))",
            "forall x: f(f(x)) != x",
            "exists=3 x: (f(f(f(x))) = x & f(x) != x)",
            "forall x: (P(x) -> P(f(x)))",
            "(exists=1 x: P(x)) | (forall x: f(x) = x)",
        ]
        for sentence in sentences:
            compiled = compile_c1f_ufomc(sentence)
            for n in range(5):
                with self.subTest(sentence=sentence, n=n):
                    self.assertEqual(
                        int(compiled.ufomc(n)),
                        _bruteforce_c1f_unlabeled_count(sentence, n),
                    )

    def test_unlabeled_identity_with_one_unary_colour(self):
        sentence = (
            "forall x: f(x) = x & "
            "forall x: (P(x) | ~P(x))"
        )
        for n in range(8):
            with self.subTest(n=n):
                self.assertEqual(int(c1f_ufomc(sentence, n)), n + 1)

    def test_unlabeled_nullary_predicate_uses_disjoint_branches(self):
        mapping_counts = [1, 1, 3, 7, 19, 47, 130]
        for n, mapping_count in enumerate(mapping_counts):
            with self.subTest(n=n):
                self.assertEqual(
                    int(c1f_ufomc("Z | ~Z", n)),
                    2 * mapping_count,
                )

    def test_unlabeled_twin_compression_preserves_count(self):
        sentence = (
            "forall x: f(f(f(f(f(x))))) = f(f(f(f(x)))) & "
            "forall x: (f(f(x)) != f(x) -> "
            "(B(x) <-> (B(f(x)) | B(f(f(x))))))"
        )
        optimized = compile_c1f_ufomc(sentence)
        uncompressed = compile_c1f_ufomc(sentence, twin_compression=False)
        self.assertLess(
            optimized.profile_statistics()[0].profile_classes,
            uncompressed.profile_statistics()[0].profile_classes,
        )
        for n in range(7):
            with self.subTest(n=n):
                self.assertEqual(
                    int(optimized.ufomc(n)),
                    int(uncompressed.ufomc(n)),
                )
                if n <= 4:
                    self.assertEqual(
                        int(optimized.ufomc(n)),
                        _bruteforce_c1f_unlabeled_count(sentence, n),
                    )

    def test_exact_unlabeled_count_rejects_higher_arity_relations(self):
        sentence = "forall x: (R(x,f(x)) | ~R(x,f(x)))"
        with self.assertRaisesRegex(ValueError, "only nullary and unary"):
            compile_c1f_ufomc(sentence)
        compiled = compile_c1f_fomc(sentence)
        with self.assertRaisesRegex(ValueError, "only nullary and unary"):
            compiled.ufomc(3)


if __name__ == "__main__":
    unittest.main()
