from __future__ import annotations

from decimal import Decimal, localcontext
from importlib.util import find_spec
from math import e, exp, factorial, lgamma, log, pi, sqrt
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
    C1FAsymptoticCertificate,
    analyze_c1f_labeled_asymptotics,
    c1f_fomc,
    c1f_labeled_asymptotics,
    c1f_unlabeled_asymptotics,
    compile_c1f_fomc,
    compile_c1f_labeled_specification,
)
from unaryfomc.specification import C1FLabeledSpecification
from unaryfomc.puiseux import project_complex_laurent_orbit


class C1FLabeledSpecificationTests(unittest.TestCase):
    def test_unrestricted_endofunction_specification(self) -> None:
        specification = compile_c1f_labeled_specification("TRUE")

        self.assertEqual(specification.tree_dimension, 1)
        self.assertEqual(specification.dependency_matrix(), ((True,),))
        self.assertAlmostEqual(specification.tree_rhs(0.1, (0.2,))[0], 0.1 * exp(0.2))
        self.assertAlmostEqual(specification.tree_jacobian(0.1, (0.2,))[0][0], 0.1 * exp(0.2))
        self.assertAlmostEqual(specification.tree_x_derivative(0.1, (0.2,))[0], exp(0.2))
        self.assertAlmostEqual(
            specification.tree_hessian_direction(0.1, (0.2,), (3.0,))[0],
            0.9 * exp(0.2),
        )
        self.assertAlmostEqual(specification.finite_cycle_correction(0.1, (0.2,)), 0.0)
        self.assertAlmostEqual(
            specification.long_cycle_determinant(0.1, (0.2,)),
            1.0 - 0.1 * exp(0.2),
        )

    def test_depth_one_short_cycle_correction_cancels_for_tautology(self) -> None:
        specification = compile_c1f_labeled_specification(
            "forall x: f(x) = f(x)"
        )

        self.assertEqual(specification.depth, 1)
        self.assertEqual(specification.tree_dimension, 1)
        self.assertAlmostEqual(
            specification.finite_cycle_correction(1.0 / e, (1.0,)),
            0.0,
        )

    def test_guarded_projection_derives_tree_and_cycle_derivatives(self) -> None:
        specification = compile_c1f_labeled_specification(
            "forall x: exists<=2 y: (f(y) = x & TRUE)"
        )

        self.assertTrue(specification.has_guarded_predecessors)
        self.assertEqual(specification.local_caps, (3,))
        self.assertEqual(specification.tree_dimension, 1)
        self.assertAlmostEqual(
            specification.tree_rhs(0.2, (0.3,))[0],
            0.2 * (1.0 + 0.3 + 0.3**2 / 2.0),
        )
        self.assertAlmostEqual(
            specification.tree_jacobian(0.2, (0.3,))[0][0],
            0.2 * (1.0 + 0.3),
        )
        self.assertAlmostEqual(
            specification.tree_hessian_direction(0.2, (0.3,), (1.0,))[0],
            0.2,
        )
        self.assertAlmostEqual(
            specification.long_cycle_matrix(0.2, (0.3,))[0][0],
            0.2 * (1.0 + 0.3),
        )
        self.assertEqual(
            specification.jacobian_transfer_residual(0.2, (0.3,)),
            0.0,
        )

    def test_guarded_surjectivity_has_cycle_only_type(self) -> None:
        specification = compile_c1f_labeled_specification(
            "forall x: exists>=1 y: (f(y) = x & TRUE)"
        )

        self.assertEqual(specification.tree_dimension, 0)
        self.assertEqual(specification.cycle_dimension, 1)
        self.assertEqual(specification.long_cycle_matrix(0.25, tuple()), ((0.25,),))


class C1FLabeledAsymptoticTests(unittest.TestCase):
    def test_arbitrary_order_labeled_expansion_matches_stirling_series(self) -> None:
        analysis = c1f_labeled_asymptotics("TRUE", expansion_order=6)
        terms = analysis.residue_expansions[0]
        leading = 1.0 / sqrt(2.0 * pi)

        self.assertEqual(
            tuple(term.power for term in terms),
            (-0.5, -1.5, -2.5, -3.5),
        )
        self.assertAlmostEqual(terms[0].constant, leading)
        self.assertAlmostEqual(terms[1].constant, -leading / 12.0)
        self.assertAlmostEqual(terms[2].constant, leading / 288.0)
        self.assertAlmostEqual(terms[3].constant, 139.0 * leading / 51840.0)
        exact = float(20**20)
        self.assertLess(
            abs(analysis.expansion_estimate(20) / exact - 1.0),
            abs(analysis.estimate(20) / exact - 1.0),
        )

    def test_invalid_expansion_orders_and_fixed_marker_lifts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            c1f_labeled_asymptotics("TRUE", expansion_order=-1)
        with self.assertRaisesRegex(ValueError, "bivariate marker recurrence"):
            c1f_labeled_asymptotics(
                "exists=1 x: P(x)",
                expansion_order=2,
            )
        with self.assertRaisesRegex(ValueError, "bivariate marker recurrence"):
            c1f_unlabeled_asymptotics(
                "exists=1 x: P(x)",
                expansion_order=2,
            )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_arbitrary_order_labeled_coefficients_are_interval_certified(self) -> None:
        analysis = c1f_labeled_asymptotics(
            "TRUE",
            expansion_order=6,
            certify=True,
            precision_bits=96,
        )

        assert analysis.certificate is not None
        self.assertEqual(len(analysis.certificate.residue_expansions[0]), 4)
        self.assertIn(
            "recursive interval Laurent-Puiseux expansion",
            analysis.certificate.verified_properties,
        )

    def test_fourier_projection_continues_after_leading_cancellation(self) -> None:
        projected = project_complex_laurent_orbit(
            (
                {-1: 1 + 0j, 1: 1 + 0j},
                {-1: 1 + 0j, 1: -1 + 0j},
            ),
            (1 + 0j, -1 + 0j),
            period=2,
            admitted_residues=(0, 1),
            leading_power=-0.5,
            half_order=2,
            tolerance=1e-12,
        )

        self.assertEqual(projected[0][0][0], -0.5)
        self.assertEqual(projected[1][0][0], -1.5)

    def test_reducible_regular_system_with_one_critical_component(self) -> None:
        specification = C1FLabeledSpecification(
            depth=0,
            successor=((True, False), (False, False)),
            multiplicities=(1, 1),
            visible_cycle_lengths=(None, None),
            transient_profiles=(0, 1),
        )
        analysis = analyze_c1f_labeled_asymptotics(specification)

        self.assertEqual(analysis.critical_component, (0,))
        self.assertAlmostEqual(analysis.radius, 1.0 / e)
        self.assertAlmostEqual(analysis.tree_values[0], 1.0)
        self.assertAlmostEqual(analysis.tree_values[1], 1.0 / e)
        self.assertEqual(analysis.perron_right, (1.0, 0.0))
        self.assertEqual(analysis.perron_left, (1.0, 0.0))
        self.assertAlmostEqual(analysis.leading_constant, 1.0 / sqrt(2.0 * pi))

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_interval_certificate_handles_reducible_regular_system(self) -> None:
        specification = C1FLabeledSpecification(
            depth=0,
            successor=((True, False), (False, False)),
            multiplicities=(1, 1),
            visible_cycle_lengths=(None, None),
            transient_profiles=(0, 1),
        )
        analysis = analyze_c1f_labeled_asymptotics(
            specification,
            certify=True,
            precision_bits=128,
        )
        certificate = analysis.certificate

        assert certificate is not None
        self.assertEqual(len(certificate.tree_values), 2)
        self.assertEqual(len(certificate.perron_right), 2)
        self.assertEqual(len(certificate.perron_left), 2)
        self.assertLessEqual(
            Decimal(certificate.perron_right[1].lower),
            Decimal(0),
        )
        self.assertGreaterEqual(
            Decimal(certificate.perron_right[1].upper),
            Decimal(0),
        )

    def test_periodic_cycle_pole_is_reported_by_residue_class(self) -> None:
        specification = C1FLabeledSpecification(
            depth=0,
            successor=((False, True), (True, False)),
            multiplicities=(1, 1),
            visible_cycle_lengths=(None, None),
            transient_profiles=tuple(),
            long_cycle_profiles=(0, 1),
        )
        analysis = analyze_c1f_labeled_asymptotics(specification)

        self.assertEqual(analysis.periodicity, 2)
        self.assertEqual(analysis.residue_powers, (0.0, None))
        self.assertAlmostEqual(analysis.residue_leading_constants[0], 1.0)
        self.assertEqual(analysis.estimate(9), 0.0)
        self.assertAlmostEqual(analysis.estimate(10), float(factorial(10)))

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_analytic_tree_cycle_pole_is_interval_certified(self) -> None:
        specification = C1FLabeledSpecification(
            depth=0,
            successor=((True, True), (False, True)),
            multiplicities=(1, 2),
            visible_cycle_lengths=(None, None),
            transient_profiles=(0,),
            long_cycle_profiles=(0, 1),
        )
        analysis = analyze_c1f_labeled_asymptotics(
            specification,
            certify=True,
            precision_bits=96,
        )

        self.assertEqual(analysis.regime, "analytic-tree-cycle-pole")
        self.assertAlmostEqual(analysis.radius, exp(-0.5) / 2.0)
        self.assertAlmostEqual(analysis.leading_constant, 1.0, places=7)
        assert analysis.certificate is not None
        self.assertEqual(
            analysis.certificate.method,
            "arb-krawczyk-analytic-tree-cycle-pole",
        )
        self.assertLessEqual(
            Decimal(analysis.certificate.leading_constant.lower),
            Decimal(1),
        )
        self.assertGreaterEqual(
            Decimal(analysis.certificate.leading_constant.upper),
            Decimal(1),
        )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_interval_certificate_handles_periodic_cycle_pole(self) -> None:
        specification = C1FLabeledSpecification(
            depth=0,
            successor=((False, True), (True, False)),
            multiplicities=(1, 1),
            visible_cycle_lengths=(None, None),
            transient_profiles=tuple(),
            long_cycle_profiles=(0, 1),
        )
        analysis = analyze_c1f_labeled_asymptotics(
            specification,
            certify=True,
            precision_bits=128,
        )
        certificate = analysis.certificate

        assert certificate is not None
        self.assertTrue(certificate.dominance_certified)
        self.assertEqual(certificate.residue_powers, (0.0, None))
        self.assertIsNone(certificate.residue_leading_constants[1])

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_interval_certificate_handles_multiple_cycle_pole(self) -> None:
        specification = C1FLabeledSpecification(
            depth=0,
            successor=((True, False), (False, True)),
            multiplicities=(1, 1),
            visible_cycle_lengths=(None, None),
            transient_profiles=tuple(),
            long_cycle_profiles=(0, 1),
        )
        analysis = analyze_c1f_labeled_asymptotics(
            specification,
            certify=True,
            precision_bits=128,
        )

        self.assertEqual(analysis.cycle_pole_order, 2)
        self.assertEqual(analysis.power, 1.0)
        assert analysis.certificate is not None
        self.assertTrue(analysis.certificate.dominance_certified)

    def test_public_direct_and_compiled_entry_points_agree(self) -> None:
        sentence = "forall x: exists<=2 y: (f(y) = x & TRUE)"
        direct = c1f_labeled_asymptotics(sentence)
        compiled = compile_c1f_fomc(sentence)
        specification = compiled.labeled_specification()
        from_compiled = compiled.labeled_asymptotics()

        self.assertTrue(specification.has_guarded_predecessors)
        self.assertEqual(direct.regime, "guarded-tree-critical")
        self.assertEqual(from_compiled.regime, direct.regime)
        self.assertAlmostEqual(from_compiled.radius, direct.radius)
        self.assertAlmostEqual(
            from_compiled.leading_constant,
            direct.leading_constant,
        )
        self.assertIsNone(direct.certificate)

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_interval_certificate_encloses_guarded_scalar_constants(self) -> None:
        sentence = "forall x: exists<=2 y: (f(y) = x & TRUE)"
        result = c1f_labeled_asymptotics(
            sentence,
            certify=True,
            precision_bits=128,
        )
        certificate = result.certificate

        self.assertIsInstance(certificate, C1FAsymptoticCertificate)
        assert certificate is not None
        with localcontext() as context:
            context.prec = 60
            expected_radius = Decimal(2).sqrt() - 1
            self.assertLess(Decimal(certificate.radius.lower), expected_radius)
            self.assertGreater(Decimal(certificate.radius.upper), expected_radius)
            self.assertLess(
                Decimal(certificate.radius.upper)
                - Decimal(certificate.radius.lower),
                Decimal("1e-18"),
            )
        self.assertGreater(Decimal(certificate.leading_constant.lower), 0)
        self.assertEqual(certificate.method, "arb-krawczyk")
        self.assertTrue(certificate.dominance_certified)
        self.assertIn(
            "phase rigidity excludes nonpositive boundary singularities",
            certificate.verified_properties,
        )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_interval_certificate_handles_guarded_multitype_system(self) -> None:
        result = c1f_labeled_asymptotics(
            "forall x: (P(x) -> "
            "exists=2 y: (f(y) = x & Q(y)))",
            certify=True,
            precision_bits=128,
        )
        certificate = result.certificate

        assert certificate is not None
        self.assertEqual(len(certificate.tree_values), 6)
        self.assertEqual(len(certificate.perron_right), 6)
        self.assertEqual(len(certificate.perron_left), 6)
        self.assertLess(
            Decimal(certificate.leading_constant.upper)
            - Decimal(certificate.leading_constant.lower),
            Decimal("1e-15"),
        )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_interval_certificate_handles_simple_cycle_pole(self) -> None:
        compiled = compile_c1f_fomc(
            "forall x: exists>=1 y: (f(y) = x & TRUE)"
        )
        result = compiled.labeled_asymptotics(
            certify=True,
            precision_bits=128,
        )
        certificate = result.certificate

        assert certificate is not None
        self.assertEqual(result.regime, "cycle-pole")
        self.assertLessEqual(Decimal(certificate.radius.lower), Decimal(1))
        self.assertGreaterEqual(Decimal(certificate.radius.upper), Decimal(1))
        self.assertLessEqual(
            Decimal(certificate.leading_constant.lower),
            Decimal(1),
        )
        self.assertGreaterEqual(
            Decimal(certificate.leading_constant.upper),
            Decimal(1),
        )

    def test_unrestricted_endofunction_leading_term(self) -> None:
        analysis = analyze_c1f_labeled_asymptotics("TRUE")

        self.assertTrue(analysis.structurally_verified)
        self.assertEqual(analysis.regime, "scalar-tree-critical")
        self.assertAlmostEqual(analysis.radius, 1.0 / e)
        self.assertAlmostEqual(analysis.exponential_growth, e)
        self.assertAlmostEqual(analysis.power, -0.5)
        self.assertAlmostEqual(analysis.leading_constant, 1.0 / sqrt(2.0 * pi))
        self.assertEqual(analysis.tree_values, (1.0,))
        self.assertAlmostEqual(analysis.tree_sqrt_coefficients[0], sqrt(2.0))
        self.assertAlmostEqual(analysis.denominator_sqrt_coefficient, sqrt(2.0))

    def test_leading_term_converges_to_n_to_n(self) -> None:
        analysis = analyze_c1f_labeled_asymptotics("TRUE")
        domain_size = 50
        exact = float(domain_size**domain_size)

        self.assertLess(abs(analysis.estimate(domain_size) / exact - 1.0), 0.002)

    def test_eventually_true_sentence_has_same_asymptotics(self) -> None:
        analysis = analyze_c1f_labeled_asymptotics("exists x: TRUE")

        self.assertEqual(analysis.eventual_start, 1)
        self.assertAlmostEqual(analysis.radius, 1.0 / e)
        self.assertAlmostEqual(analysis.leading_constant, 1.0 / sqrt(2.0 * pi))

    def test_single_unsaturated_global_counter(self) -> None:
        analysis = c1f_labeled_asymptotics("exists=2 x: P(x)")

        self.assertTrue(analysis.regime.startswith("capped-global-face:"))
        self.assertEqual(analysis.accepted_global_states, ((2,),))
        self.assertAlmostEqual(analysis.marker_growth_derivatives[0], 1.0, places=6)
        self.assertEqual(analysis.power, 1.5)
        self.assertAlmostEqual(
            analysis.leading_constant,
            1.0 / (2.0 * sqrt(2.0 * pi)),
            places=7,
        )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_running_example_global_counter_is_interval_certified(self) -> None:
        sentence = (
            "(forall x: (f(x) = x -> P(x))) & "
            "(forall x: ((f(f(x)) = x & f(x) != x) -> "
            "~(P(x) <-> P(f(x))))) & "
            "(exists=2 x: P(x))"
        )
        analysis = c1f_labeled_asymptotics(
            sentence,
            certify=True,
            precision_bits=128,
        )
        expected = Decimal("0.0445080274579757360461030487")

        self.assertEqual(analysis.power, 1.5)
        assert analysis.certificate is not None
        self.assertEqual(
            analysis.certificate.method,
            "arb-krawczyk-marker-face",
        )
        self.assertLess(
            Decimal(analysis.certificate.leading_constant.lower),
            expected,
        )
        self.assertGreater(
            Decimal(analysis.certificate.leading_constant.upper),
            expected,
        )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_multiple_exact_global_counters_are_interval_certified(self) -> None:
        analysis = c1f_labeled_asymptotics(
            "(exists=1 x: P(x)) & (exists=2 x: Q(x))",
            certify=True,
            precision_bits=96,
        )
        expected = Decimal("0.19947114020071634")

        self.assertEqual(analysis.accepted_global_states, ((1, 2),))
        self.assertEqual(analysis.power, 2.5)
        self.assertEqual(len(analysis.marker_growth_derivatives), 2)
        assert analysis.certificate is not None
        self.assertEqual(
            analysis.certificate.method,
            "arb-krawczyk-marker-face",
        )
        self.assertLess(
            Decimal(analysis.certificate.leading_constant.lower),
            expected,
        )
        self.assertGreater(
            Decimal(analysis.certificate.leading_constant.upper),
            expected,
        )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_mixed_saturated_global_face_uses_certified_smaller_radius(self) -> None:
        analysis = c1f_labeled_asymptotics(
            "(exists>=1 x: P(x)) & (exists=2 x: Q(x))",
            certify=True,
            precision_bits=96,
        )

        self.assertEqual(analysis.regime, "capped-global-face-sum")
        self.assertAlmostEqual(analysis.radius, 1.0 / (2.0 * e))
        self.assertEqual(analysis.power, 1.5)
        self.assertEqual(len(analysis.summands), 2)
        dominant = min(analysis.summands, key=lambda summand: summand.radius)
        assert dominant.certificate is not None
        self.assertEqual(
            dominant.certificate.method,
            "arb-krawczyk-marker-face",
        )
        self.assertIn(
            "certified saturated-face decomposition",
            dominant.certificate.verified_properties,
        )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_exact_marker_over_simple_cycle_pole_is_interval_certified(self) -> None:
        sentence = (
            "(forall x: exists>=1 y: (f(y) = x & TRUE)) & "
            "(exists=2 x: P(x))"
        )
        analysis = c1f_labeled_asymptotics(
            sentence,
            certify=True,
            precision_bits=96,
        )

        self.assertEqual(analysis.regime, "capped-global-face:cycle-pole")
        self.assertEqual(analysis.power, 2.0)
        self.assertAlmostEqual(analysis.leading_constant, 0.5, places=7)
        assert analysis.certificate is not None
        self.assertLess(
            Decimal(analysis.certificate.leading_constant.lower),
            Decimal("0.5"),
        )
        self.assertGreater(
            Decimal(analysis.certificate.leading_constant.upper),
            Decimal("0.5"),
        )
        self.assertIn(
            "positive exact-matrix growth derivative",
            analysis.certificate.verified_properties,
        )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_guarded_running_example_cycle_pole_is_interval_certified(self) -> None:
        sentence = (
            "(forall x: (f(x) = x -> P(x))) & "
            "(forall x: ((f(f(x)) = x & f(x) != x) -> "
            "~(P(x) <-> P(f(x))))) & "
            "(exists=2 x: P(x)) & "
            "(forall x: exists>=1 y: (f(y) = x & TRUE))"
        )
        analysis = c1f_labeled_asymptotics(
            sentence,
            certify=True,
            precision_bits=96,
        )
        expected = Decimal("0.11156508007421491446")

        self.assertEqual(analysis.regime, "capped-global-face:cycle-pole")
        self.assertEqual(analysis.radius, 1.0)
        self.assertEqual(analysis.power, 2.0)
        assert analysis.certificate is not None
        self.assertLess(
            Decimal(analysis.certificate.leading_constant.lower),
            expected,
        )
        self.assertGreater(
            Decimal(analysis.certificate.leading_constant.upper),
            expected,
        )
        self.assertTrue(analysis.certificate.dominance_certified)

    def test_unconstrained_unary_colour_uses_rank_one_reduction(self) -> None:
        analysis = analyze_c1f_labeled_asymptotics(
            "forall x: (P(x) | ~P(x))"
        )
        domain_size = 50
        exact = float((2 * domain_size) ** domain_size)

        self.assertEqual(analysis.regime, "rank-one-tree-critical")
        self.assertAlmostEqual(analysis.radius, 1.0 / (2.0 * e))
        self.assertEqual(analysis.tree_values, (0.5, 0.5))
        self.assertAlmostEqual(sum(analysis.tree_sqrt_coefficients), sqrt(2.0))
        self.assertAlmostEqual(analysis.leading_constant, 1.0 / sqrt(2.0 * pi))
        self.assertLess(abs(analysis.estimate(domain_size) / exact - 1.0), 0.002)

    def test_higher_arity_labeled_and_unlabeled_transfers(self) -> None:
        sentence = "forall x: R(x,f(x))"
        labeled = c1f_labeled_asymptotics(sentence)
        unlabeled = c1f_unlabeled_asymptotics(sentence)
        domain_size = 25
        exact_log = (
            domain_size * log(domain_size)
            + (domain_size**2 - domain_size) * log(2.0)
        )

        self.assertEqual(labeled.higher_arity_exponents, (2,))
        self.assertEqual(labeled.trace_predicate_count, 1)
        self.assertLess(abs(labeled.log_estimate(domain_size) - exact_log), 0.01)
        self.assertAlmostEqual(
            unlabeled.log_estimate(domain_size),
            labeled.log_estimate(domain_size) - lgamma(domain_size + 1),
        )

    def test_unrestricted_monadic_unlabeled_polya_asymptotic(self) -> None:
        analysis = c1f_unlabeled_asymptotics("TRUE")
        exact = float(471559682209452602)

        self.assertEqual(
            analysis.regime,
            "subcritical-polya-complete-compatibility",
        )
        self.assertAlmostEqual(analysis.exponential_growth, 2.95576528565)
        self.assertLess(abs(analysis.estimate(40) / exact - 1.0), 0.004)

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_unrestricted_monadic_unlabeled_polya_certificate(self) -> None:
        analysis = c1f_unlabeled_asymptotics(
            "TRUE",
            certify=True,
            precision_bits=96,
        )
        certificate = analysis.certificate

        assert certificate is not None
        self.assertEqual(certificate.method, "arb-polya-tail-bisection")
        self.assertTrue(certificate.dominance_certified)
        expected_radius = Decimal("0.33832185689920769520")
        self.assertLessEqual(Decimal(certificate.radius.lower), expected_radius)
        self.assertGreaterEqual(Decimal(certificate.radius.upper), expected_radius)

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_fixed_point_free_unlabeled_polya_is_interval_certified(self) -> None:
        analysis = c1f_unlabeled_asymptotics(
            "forall x: f(x) != x",
            certify=True,
            precision_bits=96,
        )

        self.assertEqual(
            analysis.regime,
            "subcritical-polya-complete-compatibility",
        )
        self.assertAlmostEqual(analysis.radius, 0.3383218568992077)
        self.assertLess(
            abs(analysis.estimate(30) / 3533715961160.0 - 1.0),
            0.03,
        )
        assert analysis.certificate is not None
        self.assertTrue(analysis.certificate.dominance_certified)
        self.assertIn(
            "finite forbidden-cycle correction with an explicit tail bound",
            analysis.certificate.verified_properties,
        )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_scalar_guarded_unlabeled_polya_is_interval_certified(self) -> None:
        analysis = c1f_unlabeled_asymptotics(
            "forall x: exists<=2 y: (f(y) = x & TRUE)",
            certify=True,
            precision_bits=96,
            polya_tail_terms=32,
        )

        self.assertEqual(analysis.regime, "subcritical-polya-multitype")
        self.assertAlmostEqual(analysis.radius, 0.4026975036715953)
        assert analysis.certificate is not None
        self.assertEqual(
            analysis.certificate.method,
            "arb-polya-finite-hierarchy-krawczyk",
        )
        self.assertTrue(analysis.certificate.dominance_certified)

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_periodic_scalar_tree_projects_certified_root_of_unity_orbit(self) -> None:
        specification = C1FLabeledSpecification(
            depth=0,
            successor=((True,),),
            multiplicities=(2,),
            visible_cycle_lengths=(None,),
            transient_profiles=(0,),
            long_cycle_profiles=(0,),
            local_caps=(3,),
            local_contributions=((1,),),
            local_acceptance=(frozenset({(0,), (2,)}),),
        )
        labeled = analyze_c1f_labeled_asymptotics(
            specification,
            certify=True,
            precision_bits=96,
        )
        analysis = c1f_unlabeled_asymptotics(
            specification,
            certify=True,
            precision_bits=96,
            polya_tail_terms=24,
        )

        assert labeled.certificate is not None
        self.assertEqual(labeled.periodicity, 2)
        self.assertEqual(labeled.residue_powers, (-0.5, None))
        self.assertEqual(labeled.estimate(21), 0.0)
        self.assertTrue(labeled.certificate.dominance_certified)
        self.assertEqual(
            labeled.certificate.residue_powers,
            (-0.5, None),
        )
        self.assertIsNone(
            labeled.certificate.residue_leading_constants[1]
        )
        assert analysis.certificate is not None
        self.assertEqual(analysis.periodicity, 2)
        self.assertEqual(analysis.residue_powers, (-0.5, None))
        self.assertEqual(analysis.estimate(21), 0.0)
        self.assertTrue(analysis.certificate.dominance_certified)
        self.assertIn(
            "complete root-of-unity dominant orbit",
            analysis.certificate.verified_properties,
        )
        self.assertIsNone(
            analysis.certificate.residue_leading_constants[1]
        )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_period_three_implicit_tree_orbit_is_certified(self) -> None:
        sentence = (
            "forall x: ((exists=0 y: (f(y) = x & TRUE)) | "
            "(exists=3 y: (f(y) = x & TRUE)))"
        )

        labeled = analyze_c1f_labeled_asymptotics(
            sentence,
            certify=True,
            precision_bits=96,
        )
        unlabeled = c1f_unlabeled_asymptotics(
            sentence,
            certify=True,
            precision_bits=96,
            polya_tail_terms=24,
        )

        for analysis in (labeled, unlabeled):
            self.assertEqual(analysis.periodicity, 3)
            self.assertEqual(analysis.residue_powers, (-0.5, None, None))
            self.assertGreater(analysis.estimate(30), 0.0)
            self.assertEqual(analysis.estimate(31), 0.0)
            self.assertEqual(analysis.estimate(32), 0.0)
            assert analysis.certificate is not None
            self.assertTrue(analysis.certificate.dominance_certified)
            self.assertIn(
                "complete root-of-unity dominant orbit",
                analysis.certificate.verified_properties,
            )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_periodic_analytic_factor_populates_both_residue_classes(self) -> None:
        specification = C1FLabeledSpecification(
            depth=1,
            successor=((True, False), (False, True)),
            multiplicities=(2, 1),
            visible_cycle_lengths=(None, 1),
            transient_profiles=(0,),
            long_cycle_profiles=(0,),
            local_caps=(3,),
            local_contributions=((1,), (1,)),
            local_acceptance=(
                frozenset({(0,), (2,)}),
                frozenset({(1,)}),
            ),
        )

        labeled = analyze_c1f_labeled_asymptotics(
            specification,
            certify=True,
            precision_bits=96,
        )
        unlabeled = c1f_unlabeled_asymptotics(
            specification,
            certify=True,
            precision_bits=96,
            polya_tail_terms=24,
        )

        for analysis in (labeled, unlabeled):
            self.assertEqual(analysis.periodicity, 2)
            self.assertEqual(analysis.residue_powers, (-0.5, -0.5))
            self.assertGreater(analysis.residue_leading_constants[0], 0.0)
            self.assertGreater(analysis.residue_leading_constants[1], 0.0)
            self.assertNotAlmostEqual(
                analysis.residue_leading_constants[0],
                analysis.residue_leading_constants[1],
            )
            assert analysis.certificate is not None
            self.assertTrue(analysis.certificate.dominance_certified)
            self.assertIsNotNone(
                analysis.certificate.residue_leading_constants[0]
            )
            self.assertIsNotNone(
                analysis.certificate.residue_leading_constants[1]
            )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_multitype_unlabeled_polya_is_interval_certified(self) -> None:
        analysis = c1f_unlabeled_asymptotics(
            "forall x: ~(P(x) & P(f(x)))",
            certify=True,
            precision_bits=96,
            polya_tail_terms=24,
        )

        self.assertEqual(analysis.regime, "subcritical-polya-multitype")
        self.assertAlmostEqual(analysis.radius, 0.21187682302549055)
        self.assertEqual(len(analysis.tree_values), 3)
        assert analysis.certificate is not None
        self.assertEqual(
            analysis.certificate.method,
            "arb-polya-multitype-krawczyk",
        )
        self.assertIn(
            "positive full Perron directions and quadratic scale",
            analysis.certificate.verified_properties,
        )
        self.assertTrue(analysis.certificate.dominance_certified)
        self.assertIn(
            "exact span-one critical-tree coefficient witness",
            analysis.certificate.verified_properties,
        )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_subcritical_unlabeled_cycle_pole_is_interval_certified(self) -> None:
        sentence = (
            "(forall x: exists>=1 y: (f(y) = x & TRUE)) & "
            "(forall x: (P(x) | ~P(x)))"
        )
        analysis = c1f_unlabeled_asymptotics(
            sentence,
            certify=True,
            precision_bits=96,
            polya_tail_terms=48,
        )

        self.assertAlmostEqual(analysis.radius, 0.5)
        self.assertEqual(analysis.power, 0.0)
        assert analysis.certificate is not None
        self.assertEqual(
            analysis.certificate.method,
            "arb-polya-exact-cycle-pole",
        )
        self.assertTrue(analysis.certificate.dominance_certified)
        self.assertLess(
            abs(analysis.estimate(40) / 3807299401210.0 - 1.0),
            0.0001,
        )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_periodic_unlabeled_cycle_pole_projects_exact_support(self) -> None:
        sentence = (
            "(forall x: exists>=1 y: (f(y) = x & TRUE)) & "
            "(forall x: (P(f(x)) <-> ~P(x))) & "
            "(forall x: (Q(x) | ~Q(x)))"
        )
        analysis = c1f_unlabeled_asymptotics(
            sentence,
            certify=True,
            precision_bits=96,
            polya_tail_terms=32,
        )

        self.assertEqual(analysis.periodicity, 2)
        self.assertEqual(analysis.residue_powers, (0.0, None))
        self.assertEqual(analysis.estimate(21), 0.0)
        self.assertGreater(analysis.estimate(20), 0.0)
        self.assertLess(abs(analysis.estimate(12) / 5876.0 - 1.0), 0.02)
        assert analysis.certificate is not None
        self.assertTrue(analysis.certificate.dominance_certified)
        self.assertEqual(analysis.certificate.residue_powers, (0.0, None))
        self.assertIsNone(analysis.certificate.residue_leading_constants[1])

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_unlabeled_analytic_tree_cycle_pole_is_interval_certified(self) -> None:
        specification = C1FLabeledSpecification(
            depth=0,
            successor=((True, True), (False, True)),
            multiplicities=(1, 2),
            visible_cycle_lengths=(None, None),
            transient_profiles=(0,),
            long_cycle_profiles=(0, 1),
        )
        analysis = c1f_unlabeled_asymptotics(
            specification,
            certify=True,
            precision_bits=96,
            polya_tail_terms=32,
        )

        self.assertEqual(analysis.power, 0.0)
        self.assertAlmostEqual(analysis.radius, 0.28689373816629005)
        self.assertAlmostEqual(analysis.leading_constant, 1.3111305625, places=7)
        assert analysis.certificate is not None
        self.assertEqual(
            analysis.certificate.method,
            "arb-polya-analytic-tree-cycle-pole",
        )
        self.assertGreater(
            Decimal(analysis.certificate.leading_constant.lower),
            Decimal("1.31"),
        )
        self.assertLess(
            Decimal(analysis.certificate.leading_constant.upper),
            Decimal("1.32"),
        )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_base_running_example_unlabeled_marker_is_interval_certified(self) -> None:
        sentence = (
            "(forall x: (f(x) = x -> P(x))) & "
            "(forall x: ((f(f(x)) = x & f(x) != x) -> "
            "~(P(x) <-> P(f(x))))) & "
            "(exists=2 x: P(x))"
        )
        analysis = c1f_unlabeled_asymptotics(
            sentence,
            certify=True,
            precision_bits=96,
            polya_tail_terms=16,
        )

        self.assertEqual(
            analysis.regime,
            "capped-global-face:subcritical-polya-multitype",
        )
        self.assertEqual(analysis.power, 1.5)
        assert analysis.certificate is not None
        self.assertEqual(
            analysis.certificate.method,
            "arb-polya-global-marker-derivative",
        )
        derivative = Decimal(str(analysis.marker_growth_derivatives[0]))
        certified_derivative = analysis.certificate.marker_growth_derivatives[0]
        assert certified_derivative is not None
        self.assertLessEqual(
            Decimal(certified_derivative.lower),
            derivative,
        )
        self.assertGreaterEqual(
            Decimal(certified_derivative.upper),
            derivative,
        )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_saturated_unlabeled_global_face_has_radius_certificate(self) -> None:
        analysis = c1f_unlabeled_asymptotics(
            "exists>=1 x: P(x)",
            certify=True,
            precision_bits=96,
            polya_tail_terms=16,
        )

        self.assertEqual(
            analysis.regime,
            "capped-global-face-sum:subcritical-polya",
        )
        self.assertEqual(len(analysis.summands), 2)
        dominant = min(analysis.summands, key=lambda result: result.radius)
        assert dominant.certificate is not None
        self.assertEqual(dominant.certificate.method, "arb-polya-global-face")
        self.assertIn(
            "certified saturated Pólya face radius separation",
            dominant.certificate.verified_properties,
        )

    def test_monadic_unlabeled_nullary_branches_are_summed(self) -> None:
        analysis = c1f_unlabeled_asymptotics("Z | ~Z")

        self.assertEqual(
            analysis.regime,
            "nullary-branch-sum:subcritical-polya",
        )
        self.assertEqual(len(analysis.summands), 2)
        plain = c1f_unlabeled_asymptotics("TRUE")
        self.assertAlmostEqual(
            analysis.estimate(40) / plain.estimate(40),
            2.0,
        )

    def test_multiple_independent_tree_critical_components(self) -> None:
        specification = C1FLabeledSpecification(
            depth=0,
            successor=((True, False), (False, True)),
            multiplicities=(1, 1),
            visible_cycle_lengths=(None, None),
            transient_profiles=(0, 1),
        )
        analysis = analyze_c1f_labeled_asymptotics(specification)

        self.assertEqual(analysis.regime, "multiple-tree-critical")
        self.assertEqual(analysis.critical_components, ((0,), (1,)))
        self.assertEqual(analysis.power, 0.0)
        self.assertAlmostEqual(analysis.leading_constant, 0.5, places=6)

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_multiple_periodic_critical_blocks_are_expanded_and_certified(self) -> None:
        specification = C1FLabeledSpecification(
            depth=0,
            successor=((True, False), (False, True)),
            multiplicities=(2, 2),
            visible_cycle_lengths=(None, None),
            transient_profiles=(0, 1),
            local_caps=(3,),
            local_contributions=((1,), (1,)),
            local_acceptance=(
                frozenset({(0,), (2,)}),
                frozenset({(0,), (2,)}),
            ),
        )

        labeled = analyze_c1f_labeled_asymptotics(
            specification,
            expansion_order=4,
            certify=True,
            precision_bits=96,
        )
        unlabeled = c1f_unlabeled_asymptotics(
            specification,
            expansion_order=4,
            certify=True,
            precision_bits=96,
            polya_tail_terms=24,
        )

        for analysis in (labeled, unlabeled):
            self.assertEqual(analysis.critical_components, ((0,), (1,)))
            self.assertEqual(analysis.periodicity, 2)
            self.assertEqual(analysis.residue_powers, (0.0, None))
            self.assertEqual(len(analysis.residue_expansions[0]), 1)
            self.assertEqual(analysis.residue_expansions[1], tuple())
            assert analysis.certificate is not None
            self.assertTrue(analysis.certificate.dominance_certified)
            self.assertTrue(analysis.certificate.residue_expansions[0])

        domain_size = 20
        exact_labeled = factorial(domain_size) * 8 ** (domain_size // 2)
        self.assertLess(
            abs(labeled.expansion_estimate(domain_size) / exact_labeled - 1.0),
            1e-6,
        )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_unlabeled_higher_terms_improve_and_are_interval_enclosed(self) -> None:
        analysis = c1f_unlabeled_asymptotics(
            "TRUE",
            expansion_order=4,
            certify=True,
            precision_bits=96,
            polya_tail_terms=64,
        )
        exact = 471559682209452602.0

        self.assertLess(
            abs(analysis.expansion_estimate(40) / exact - 1.0),
            abs(analysis.estimate(40) / exact - 1.0),
        )
        assert analysis.certificate is not None
        self.assertEqual(len(analysis.certificate.residue_expansions[0]), 3)
        self.assertIn(
            "Cauchy bounds for every omitted expansion coefficient",
            analysis.certificate.verified_properties,
        )

    @unittest.skipUnless(find_spec("flint"), "python-flint is unavailable")
    def test_repeated_tree_critical_blocks_are_interval_certified(self) -> None:
        specification = C1FLabeledSpecification(
            depth=0,
            successor=((True, False), (False, True)),
            multiplicities=(1, 1),
            visible_cycle_lengths=(None, None),
            transient_profiles=(0, 1),
        )
        analysis = analyze_c1f_labeled_asymptotics(
            specification,
            certify=True,
            precision_bits=128,
        )

        assert analysis.certificate is not None
        self.assertEqual(
            analysis.certificate.method,
            "arb-krawczyk-repeated-block",
        )
        self.assertLess(
            Decimal(analysis.certificate.leading_constant.lower),
            Decimal("0.5"),
        )
        self.assertGreater(
            Decimal(analysis.certificate.leading_constant.upper),
            Decimal("0.5"),
        )

    def test_irreducible_multitype_system_is_solved_numerically(self) -> None:
        analysis = analyze_c1f_labeled_asymptotics(
            "forall x: ~(P(x) & P(f(x)))"
        )
        exact_at_15 = 632924646574215596850

        self.assertEqual(analysis.regime, "multitype-tree-critical")
        self.assertEqual(len(analysis.tree_values), 3)
        self.assertAlmostEqual(analysis.radius, 0.2222181377976229)
        self.assertLess(analysis.critical_residual, 1e-9)
        self.assertAlmostEqual(
            sum(
                left * right
                for left, right in zip(analysis.perron_left, analysis.perron_right)
            ),
            1.0,
        )
        self.assertLess(
            abs(analysis.estimate(15) / float(exact_at_15) - 1.0),
            0.012,
        )

    def test_forbidding_fixed_points_changes_the_analytic_constant(self) -> None:
        analysis = analyze_c1f_labeled_asymptotics(
            "forall x: f(x) != x"
        )
        domain_size = 100
        exact = (domain_size - 1) ** domain_size

        self.assertAlmostEqual(analysis.finite_cycle_correction, -1.0)
        self.assertAlmostEqual(
            analysis.leading_constant,
            exp(-1.0) / sqrt(2.0 * pi),
        )
        self.assertLess(
            abs(analysis.estimate(domain_size) / float(exact) - 1.0),
            0.006,
        )

    def test_guarded_bounded_indegree_two_leading_term(self) -> None:
        sentence = "forall x: exists<=2 y: (f(y) = x & TRUE)"
        analysis = analyze_c1f_labeled_asymptotics(sentence)

        expected_kappa = sqrt(4.0 + 2.0 * sqrt(2.0))
        expected_delta = (sqrt(2.0) - 1.0) * expected_kappa
        self.assertEqual(analysis.regime, "guarded-tree-critical")
        self.assertAlmostEqual(analysis.radius, sqrt(2.0) - 1.0, places=10)
        self.assertAlmostEqual(analysis.tree_values[0], sqrt(2.0), places=9)
        self.assertAlmostEqual(
            analysis.tree_sqrt_coefficients[0],
            expected_kappa,
            places=9,
        )
        self.assertAlmostEqual(
            analysis.denominator_sqrt_coefficient,
            expected_delta,
            places=8,
        )
        exact = float(c1f_fomc(sentence, 40))
        self.assertLess(abs(analysis.estimate(40) / exact - 1.0), 0.005)

    def test_guarded_cycle_poles_recover_permutations(self) -> None:
        sentences = (
            "forall x: exists>=1 y: (f(y) = x & TRUE)",
            "forall x: exists<=1 y: (f(y) = x & TRUE)",
        )
        for sentence in sentences:
            with self.subTest(sentence=sentence):
                analysis = analyze_c1f_labeled_asymptotics(sentence)
                self.assertEqual(analysis.regime, "cycle-pole")
                self.assertEqual(analysis.cycle_pole_order, 1)
                self.assertAlmostEqual(analysis.radius, 1.0)
                self.assertAlmostEqual(analysis.leading_constant, 1.0, places=10)
                self.assertAlmostEqual(
                    analysis.estimate(12) / factorial(12),
                    1.0,
                    places=10,
                )

    def test_cycle_independence_is_not_relaxed_by_newton_tolerance(self) -> None:
        sentence = (
            "forall x: ((P(x) <-> Q(x)) & (P(x) -> "
            "exists>=1 y: (f(y) = x & R(y))))"
        )
        specification = compile_c1f_labeled_specification(sentence)

        self.assertFalse(specification.cycle_transfers_are_tree_independent())
        analysis = analyze_c1f_labeled_asymptotics(
            specification,
            tolerance=0.01,
        )
        self.assertNotEqual(analysis.regime, "cycle-pole")

    def test_qualified_guarded_multitype_matches_exact_coefficients(self) -> None:
        sentence = (
            "forall x: (P(x) -> "
            "exists=2 y: (f(y) = x & Q(y)))"
        )
        analysis = analyze_c1f_labeled_asymptotics(sentence)
        exact = float(c1f_fomc(sentence, 16))

        self.assertEqual(analysis.regime, "guarded-tree-critical")
        self.assertEqual(len(analysis.tree_values), 6)
        self.assertLess(analysis.critical_residual, 1e-9)
        self.assertLess(abs(analysis.estimate(16) / exact - 1.0), 0.007)

    def test_unsupported_systems_are_rejected_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "cardinality markers"):
            compile_c1f_labeled_specification("exists=1 x: P(x)")
        marker_analysis = analyze_c1f_labeled_asymptotics("exists=1 x: P(x)")
        self.assertEqual(marker_analysis.power, 0.5)
        reducible = analyze_c1f_labeled_asymptotics(
            "forall x: (P(x) -> P(f(x)))"
        )
        self.assertEqual(len(reducible.critical_component), 1)
        nullary_sum = analyze_c1f_labeled_asymptotics("Z | ~Z")
        self.assertEqual(nullary_sum.regime, "nullary-branch-sum")
        self.assertEqual(len(nullary_sum.summands), 2)
        self.assertLess(
            abs(nullary_sum.estimate(50) / (2.0 * float(50**50)) - 1.0),
            0.002,
        )


if __name__ == "__main__":
    unittest.main()
