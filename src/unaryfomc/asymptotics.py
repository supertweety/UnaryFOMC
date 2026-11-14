"""Effective asymptotics for regular C1F profile systems."""

from __future__ import annotations

import cmath
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product
from math import e, exp, factorial, gamma, gcd, lgamma, log, pi, sqrt
from typing import Sequence

from .c1f import (
    C1FAtom,
    C1FBinary,
    C1FBooleanConstant,
    C1FEquality,
    C1FFormula,
    C1FNot,
    C1FPredecessorCount,
    C1FQuantifier,
    CompiledC1F,
    _guarded_type_satisfies_local,
    compile_c1f_fomc,
)
from .specification import (
    C1FLabeledSpecification,
    _capped_add,
    _capped_exponential_factor,
    _distribution_multiply,
    _matrix_determinant,
    _matrix_multiply,
    _matrix_power,
    _matrix_trace,
)
from .puiseux import (
    compute_tree_puiseux_coefficients,
    compute_unlabeled_tree_hierarchy,
    labeled_local_laurent_coefficients,
    project_complex_laurent_orbit,
    unlabeled_local_laurent_coefficients,
)


class C1FAsymptoticCertificationError(ValueError):
    """Raised when an optional interval certificate cannot be established."""


@dataclass(frozen=True)
class _C1FPeriodicOrbit:
    """Exact root-of-unity phase data for an implicit tree system."""

    period: int
    tree_residues: tuple[int, ...]
    object_residues: tuple[int, ...]
    witness_profile: int
    witness_support: tuple[int, ...]


@dataclass(frozen=True)
class C1FRealInterval:
    """Printable decimal view of a certified Arb real ball."""

    lower: str
    upper: str
    enclosure: str

    def __str__(self) -> str:
        return self.enclosure


@dataclass(frozen=True)
class C1FAsymptoticTerm:
    """One coefficient ``constant * n**power`` in an asymptotic expansion."""

    power: float
    constant: float


@dataclass(frozen=True)
class C1FCertifiedAsymptoticTerm:
    """An interval-certified coefficient at a fixed power of ``n``."""

    power: float
    constant: C1FRealInterval


@dataclass(frozen=True)
class C1FAsymptoticCertificate:
    """Rigorous certificate data for computed asymptotic parameters.

    ``verified_properties`` is the precise proof inventory.  Tree-critical
    certificates isolate the local characteristic solution.  Exact support
    and phase checks additionally establish global dominance for recognized
    aperiodic and periodic systems.  No certificate is a finite-n error bound.
    """

    precision_bits: int
    method: str
    regime: str
    radius: C1FRealInterval
    exponential_growth: C1FRealInterval
    leading_constant: C1FRealInterval
    finite_cycle_correction: C1FRealInterval
    denominator_coefficient: C1FRealInterval
    tree_values: tuple[C1FRealInterval, ...]
    tree_sqrt_coefficients: tuple[C1FRealInterval, ...]
    perron_right: tuple[C1FRealInterval, ...]
    perron_left: tuple[C1FRealInterval, ...]
    verified_properties: tuple[str, ...]
    dominance_certified: bool = False
    residue_powers: tuple[float | None, ...] = tuple()
    residue_leading_constants: tuple[C1FRealInterval | None, ...] = tuple()
    residue_polynomial_coefficients: tuple[
        tuple[C1FRealInterval, ...], ...
    ] = tuple()
    residue_expansions: tuple[
        tuple[C1FCertifiedAsymptoticTerm, ...], ...
    ] = tuple()
    marker_growth_derivative: C1FRealInterval | None = None
    marker_growth_derivatives: tuple[C1FRealInterval | None, ...] = tuple()


@dataclass(frozen=True)
class C1FLabeledAsymptotics:
    """Asymptotic data for a structurally verified labeled C1F specification.

    The estimate has the form

    ``n! * radius**(-n) * n**power * leading_constant``.

    Regular tree-critical results record their leading term on each certified
    periodic residue class.  Exact tree-independent cycle-pole results also
    record the complete principal polynomial.  Composite nullary and
    capped-state results retain their disjoint contributions in ``summands``.
    Ordinary result fields use binary64 arithmetic; when requested,
    ``certificate`` contains rigorous Arb enclosures for the properties proved
    in its verification inventory.
    """

    radius: float
    exponential_growth: float
    power: float
    leading_constant: float
    tree_values: tuple[float, ...]
    tree_sqrt_coefficients: tuple[float, ...]
    finite_cycle_correction: float
    denominator_sqrt_coefficient: float
    perron_right: tuple[float, ...]
    perron_left: tuple[float, ...]
    critical_residual: float
    eventual_start: int
    periodicity: int = 1
    regime: str = "scalar-tree-critical"
    structurally_verified: bool = True
    cycle_pole_order: int | None = None
    pole_denominator_coefficient: float | None = None
    critical_component: tuple[int, ...] = tuple()
    critical_components: tuple[tuple[int, ...], ...] = tuple()
    critical_directions: tuple[tuple[float, ...], ...] = tuple()
    critical_left_directions: tuple[tuple[float, ...], ...] = tuple()
    residue_powers: tuple[float | None, ...] = tuple()
    residue_leading_constants: tuple[float, ...] = tuple()
    residue_polynomial_coefficients: tuple[tuple[float, ...], ...] = tuple()
    residue_expansions: tuple[tuple[C1FAsymptoticTerm, ...], ...] = tuple()
    higher_arity_exponents: tuple[int, ...] = tuple()
    trace_predicate_count: int = 0
    global_marker_cap: int | None = None
    accepted_marker_states: tuple[int, ...] = tuple()
    marker_growth_derivative: float | None = None
    global_marker_caps: tuple[int, ...] = tuple()
    accepted_global_states: tuple[tuple[int, ...], ...] = tuple()
    marker_growth_derivatives: tuple[float, ...] = tuple()
    summands: tuple["C1FLabeledAsymptotics", ...] = tuple()
    certificate: C1FAsymptoticCertificate | None = None

    def residue_term(self, domain_size: int) -> tuple[float, float] | None:
        """Return ``(power, constant)`` for the residue containing ``n``."""

        if not self.residue_powers:
            return self.power, self.leading_constant
        residue = domain_size % self.periodicity
        power = self.residue_powers[residue]
        constant = self.residue_leading_constants[residue]
        if power is None or constant == 0.0:
            return None
        return power, constant

    def expansion_terms(self, domain_size: int) -> tuple[C1FAsymptoticTerm, ...]:
        """Return all computed terms for the residue containing ``n``."""

        if self.residue_expansions:
            return self.residue_expansions[domain_size % self.periodicity]
        term = self.residue_term(domain_size)
        if term is None:
            return tuple()
        return (C1FAsymptoticTerm(term[0], term[1]),)

    def expansion_estimate(self, domain_size: int) -> float:
        """Evaluate every available asymptotic term at ``domain_size``."""

        if domain_size <= 0:
            raise ValueError("Asymptotic estimates require a positive domain size")
        if self.summands:
            return sum(
                summand.expansion_estimate(domain_size)
                for summand in self.summands
            )
        terms = self.expansion_terms(domain_size)
        if not terms:
            return 0.0
        polynomial = sum(
            term.constant * domain_size**term.power
            for term in terms
        )
        if polynomial <= 0.0:
            return 0.0
        higher_arity_log = log(2.0) * (
            sum(domain_size**arity for arity in self.higher_arity_exponents)
            - self.trace_predicate_count * domain_size
        )
        return exp(
            lgamma(domain_size + 1)
            - domain_size * log(self.radius)
            + log(polynomial)
            + higher_arity_log
        )

    def log_estimate(self, domain_size: int) -> float:
        """Return the logarithm of the leading asymptotic estimate."""

        if domain_size <= 0:
            raise ValueError("Asymptotic estimates require a positive domain size")
        if self.summands:
            logarithms = tuple(
                summand.log_estimate(domain_size)
                for summand in self.summands
            )
            finite = tuple(value for value in logarithms if value != float("-inf"))
            if not finite:
                return float("-inf")
            maximum = max(finite)
            return maximum + log(sum(exp(value - maximum) for value in finite))
        term = self.residue_term(domain_size)
        if term is None:
            return float("-inf")
        power, leading_constant = term
        if self.residue_polynomial_coefficients:
            polynomial = self.residue_polynomial_coefficients[
                domain_size % self.periodicity
            ]
            polynomial_value = sum(
                coefficient * domain_size**degree
                for degree, coefficient in enumerate(polynomial)
            )
            if polynomial_value <= 0.0:
                return float("-inf")
            power = 0.0
            leading_constant = polynomial_value
        higher_arity_log = log(2.0) * (
            sum(domain_size**arity for arity in self.higher_arity_exponents)
            - self.trace_predicate_count * domain_size
        )
        return (
            lgamma(domain_size + 1)
            - domain_size * log(self.radius)
            + power * log(domain_size)
            + log(leading_constant)
            + higher_arity_log
        )

    def estimate(self, domain_size: int) -> float:
        """Return the leading asymptotic estimate in ordinary scale."""

        return exp(self.log_estimate(domain_size))


@dataclass(frozen=True)
class C1FUnlabeledAsymptotics:
    """Leading unlabeled asymptotic ``C rho**(-n) n**power``."""

    radius: float
    exponential_growth: float
    power: float
    leading_constant: float
    periodicity: int = 1
    regime: str = "subcritical-polya"
    labeled: C1FLabeledAsymptotics | None = None
    tree_values: tuple[float, ...] = tuple()
    critical_components: tuple[tuple[int, ...], ...] = tuple()
    global_marker_caps: tuple[int, ...] = tuple()
    accepted_global_states: tuple[tuple[int, ...], ...] = tuple()
    marker_growth_derivatives: tuple[float, ...] = tuple()
    residue_powers: tuple[float | None, ...] = tuple()
    residue_leading_constants: tuple[float, ...] = tuple()
    residue_expansions: tuple[tuple[C1FAsymptoticTerm, ...], ...] = tuple()
    summands: tuple["C1FUnlabeledAsymptotics", ...] = tuple()
    certificate: C1FAsymptoticCertificate | None = None

    def residue_term(self, domain_size: int) -> tuple[float, float] | None:
        """Return ``(power, constant)`` for the residue containing ``n``."""

        if not self.residue_powers:
            return self.power, self.leading_constant
        residue = domain_size % self.periodicity
        power = self.residue_powers[residue]
        constant = self.residue_leading_constants[residue]
        if power is None or constant == 0.0:
            return None
        return power, constant

    def expansion_terms(self, domain_size: int) -> tuple[C1FAsymptoticTerm, ...]:
        if self.residue_expansions:
            return self.residue_expansions[domain_size % self.periodicity]
        term = self.residue_term(domain_size)
        if term is None:
            return tuple()
        return (C1FAsymptoticTerm(term[0], term[1]),)

    def expansion_estimate(self, domain_size: int) -> float:
        if domain_size <= 0:
            raise ValueError("Asymptotic estimates require a positive domain size")
        if self.summands:
            return sum(
                summand.expansion_estimate(domain_size)
                for summand in self.summands
            )
        if self.labeled is not None:
            return self.labeled.expansion_estimate(domain_size) / exp(
                lgamma(domain_size + 1)
            )
        value = sum(
            term.constant * domain_size**term.power
            for term in self.expansion_terms(domain_size)
        )
        if value <= 0.0:
            return 0.0
        return self.radius ** (-domain_size) * value

    def log_estimate(self, domain_size: int) -> float:
        if domain_size <= 0:
            raise ValueError("Asymptotic estimates require a positive domain size")
        if self.summands:
            logarithms = tuple(
                summand.log_estimate(domain_size)
                for summand in self.summands
            )
            finite = tuple(value for value in logarithms if value != float("-inf"))
            if not finite:
                return float("-inf")
            maximum = max(finite)
            return maximum + log(
                sum(exp(value - maximum) for value in finite)
            )
        if self.labeled is not None:
            return self.labeled.log_estimate(domain_size) - lgamma(domain_size + 1)
        term = self.residue_term(domain_size)
        if term is None:
            return float("-inf")
        power, leading_constant = term
        return (
            -domain_size * log(self.radius)
            + power * log(domain_size)
            + log(leading_constant)
        )

    def estimate(self, domain_size: int) -> float:
        return exp(self.log_estimate(domain_size))


def _shift_expansions(
    expansions: Sequence[Sequence[C1FAsymptoticTerm]],
    power: float,
    factor: float,
) -> tuple[tuple[C1FAsymptoticTerm, ...], ...]:
    return tuple(
        tuple(
            C1FAsymptoticTerm(term.power + power, term.constant * factor)
            for term in residue
        )
        for residue in expansions
    )


def _solve_linear_system(
    matrix: Sequence[Sequence[complex | float]],
    right_hand_side: Sequence[complex | float],
) -> tuple[complex | float, ...]:
    size = len(matrix)
    work = [
        [*row, right_hand_side[index]]
        for index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row: abs(work[row][column]))
        if abs(work[pivot_row][column]) < 1e-14:
            raise ValueError("Singular linear system")
        if pivot_row != column:
            work[column], work[pivot_row] = work[pivot_row], work[column]
        pivot = work[column][column]
        for trailing in range(column, size + 1):
            work[column][trailing] /= pivot
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            for trailing in range(column, size + 1):
                work[row][trailing] -= factor * work[column][trailing]
    return tuple(work[row][size] for row in range(size))


def _is_strongly_connected(matrix: Sequence[Sequence[bool]]) -> bool:
    size = len(matrix)
    if size == 0:
        return False

    def reaches_all(reverse: bool) -> bool:
        seen = {0}
        pending = [0]
        while pending:
            source = pending.pop()
            for target in range(size):
                adjacent = matrix[target][source] if reverse else matrix[source][target]
                if adjacent and target not in seen:
                    seen.add(target)
                    pending.append(target)
        return len(seen) == size

    return reaches_all(False) and reaches_all(True)


def _strongly_connected_components(
    matrix: Sequence[Sequence[bool]],
) -> tuple[tuple[int, ...], ...]:
    """Return SCCs of a Boolean adjacency matrix in a stable order."""

    size = len(matrix)
    index = 0
    indices = [-1] * size
    lowlinks = [0] * size
    stack: list[int] = []
    on_stack = [False] * size
    components: list[tuple[int, ...]] = []

    def visit(vertex: int) -> None:
        nonlocal index
        indices[vertex] = index
        lowlinks[vertex] = index
        index += 1
        stack.append(vertex)
        on_stack[vertex] = True
        for target, adjacent in enumerate(matrix[vertex]):
            if not adjacent:
                continue
            if indices[target] < 0:
                visit(target)
                lowlinks[vertex] = min(lowlinks[vertex], lowlinks[target])
            elif on_stack[target]:
                lowlinks[vertex] = min(lowlinks[vertex], indices[target])
        if lowlinks[vertex] != indices[vertex]:
            return
        component: list[int] = []
        while True:
            member = stack.pop()
            on_stack[member] = False
            component.append(member)
            if member == vertex:
                break
        components.append(tuple(sorted(component)))

    for vertex in range(size):
        if indices[vertex] < 0:
            visit(vertex)
    return tuple(sorted(components, key=lambda component: component[0]))


def _perron_eigenvalue(matrix: Sequence[Sequence[float]]) -> float:
    """Compute the Perron root by power iteration on ``I + matrix``."""

    size = len(matrix)
    vector = tuple(1.0 / size for _ in range(size))
    shifted_eigenvalue = 1.0
    for _ in range(2000):
        product = tuple(
            vector[row]
            + sum(matrix[row][column] * vector[column] for column in range(size))
            for row in range(size)
        )
        shifted_eigenvalue = sum(product)
        if shifted_eigenvalue <= 0.0:
            raise ValueError("The dependency matrix has no positive Perron root")
        next_vector = tuple(value / shifted_eigenvalue for value in product)
        if max(abs(a - b) for a, b in zip(next_vector, vector)) < 1e-14:
            vector = next_vector
            break
        vector = next_vector
    product = tuple(
        vector[row]
        + sum(matrix[row][column] * vector[column] for column in range(size))
        for row in range(size)
    )
    shifted_eigenvalue = sum(product)
    return shifted_eigenvalue - 1.0


def _critical_residual(
    specification: C1FLabeledSpecification,
    logarithms: Sequence[float],
) -> tuple[float, ...]:
    x = exp(logarithms[0])
    trees = tuple(exp(value) for value in logarithms[1:])
    rhs = specification.tree_rhs(x, trees)
    equations = tuple(
        log(tree) - log(rhs_value)
        for tree, rhs_value in zip(trees, rhs)
    )
    return equations + (specification.long_cycle_determinant(x, trees),)


def _solve_multitype_critical_point(
    specification: C1FLabeledSpecification,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[float, tuple[float, ...], float]:
    dependencies = specification.dependency_matrix()
    dimension = specification.tree_dimension
    multiplicities = tuple(
        specification.multiplicities[profile]
        for profile in specification.transient_profiles
    )
    total_multiplicity = sum(multiplicities)
    base = tuple(value / total_multiplicity for value in multiplicities)
    scaled_dependency = tuple(
        tuple(
            base[row] if dependencies[row][column] else 0.0
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    perron_scale = _perron_eigenvalue(scaled_dependency)
    if perron_scale <= 0.0:
        raise ValueError("The transient dependency system is not tree-critical")
    initial_trees = tuple(value / perron_scale for value in base)
    initial_log_x = sum(
        log(initial_trees[row] / multiplicities[row])
        - sum(
            initial_trees[column]
            for column in range(dimension)
            if dependencies[row][column]
        )
        for row in range(dimension)
    ) / dimension
    logarithms = [initial_log_x, *(log(value) for value in initial_trees)]

    residual_norm = float("inf")
    for _ in range(max_iterations):
        residual = _critical_residual(specification, logarithms)
        residual_norm = max(abs(value) for value in residual)
        # A determinant has a zero of order equal to the number of coincident
        # critical SCCs.  Stopping at the caller's first-order tolerance can
        # therefore leave each Perron root only at its square/cubic root.
        if residual_norm <= min(tolerance, 1e-14):
            break

        jacobian = [[0.0] * (dimension + 1) for _ in range(dimension + 1)]
        for column in range(dimension + 1):
            step = 1e-5 * (1.0 + abs(logarithms[column]))
            plus = list(logarithms)
            minus = list(logarithms)
            plus[column] += step
            minus[column] -= step
            plus_residual = _critical_residual(specification, plus)
            minus_residual = _critical_residual(specification, minus)
            for row in range(dimension + 1):
                jacobian[row][column] = (
                    plus_residual[row] - minus_residual[row]
                ) / (2.0 * step)
        update = _solve_linear_system(jacobian, tuple(-value for value in residual))

        step_scale = 1.0
        while step_scale >= 1e-8:
            candidate = [
                value + step_scale * delta
                for value, delta in zip(logarithms, update)
            ]
            try:
                candidate_residual = _critical_residual(specification, candidate)
                candidate_norm = max(abs(value) for value in candidate_residual)
            except (OverflowError, ValueError):
                candidate_norm = float("inf")
            if candidate_norm < residual_norm:
                logarithms = candidate
                break
            step_scale *= 0.5
        else:
            raise ValueError(
                "The multitype critical-point Newton iteration failed to decrease its residual"
            )
    else:
        raise ValueError(
            "The multitype critical-point Newton iteration did not converge; "
            f"last residual was {residual_norm:.3g}"
        )

    radius = exp(logarithms[0])
    trees = tuple(exp(value) for value in logarithms[1:])
    return radius, trees, residual_norm


def _kernel_vector(
    matrix: Sequence[Sequence[float]],
    *,
    allow_zeros: bool = False,
) -> tuple[float, ...]:
    """Find a non-negative normalized kernel vector of ``I - matrix``."""

    size = len(matrix)
    operator = [
        [
            (1.0 if row == column else 0.0) - matrix[row][column]
            for column in range(size)
        ]
        for row in range(size)
    ]
    best: tuple[float, tuple[float, ...]] | None = None
    for replaced_row in range(size):
        constrained = [row[:] for row in operator]
        constrained[replaced_row] = [1.0] * size
        right_hand_side = [0.0] * size
        right_hand_side[replaced_row] = 1.0
        try:
            candidate = _solve_linear_system(constrained, right_hand_side)
        except ValueError:
            continue
        residual = max(
            abs(sum(operator[row][column] * candidate[column] for column in range(size)))
            for row in range(size)
        )
        if allow_zeros:
            if min(candidate) < -1e-9:
                continue
            candidate = tuple(max(0.0, value) for value in candidate)
        elif min(candidate) <= 0.0:
            continue
        if best is None or residual < best[0]:
            best = (residual, candidate)
    if best is None or best[0] > 1e-6:
        raise ValueError("The critical Jacobian has no reliable positive Perron vector")
    total = sum(best[1])
    return tuple(value / total for value in best[1])


def _principal_submatrix(
    matrix: Sequence[Sequence[float]],
    indices: Sequence[int],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(matrix[row][column] for column in indices)
        for row in indices
    )


def _transpose(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(matrix[row][column] for row in range(len(matrix)))
        for column in range(len(matrix))
    )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _critical_kernel_bases(
    matrix: Sequence[Sequence[float]],
    components: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """Compute normalized full left/right directions for simple critical SCCs."""

    dimension = len(matrix)
    operator = [
        [
            (1.0 if row == column else 0.0) - matrix[row][column]
            for column in range(dimension)
        ]
        for row in range(dimension)
    ]

    def basis(source: Sequence[Sequence[float]]) -> list[tuple[float, ...]]:
        result = []
        for selected in range(len(components)):
            constrained = [list(row) for row in source]
            rhs = [0.0] * dimension
            for component_index, component in enumerate(components):
                pivot = component[-1]
                constrained[pivot] = [
                    1.0 if column in component else 0.0
                    for column in range(dimension)
                ]
                rhs[pivot] = 1.0 if component_index == selected else 0.0
            vector = _solve_linear_system(constrained, rhs)
            if min(vector) < -1e-8:
                raise ValueError(
                    "The critical components do not yield non-negative Perron directions"
                )
            result.append(tuple(max(0.0, value) for value in vector))
        return result

    right = basis(operator)
    transposed = [
        [operator[column][row] for column in range(dimension)]
        for row in range(dimension)
    ]
    raw_left = basis(transposed)
    left = []
    for index, vector in enumerate(raw_left):
        normalization = _dot(vector, right[index])
        if normalization <= 0.0:
            raise ValueError("A critical left/right normalization is not positive")
        left.append(tuple(value / normalization for value in vector))
    return tuple(right), tuple(left)


def _components_are_independent(
    dependencies: Sequence[Sequence[bool]],
    components: Sequence[Sequence[int]],
) -> bool:
    critical = {
        vertex: component_index
        for component_index, component in enumerate(components)
        for vertex in component
    }
    for source_component, component in enumerate(components):
        reached = set(component)
        pending = list(component)
        while pending:
            source = pending.pop()
            for target, adjacent in enumerate(dependencies[source]):
                if not adjacent or target in reached:
                    continue
                target_component = critical.get(target)
                if (
                    target_component is not None
                    and target_component != source_component
                ):
                    return False
                reached.add(target)
                pending.append(target)
    return True


def _denominator_sqrt_coefficient(
    specification: C1FLabeledSpecification,
    radius: float,
    trees: Sequence[float],
    tree_sqrt_coefficients: Sequence[float],
) -> float:
    critical_value = specification.long_cycle_determinant(radius, trees)
    max_step = min(
        tree / coefficient
        for tree, coefficient in zip(trees, tree_sqrt_coefficients)
        if coefficient > 0.0
    )
    step = min(1e-4, max_step / 10.0)

    def slope(local_step: float) -> float:
        shifted = tuple(
            tree - local_step * coefficient
            for tree, coefficient in zip(trees, tree_sqrt_coefficients)
        )
        value = specification.long_cycle_determinant(radius, shifted)
        return (value - critical_value) / local_step

    coarse = slope(step)
    fine = slope(step / 2.0)
    return 2.0 * fine - coarse


def _denominator_direction_coefficient(
    specification: C1FLabeledSpecification,
    radius: float,
    trees: Sequence[float],
    tree_sqrt_coefficients: Sequence[float],
    order: int,
) -> tuple[float, float]:
    """Estimate the coefficient of ``X**order`` in the cycle determinant."""

    max_step = min(
        (
            tree / coefficient
            for tree, coefficient in zip(trees, tree_sqrt_coefficients)
            if coefficient > 0.0
        ),
        default=1.0,
    )
    step = min(2e-3, max_step / 20.0)

    def scaled(local_step: float) -> float:
        shifted = tuple(
            tree - local_step * coefficient
            for tree, coefficient in zip(trees, tree_sqrt_coefficients)
        )
        return (
            specification.long_cycle_determinant(radius, shifted)
            / local_step**order
        )

    coarse = scaled(step)
    fine = scaled(step / 2.0)
    finer = scaled(step / 4.0)
    first = 2.0 * fine - coarse
    second = 2.0 * finer - fine
    extrapolated = (4.0 * second - first) / 3.0
    return extrapolated, abs(extrapolated - second)


def _largest_count_threshold(formula: C1FFormula) -> int:
    if isinstance(formula, (C1FBooleanConstant, C1FAtom, C1FEquality)):
        return 0
    if isinstance(formula, C1FNot):
        return _largest_count_threshold(formula.body)
    if isinstance(formula, C1FBinary):
        return max(
            _largest_count_threshold(formula.left),
            _largest_count_threshold(formula.right),
        )
    if isinstance(formula, C1FQuantifier):
        local = formula.count if formula.count is not None else 0
        return max(local, _largest_count_threshold(formula.body))
    if isinstance(formula, C1FPredecessorCount):
        return max(formula.count, _largest_count_threshold(formula.body))
    raise TypeError(f"Unsupported C1F node: {type(formula)!r}")


def _is_local_formula(formula: C1FFormula) -> bool:
    if isinstance(formula, (C1FBooleanConstant, C1FAtom, C1FEquality)):
        return True
    if isinstance(formula, C1FNot):
        return _is_local_formula(formula.body)
    if isinstance(formula, C1FBinary):
        return _is_local_formula(formula.left) and _is_local_formula(formula.right)
    if isinstance(formula, C1FPredecessorCount):
        return _is_local_formula(formula.body)
    if isinstance(formula, C1FQuantifier):
        return False
    raise TypeError(f"Unsupported C1F node: {type(formula)!r}")


def _entailed_local_filters(formula: C1FFormula) -> tuple[C1FFormula, ...]:
    """Collect local conditions imposed on every element by conjunctions."""

    filters: list[C1FFormula] = []

    def visit(current: C1FFormula) -> None:
        if isinstance(current, C1FBinary) and current.op == "&":
            visit(current.left)
            visit(current.right)
            return
        if (
            isinstance(current, C1FQuantifier)
            and current.kind == "forall"
            and _is_local_formula(current.body)
        ):
            filters.append(current.body)

    visit(formula)
    return tuple(filters)


def _capped_state_add(
    left: tuple[int, ...],
    right: tuple[int, ...],
    caps: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        min(cap, left_value + right_value)
        for cap, left_value, right_value in zip(caps, left, right)
    )


def _reachable_local_states(
    contributions: Sequence[tuple[int, ...]],
    caps: tuple[int, ...],
) -> frozenset[tuple[int, ...]]:
    zero = tuple(0 for _ in caps)
    reached = {zero}
    pending = [zero]
    distinct_contributions = tuple(dict.fromkeys(contributions))
    while pending:
        state = pending.pop()
        for contribution in distinct_contributions:
            updated = _capped_state_add(state, contribution, caps)
            if updated not in reached:
                reached.add(updated)
                pending.append(updated)
    return frozenset(reached)


def _productive_tree_profiles(
    successor: Sequence[Sequence[bool]],
    visible_cycle_lengths: Sequence[int | None],
    contributions: Sequence[tuple[int, ...]],
    acceptance: Sequence[frozenset[tuple[int, ...]]],
    caps: tuple[int, ...],
) -> tuple[int, ...]:
    """Return types that support at least one finite rooted in-tree."""

    candidates = tuple(
        profile
        for profile, visible_length in enumerate(visible_cycle_lengths)
        if visible_length is None
    )
    productive: set[int] = set()
    while True:
        added = []
        for root in candidates:
            if root in productive:
                continue
            child_contributions = tuple(
                contributions[child]
                for child in productive
                if successor[child][root]
            )
            reachable = _reachable_local_states(child_contributions, caps)
            if acceptance[root] & reachable:
                added.append(root)
        if not added:
            break
        productive.update(added)
    return tuple(profile for profile in candidates if profile in productive)


def _eventual_marker_start(monadic, active_type_ids: Sequence[int]) -> int:
    """Verify that global markers become deterministic on the active types."""

    plan = monadic.cardinality_plan
    eventual_start = _largest_count_threshold(monadic.formula) + 1
    if not plan.caps:
        if not plan.satisfies(monadic.formula, tuple(), eventual_start):
            raise ValueError(
                "The compiled marker-free branch is eventually rejecting; "
                "it has no positive infinite-support asymptotic regime"
            )
        return eventual_start

    per_element_states = {
        plan.profile_states[type_id]
        for type_id in active_type_ids
    }
    if len(per_element_states) != 1:
        raise ValueError(
            "Labeled specification extraction does not yet support global "
            "cardinality markers that distinguish active guarded types"
        )
    per_element = next(iter(per_element_states))
    eventual_start = max(eventual_start, max(plan.caps, default=0))
    for domain_size in (eventual_start, eventual_start + 1):
        marker_state = tuple(
            min(cap, domain_size * contribution)
            for cap, contribution in zip(plan.caps, per_element)
        )
        if not plan.satisfies(monadic.formula, marker_state, domain_size):
            raise ValueError(
                "The compiled guarded branch is eventually rejecting; "
                "it has no positive infinite-support asymptotic regime"
            )
    return eventual_start


def _build_labeled_specification(
    monadic: object,
    active_old_types: Sequence[int],
    *,
    eventual_start: int,
    multiplicity_scales: Sequence[float] | None = None,
) -> C1FLabeledSpecification:
    predecessor_plan = monadic.predecessor_plan
    active_old_types = tuple(active_old_types)
    type_count = len(active_old_types)
    successor = tuple(
        tuple(
            monadic.successor[
                predecessor_plan.types[old_child].profile_id
            ][predecessor_plan.types[old_root].profile_id]
            for old_root in active_old_types
        )
        for old_child in active_old_types
    )
    multiplicities = tuple(
        monadic.profile_multiplicities[
            predecessor_plan.types[old_type].profile_id
        ]
        * (1.0 if multiplicity_scales is None else multiplicity_scales[index])
        for index, old_type in enumerate(active_old_types)
    )
    visible_cycle_lengths = tuple(
        monadic.profiles[
            predecessor_plan.types[old_type].profile_id
        ].visible_cycle_length
        for old_type in active_old_types
    )
    long_cycle_profiles = tuple(
        type_id
        for type_id, visible_length in enumerate(visible_cycle_lengths)
        if visible_length is None
    )

    if predecessor_plan.constraints:
        caps = predecessor_plan.caps
        contributions = tuple(
            predecessor_plan.contributions[old_type]
            for old_type in active_old_types
        )
        all_states = tuple(product(*(range(cap + 1) for cap in caps)))
        acceptance = tuple(
            frozenset(
                state
                for state in all_states
                if predecessor_plan.accepts(old_type, state)
            )
            for old_type in active_old_types
        )
        transient_profiles = _productive_tree_profiles(
            successor,
            visible_cycle_lengths,
            contributions,
            acceptance,
            caps,
        )
    else:
        caps = tuple()
        contributions = tuple(tuple() for _ in range(type_count))
        acceptance = tuple(frozenset((tuple(),)) for _ in range(type_count))
        transient_profiles = long_cycle_profiles

    return C1FLabeledSpecification(
        depth=monadic.depth,
        successor=successor,
        multiplicities=multiplicities,
        visible_cycle_lengths=visible_cycle_lengths,
        transient_profiles=transient_profiles,
        eventual_start=eventual_start,
        long_cycle_profiles=long_cycle_profiles,
        local_caps=caps,
        local_contributions=contributions if caps else tuple(),
        local_acceptance=acceptance if caps else tuple(),
    )


def compile_c1f_labeled_specification(
    formula: str | C1FFormula | CompiledC1F,
    *,
    function_symbol: str = "f",
    profile_pruning: bool = True,
    twin_compression: bool = True,
) -> C1FLabeledSpecification:
    """Extract the finite EGF specification used by labeled C1F counting.

    The public specification layer handles a single monadic branch.  Universal
    local guarded conditions are compiled into type admissibility and capped
    predecessor projections.  Genuinely population-sensitive global markers
    are rejected explicitly.
    """

    compiled = (
        formula
        if isinstance(formula, CompiledC1F)
        else compile_c1f_fomc(
            formula,
            function_symbol=function_symbol,
            profile_pruning=profile_pruning,
            twin_compression=twin_compression,
        )
    )
    if len(compiled.branches) != 1:
        raise ValueError(
            "Labeled specification extraction currently requires one nullary branch"
        )
    branch = compiled.branches[0]
    if branch.higher_relation_arities or branch.trace_predicate_count:
        raise ValueError(
            "Labeled specification extraction currently supports monadic predicates only"
        )
    monadic = branch.monadic
    predecessor_plan = monadic.predecessor_plan
    local_filters = _entailed_local_filters(monadic.formula)
    active_old_types = tuple(
        guarded_type.id
        for guarded_type in predecessor_plan.types
        if all(
            _guarded_type_satisfies_local(
                guarded_type,
                local_filter,
                monadic.profiles,
                monadic.predicate_index,
                predecessor_plan.predecessor_index,
            )
            for local_filter in local_filters
        )
    )
    if not active_old_types:
        raise ValueError(
            "The universal guarded conditions reject every local type; "
            "there is no positive infinite-support asymptotic regime"
        )
    eventual_start = _eventual_marker_start(monadic, active_old_types)

    return _build_labeled_specification(
        monadic,
        active_old_types,
        eventual_start=eventual_start,
    )


def _matrix_period(matrix: Sequence[Sequence[float]]) -> int:
    """Return the period of a strongly connected non-negative matrix."""

    size = len(matrix)
    distances = [-1] * size
    distances[0] = 0
    pending = [0]
    while pending:
        source = pending.pop()
        for target, value in enumerate(matrix[source]):
            if value <= 0.0 or distances[target] >= 0:
                continue
            distances[target] = distances[source] + 1
            pending.append(target)
    period = 0
    for source, row in enumerate(matrix):
        for target, value in enumerate(row):
            if value > 0.0:
                period = gcd(
                    period,
                    abs(distances[source] + 1 - distances[target]),
                )
    return period


def _exact_tree_supports(
    specification: C1FLabeledSpecification,
    *,
    unlabeled: bool,
    degree_limit: int = 16,
) -> tuple[tuple[int, ...], ...]:
    """Compute an exact finite prefix of every rooted-tree support."""

    from .c1f import (
        _atom_times_series,
        _series_add_in_place,
        _series_append_state,
        _series_exp,
        _series_filter_local_state,
        _series_multiset,
        _zero_series,
    )

    collection = _series_multiset if unlabeled else _series_exp
    transient = specification.transient_profiles
    trees = {
        profile: _zero_series(degree_limit)
        for profile in transient
    }
    for _ in range(degree_limit):
        next_trees = {}
        for root_profile in transient:
            incoming = _zero_series(degree_limit)
            for child_profile in transient:
                if not specification.successor[child_profile][root_profile]:
                    continue
                lifted = _series_append_state(
                    trees[child_profile],
                    specification._local_contribution(child_profile),
                    degree_limit,
                )
                _series_add_in_place(incoming, lifted)
            children = collection(
                incoming,
                specification.local_caps,
                degree_limit,
            )
            accepted = specification._accepted_states(root_profile)
            valid = _series_filter_local_state(
                children,
                0,
                lambda state, accepted=accepted: state in accepted,
                degree_limit,
            )
            next_trees[root_profile] = _atom_times_series(
                tuple(),
                specification.multiplicities[root_profile],
                valid,
                tuple(),
                degree_limit,
            )
        trees = next_trees
    return tuple(
        tuple(
            degree
            for degree, coefficient in enumerate(trees[profile])
            if coefficient
        )
        for profile in transient
    )


def _modular_child_states(
    specification: C1FLabeledSpecification,
    root_profile: int,
    tree_residues: Sequence[set[int]],
    modulus: int,
) -> frozenset[tuple[tuple[int, ...], int]]:
    """Return all capped child states and size residues at one root."""

    zero = tuple(0 for _ in specification.local_caps)
    options = {
        (specification._local_contribution(child_profile), residue)
        for child_index, child_profile in enumerate(
            specification.transient_profiles
        )
        if specification.successor[child_profile][root_profile]
        for residue in tree_residues[child_index]
    }
    reached = {(zero, 0)}
    pending = [(zero, 0)]
    while pending:
        state, residue = pending.pop()
        for contribution, child_residue in options:
            updated = (
                _capped_add(state, contribution, specification.local_caps),
                (residue + child_residue) % modulus,
            )
            if updated not in reached:
                reached.add(updated)
                pending.append(updated)
    return frozenset(reached)


def _modular_tree_residues(
    specification: C1FLabeledSpecification,
    modulus: int,
) -> tuple[frozenset[int], ...]:
    """Compute the exact least fixed point of tree-size residues modulo d."""

    residues = [set() for _ in specification.transient_profiles]
    while True:
        changed = False
        for coordinate, root_profile in enumerate(
            specification.transient_profiles
        ):
            child_states = _modular_child_states(
                specification,
                root_profile,
                residues,
                modulus,
            )
            generated = {
                (1 + residue) % modulus
                for state, residue in child_states
                if state in specification._accepted_states(root_profile)
            }
            before = len(residues[coordinate])
            residues[coordinate].update(generated)
            changed |= len(residues[coordinate]) != before
        if not changed:
            return tuple(frozenset(values) for values in residues)


def _modular_cycle_edge_residues(
    specification: C1FLabeledSpecification,
    tree_residues: Sequence[set[int] | frozenset[int]],
    modulus: int,
) -> tuple[tuple[frozenset[int], ...], ...]:
    """Return exact size residues carried by every cycle-transfer edge."""

    child_states = {
        current: _modular_child_states(
            specification,
            current,
            [set(values) for values in tree_residues],
            modulus,
        )
        for current in range(specification.profile_count)
    }
    return tuple(
        tuple(
            frozenset()
            if not specification.successor[previous][current]
            else frozenset(
                (1 + residue) % modulus
                for state, residue in child_states[current]
                if _capped_add(
                    state,
                    specification._local_contribution(previous),
                    specification.local_caps,
                ) in specification._accepted_states(current)
            )
            for current in range(specification.profile_count)
        )
        for previous in range(specification.profile_count)
    )


def _modular_closed_walk_residues(
    active_profiles: Sequence[int],
    edge_residues: Sequence[Sequence[frozenset[int]]],
    modulus: int,
    length: int,
) -> set[int]:
    result: set[int] = set()
    for start in active_profiles:
        paths = {(start, 0)}
        for _ in range(length):
            paths = {
                (current, (residue + edge_residue) % modulus)
                for previous, residue in paths
                for current in active_profiles
                for edge_residue in edge_residues[previous][current]
            }
            if not paths:
                break
        result.update(
            residue for profile, residue in paths if profile == start
        )
    return result


def _modular_long_cycle_residues(
    active_profiles: Sequence[int],
    edge_residues: Sequence[Sequence[frozenset[int]]],
    modulus: int,
    minimum_length: int,
) -> set[int]:
    """Return residues of all closed walks longer than ``minimum_length``."""

    result: set[int] = set()
    target_length = minimum_length + 1
    for start in active_profiles:
        reached = {(start, 0, 0)}
        pending = [(start, 0, 0)]
        while pending:
            previous, residue, length = pending.pop()
            for current in active_profiles:
                for edge_residue in edge_residues[previous][current]:
                    updated_length = min(target_length, length + 1)
                    updated = (
                        current,
                        (residue + edge_residue) % modulus,
                        updated_length,
                    )
                    if updated not in reached:
                        reached.add(updated)
                        pending.append(updated)
        result.update(
            residue
            for profile, residue, length in reached
            if profile == start and length == target_length
        )
    return result


def _modular_object_residues(
    specification: C1FLabeledSpecification,
    tree_residues: Sequence[set[int] | frozenset[int]],
    modulus: int,
) -> tuple[int, ...]:
    """Compute all residues admitted by cycles and the outer collection."""

    edges = _modular_cycle_edge_residues(
        specification,
        tree_residues,
        modulus,
    )
    connected: set[int] = set()
    for cycle_length in range(1, specification.depth + 1):
        active = tuple(
            profile
            for profile, visible_length in enumerate(
                specification.visible_cycle_lengths
            )
            if visible_length == cycle_length
        )
        connected.update(
            _modular_closed_walk_residues(
                active,
                edges,
                modulus,
                cycle_length,
            )
        )
    connected.update(
        _modular_long_cycle_residues(
            specification._long_cycle_profiles(),
            edges,
            modulus,
            specification.depth,
        )
    )

    objects = {0}
    pending = [0]
    while pending:
        residue = pending.pop()
        for component_residue in connected:
            updated = (residue + component_residue) % modulus
            if updated not in objects:
                objects.add(updated)
                pending.append(updated)
    return tuple(sorted(objects))


def _exact_periodic_orbit(
    specification: C1FLabeledSpecification,
    critical_component: Sequence[int],
    *,
    unlabeled: bool,
    degree_limit: int = 16,
) -> _C1FPeriodicOrbit | None:
    """Construct and prove a finite root-of-unity orbit from exact supports."""

    supports = _exact_tree_supports(
        specification,
        unlabeled=unlabeled,
        degree_limit=degree_limit,
    )
    period = 0
    witness_coordinate = -1
    for coordinate in critical_component:
        support = supports[coordinate]
        if len(support) < 2:
            continue
        witness_coordinate = coordinate
        anchor = support[0]
        for degree in support[1:]:
            period = gcd(period, degree - anchor)
    if period <= 1 or witness_coordinate < 0:
        return None

    modular = _modular_tree_residues(specification, period)
    if any(len(values) != 1 for values in modular):
        return None
    tree_residues = tuple(next(iter(values)) for values in modular)
    if any(
        any(degree % period != tree_residues[index] for degree in support)
        for index, support in enumerate(supports)
    ):
        return None
    witness_profile = specification.transient_profiles[witness_coordinate]
    return _C1FPeriodicOrbit(
        period=period,
        tree_residues=tree_residues,
        object_residues=_modular_object_residues(
            specification,
            modular,
            period,
        ),
        witness_profile=witness_profile,
        witness_support=supports[witness_coordinate],
    )


@dataclass(frozen=True)
class _ComplexDirectional:
    """A complex value with one formal directional derivative."""

    value: complex
    derivative: complex = 0j

    @classmethod
    def coerce(cls, value: complex | float | "_ComplexDirectional") -> "_ComplexDirectional":
        return value if isinstance(value, cls) else cls(complex(value))

    def __add__(self, other: complex | float | "_ComplexDirectional") -> "_ComplexDirectional":
        other = self.coerce(other)
        return _ComplexDirectional(
            self.value + other.value,
            self.derivative + other.derivative,
        )

    __radd__ = __add__

    def __neg__(self) -> "_ComplexDirectional":
        return _ComplexDirectional(-self.value, -self.derivative)

    def __sub__(self, other: complex | float | "_ComplexDirectional") -> "_ComplexDirectional":
        return self + (-self.coerce(other))

    def __rsub__(self, other: complex | float | "_ComplexDirectional") -> "_ComplexDirectional":
        return self.coerce(other) - self

    def __mul__(self, other: complex | float | "_ComplexDirectional") -> "_ComplexDirectional":
        other = self.coerce(other)
        return _ComplexDirectional(
            self.value * other.value,
            self.derivative * other.value + self.value * other.derivative,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: complex | float | "_ComplexDirectional") -> "_ComplexDirectional":
        other = self.coerce(other)
        return _ComplexDirectional(
            self.value / other.value,
            (
                self.derivative * other.value
                - self.value * other.derivative
            ) / (other.value * other.value),
        )

    def exp(self) -> "_ComplexDirectional":
        value = cmath.exp(self.value)
        return _ComplexDirectional(value, value * self.derivative)


def _directional_distribution_multiply(
    left: dict[tuple[int, ...], _ComplexDirectional],
    right: dict[tuple[int, ...], _ComplexDirectional],
    caps: tuple[int, ...],
) -> dict[tuple[int, ...], _ComplexDirectional]:
    result: dict[tuple[int, ...], _ComplexDirectional] = {}
    for left_state, left_value in left.items():
        for right_state, right_value in right.items():
            state = _capped_add(left_state, right_state, caps)
            result[state] = result.get(state, _ComplexDirectional(0j)) + (
                left_value * right_value
            )
    return result


def _directional_exponential_factor(
    value: _ComplexDirectional,
    contribution: tuple[int, ...],
    caps: tuple[int, ...],
) -> dict[tuple[int, ...], _ComplexDirectional]:
    zero = tuple(0 for _ in caps)
    if not any(contribution):
        return {zero: value.exp()}
    saturation_count = max(
        (cap + increment - 1) // increment
        for cap, increment in zip(caps, contribution)
        if increment
    )
    result: dict[tuple[int, ...], _ComplexDirectional] = {}
    term = _ComplexDirectional(1 + 0j)
    partial = _ComplexDirectional(0j)
    for count in range(saturation_count):
        state = tuple(
            min(cap, count * increment)
            for cap, increment in zip(caps, contribution)
        )
        result[state] = result.get(state, _ComplexDirectional(0j)) + term
        partial += term
        term = term * value / (count + 1)
    saturated = tuple(
        cap if increment else 0
        for cap, increment in zip(caps, contribution)
    )
    result[saturated] = (
        result.get(saturated, _ComplexDirectional(0j))
        + value.exp()
        - partial
    )
    return result


def _directional_child_distribution(
    specification: C1FLabeledSpecification,
    root_profile: int,
    trees: Sequence[_ComplexDirectional],
) -> dict[tuple[int, ...], _ComplexDirectional]:
    grouped: dict[tuple[int, ...], _ComplexDirectional] = {}
    for coordinate, child_profile in enumerate(specification.transient_profiles):
        if not specification.successor[child_profile][root_profile]:
            continue
        contribution = specification._local_contribution(child_profile)
        grouped[contribution] = grouped.get(
            contribution,
            _ComplexDirectional(0j),
        ) + trees[coordinate]
    zero = tuple(0 for _ in specification.local_caps)
    distribution = {zero: _ComplexDirectional(1 + 0j)}
    for contribution, value in grouped.items():
        distribution = _directional_distribution_multiply(
            distribution,
            _directional_exponential_factor(
                value,
                contribution,
                specification.local_caps,
            ),
            specification.local_caps,
        )
    return distribution


def _directional_projected_value(
    specification: C1FLabeledSpecification,
    root_profile: int,
    distribution: dict[tuple[int, ...], _ComplexDirectional],
    extra: tuple[int, ...] | None = None,
) -> _ComplexDirectional:
    total = _ComplexDirectional(0j)
    accepted = specification._accepted_states(root_profile)
    for state, value in distribution.items():
        final_state = (
            state
            if extra is None
            else _capped_add(state, extra, specification.local_caps)
        )
        if final_state in accepted:
            total += value
    return total


def _directional_cycle_matrix(
    specification: C1FLabeledSpecification,
    active_profiles: Sequence[int],
    x: _ComplexDirectional,
    trees: Sequence[_ComplexDirectional],
) -> tuple[tuple[_ComplexDirectional, ...], ...]:
    distributions = {
        profile: _directional_child_distribution(
            specification,
            profile,
            trees,
        )
        for profile in active_profiles
    }
    return tuple(
        tuple(
            _ComplexDirectional(0j)
            if not specification.successor[previous][current]
            else x
            * specification.multiplicities[current]
            * _directional_projected_value(
                specification,
                current,
                distributions[current],
                specification._local_contribution(previous),
            )
            for current in active_profiles
        )
        for previous in active_profiles
    )


def _complex_determinant_direction(
    entries: Sequence[Sequence[_ComplexDirectional]],
) -> tuple[complex, complex]:
    size = len(entries)
    if size == 0:
        return 1 + 0j, 0j
    values = tuple(tuple(entry.value for entry in row) for row in entries)
    determinant = _matrix_determinant(values)
    derivative = 0j
    for row in range(size):
        for column in range(size):
            minor = tuple(
                tuple(
                    values[other_row][other_column]
                    for other_column in range(size)
                    if other_column != column
                )
                for other_row in range(size)
                if other_row != row
            )
            derivative += (
                (-1) ** (row + column)
                * _matrix_determinant(minor)
                * entries[row][column].derivative
            )
    return determinant, derivative


def _directional_finite_cycle_correction(
    specification: C1FLabeledSpecification,
    x: _ComplexDirectional,
    trees: Sequence[_ComplexDirectional],
) -> _ComplexDirectional:
    long_matrix = _directional_cycle_matrix(
        specification,
        specification._long_cycle_profiles(),
        x,
        trees,
    )
    correction = _ComplexDirectional(0j)
    for cycle_length in range(1, specification.depth + 1):
        visible_profiles = tuple(
            profile
            for profile, visible_length in enumerate(
                specification.visible_cycle_lengths
            )
            if visible_length == cycle_length
        )
        visible_matrix = _directional_cycle_matrix(
            specification,
            visible_profiles,
            x,
            trees,
        )
        correction += (
            _matrix_trace(_matrix_power(visible_matrix, cycle_length))
            - _matrix_trace(_matrix_power(long_matrix, cycle_length))
        ) / cycle_length
    return correction


def _project_periodic_labeled_orbit(
    specification: C1FLabeledSpecification,
    analysis: C1FLabeledAsymptotics,
    *,
    expansion_order: int = 0,
) -> C1FLabeledAsymptotics:
    """Compute and project a regular Laurent-Puiseux expansion."""

    if expansion_order < 0:
        raise ValueError("expansion_order must be non-negative")
    if not analysis.critical_components:
        return analysis
    orbit = _exact_periodic_orbit(
        specification,
        analysis.critical_component,
        unlabeled=False,
    )
    if orbit is None and expansion_order == 0:
        return analysis

    period = 1 if orbit is None else orbit.period
    tree_residues = (
        (0,) * specification.tree_dimension
        if orbit is None
        else orbit.tree_residues
    )
    admitted = (0,) if orbit is None else orbit.object_residues
    root = cmath.exp(2j * pi / period)
    phases = tuple(root**index for index in range(period))
    critical_count = len(analysis.critical_components)

    def solve_linear(
        matrix: Sequence[Sequence[complex]],
        rhs: Sequence[complex],
    ) -> tuple[complex, ...]:
        return _solve_linear_system(matrix, rhs)

    def expansion(half_order: int) -> tuple[tuple[tuple[float, float], ...], ...]:
        tree_order = max(1, critical_count + half_order)
        tree_coefficients = compute_tree_puiseux_coefficients(
            specification,
            complex(analysis.radius),
            tuple(complex(value) for value in analysis.tree_values),
            tuple(-complex(value) for value in analysis.tree_sqrt_coefficients),
            tuple(
                tuple(complex(value) for value in direction)
                for direction in analysis.critical_directions
            ),
            tuple(
                tuple(complex(value) for value in direction)
                for direction in analysis.critical_left_directions
            ),
            tree_order,
            solve_linear,
        )
        local = [
            labeled_local_laurent_coefficients(
                specification,
                complex(analysis.radius),
                tree_coefficients,
                denominator_valuation=critical_count,
                half_order=half_order,
                phase=phase,
                tree_residues=tree_residues,
            )
            for phase in phases
        ]
        return project_complex_laurent_orbit(
            local,
            phases,
            period=period,
            admitted_residues=admitted,
            leading_power=analysis.power,
            half_order=half_order,
            tolerance=1e-7,
        )

    used_order = expansion_order
    projected = expansion(used_order)
    while any(not projected[residue] for residue in admitted) and used_order < max(
        8,
        expansion_order,
    ):
        used_order += 1
        projected = expansion(used_order)
    if any(not projected[residue] for residue in admitted):
        raise ValueError(
            "No nonzero periodic term was found through Puiseux order "
            f"{used_order}; request a larger expansion_order"
        )
    residue_powers = tuple(
        terms[0][0] if terms else None
        for terms in projected
    )
    residue_constants = tuple(
        terms[0][1] if terms else 0.0
        for terms in projected
    )
    selected = next(
        (terms[0] for terms in projected if terms),
        None,
    )
    if selected is None:
        raise ValueError("The periodic orbit has no infinite admitted residue class")
    return replace(
        analysis,
        power=selected[0],
        leading_constant=selected[1],
        periodicity=period,
        residue_powers=residue_powers,
        residue_leading_constants=residue_constants,
        residue_expansions=tuple(
            tuple(C1FAsymptoticTerm(power, constant) for power, constant in terms)
            for terms in projected
        ),
    )


def _project_periodic_unlabeled_orbit(
    specification: C1FLabeledSpecification,
    analysis: C1FUnlabeledAsymptotics,
    critical_components: Sequence[Sequence[int]],
    *,
    tail_terms: int,
    tolerance: float,
    max_iterations: int,
    expansion_order: int = 0,
) -> C1FUnlabeledAsymptotics:
    """Continue and project the retained subcritical Pólya expansion."""

    flattened_component = tuple(
        index for component in critical_components for index in component
    )

    orbit = _exact_periodic_orbit(
        specification,
        flattened_component,
        unlabeled=True,
    )
    if orbit is None and expansion_order == 0:
        return analysis

    period = 1 if orbit is None else orbit.period
    tree_residues = (
        (0,) * specification.tree_dimension
        if orbit is None
        else orbit.tree_residues
    )
    admitted = (0,) if orbit is None else orbit.object_residues
    root = cmath.exp(2j * pi / period)
    phases = tuple(root**index for index in range(period))
    epsilon_values = (4e-4, 1e-4, 2.5e-5, 6.25e-6)
    sample_roots = tuple(sqrt(value) for value in epsilon_values)
    sampled_levels = tuple(
        _solve_unlabeled_tree_levels(
            specification,
            analysis.radius * (1.0 - epsilon),
            tail_terms,
            tolerance=min(tolerance, 1e-12),
            max_iterations=max(max_iterations, 120),
        )
        for epsilon in epsilon_values
    )

    def interpolate(values: Sequence[float]) -> tuple[complex, ...]:
        scale = sample_roots[0]
        scaled_roots = tuple(value / scale for value in sample_roots)
        matrix = tuple(
            tuple(complex(root_value**degree) for degree in range(len(values)))
            for root_value in scaled_roots
        )
        scaled = _solve_linear_system(matrix, tuple(complex(value) for value in values))
        return tuple(
            coefficient / scale**degree
            for degree, coefficient in enumerate(scaled)
        )

    level_values: list[tuple[complex, ...] | None] = [None]
    first_raw: tuple[complex, ...] | None = None
    for level in range(1, tail_terms + 1):
        coordinates = tuple(
            interpolate(
                tuple(
                    (sample[level] or tuple())[coordinate]
                    for sample in sampled_levels
                )
            )
            for coordinate in range(specification.tree_dimension)
        )
        level_values.append(tuple(value[0] for value in coordinates))
        if level == 1:
            first_raw = tuple(value[1] for value in coordinates)
    assert first_raw is not None

    real_level_values = [
        None
        if values is None
        else tuple(value.real for value in values)
        for values in level_values
    ]
    level_copy = list(real_level_values)
    _, tree_jacobian = _unlabeled_tree_map(
        specification,
        analysis.radius,
        level_copy,
        1,
        real_level_values[1] or tuple(),
    )
    right_directions, left_directions = _critical_kernel_bases(
        tree_jacobian,
        critical_components,
    )
    first_coefficients = tuple(
        sum(
            _dot(left, tuple(value.real for value in first_raw)) * right[index]
            for left, right in zip(left_directions, right_directions)
        )
        for index in range(specification.tree_dimension)
    )

    def solve_linear(
        matrix: Sequence[Sequence[complex]],
        rhs: Sequence[complex],
    ) -> tuple[complex, ...]:
        return _solve_linear_system(matrix, rhs)

    denominator_valuation = int(round(2.0 * (analysis.power + 1.0)))

    def expansion(half_order: int) -> tuple[tuple[tuple[float, float], ...], ...]:
        hierarchy = compute_unlabeled_tree_hierarchy(
            specification,
            complex(analysis.radius),
            level_values,
            tuple(complex(value) for value in first_coefficients),
            tuple(
                tuple(complex(value) for value in direction)
                for direction in right_directions
            ),
            tuple(
                tuple(complex(value) for value in direction)
                for direction in left_directions
            ),
            max(2, denominator_valuation + half_order),
            solve_linear,
        )
        local = [
            unlabeled_local_laurent_coefficients(
                specification,
                complex(analysis.radius),
                hierarchy,
                denominator_valuation=denominator_valuation,
                half_order=half_order,
                phase=phase,
                tree_residues=tree_residues,
            )
            for phase in phases
        ]
        local[0][-denominator_valuation] = complex(
            analysis.leading_constant * gamma(analysis.power + 1.0)
        )
        return project_complex_laurent_orbit(
            local,
            phases,
            period=period,
            admitted_residues=admitted,
            leading_power=analysis.power,
            half_order=half_order,
            tolerance=2e-5,
        )

    used_order = expansion_order
    projected = expansion(used_order)
    while any(not projected[residue] for residue in admitted) and used_order < max(
        8,
        expansion_order,
    ):
        used_order += 1
        projected = expansion(used_order)
    if any(not projected[residue] for residue in admitted):
        raise ValueError(
            "No nonzero periodic Pólya term was found through Puiseux order "
            f"{used_order}"
        )
    residue_powers = tuple(terms[0][0] if terms else None for terms in projected)
    residue_constants = tuple(terms[0][1] if terms else 0.0 for terms in projected)
    selected = next((terms[0] for terms in projected if terms), None)
    if selected is None:
        raise ValueError("The periodic Pólya orbit has no infinite residue class")
    return replace(
        analysis,
        power=selected[0],
        leading_constant=selected[1],
        periodicity=period,
        residue_powers=residue_powers,
        residue_leading_constants=residue_constants,
        residue_expansions=tuple(
            tuple(C1FAsymptoticTerm(power, constant) for power, constant in terms)
            for terms in projected
        ),
    )


def _integer_matrix_multiply(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    size = len(left)
    return tuple(
        tuple(
            sum(left[row][middle] * right[middle][column] for middle in range(size))
            for column in range(size)
        )
        for row in range(size)
    )


def _determinant_polynomial(
    matrix: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """Return the coefficients of ``det(I - x A)`` exactly.

    Faddeev--LeVerrier expresses the coefficients through the traces of
    ``A, ..., A**d``.  Keeping the recurrence in ``Fraction`` also guards the
    exact divisions, although an integral matrix has integral output.
    """

    size = len(matrix)
    if size == 0:
        return (1,)
    power = tuple(tuple(int(value) for value in row) for row in matrix)
    traces: list[int] = []
    for exponent in range(1, size + 1):
        traces.append(sum(power[index][index] for index in range(size)))
        if exponent != size:
            power = _integer_matrix_multiply(power, matrix)
    coefficients = [Fraction(1)]
    for degree in range(1, size + 1):
        coefficient = -sum(
            coefficients[degree - exponent] * traces[exponent - 1]
            for exponent in range(1, degree + 1)
        ) / degree
        coefficients.append(coefficient)
    if any(coefficient.denominator != 1 for coefficient in coefficients):
        raise ValueError("Internal error: a determinant coefficient is non-integral")
    return tuple(int(coefficient) for coefficient in coefficients)


def _tree_independent_cycle_polynomials(
    specification: C1FLabeledSpecification,
) -> tuple[tuple[int, ...], tuple[Fraction, ...]]:
    """Return ``Q`` and ``H`` with ``F(x) = exp(H(x)) / Q(x)``."""

    zero = tuple(0.0 for _ in range(specification.tree_dimension))

    def unit_matrix(profiles: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        values = specification.cycle_matrix(profiles, 1.0, zero)
        rounded = tuple(
            tuple(int(round(value)) for value in row)
            for row in values
        )
        if any(
            abs(values[row][column] - rounded[row][column]) > 1e-12
            for row in range(len(values))
            for column in range(len(values))
        ):
            raise ValueError("Cycle-independent transfers are not integral")
        return rounded

    long_matrix = unit_matrix(specification._long_cycle_profiles())
    denominator = _determinant_polynomial(long_matrix)
    correction = [Fraction(0) for _ in range(specification.depth + 1)]
    long_power = tuple(
        tuple(1 if row == column else 0 for column in range(len(long_matrix)))
        for row in range(len(long_matrix))
    )
    for cycle_length in range(1, specification.depth + 1):
        if cycle_length == 1:
            long_power = long_matrix
        else:
            long_power = _integer_matrix_multiply(long_power, long_matrix)
        visible_profiles = tuple(
            profile
            for profile, visible_length in enumerate(
                specification.visible_cycle_lengths
            )
            if visible_length == cycle_length
        )
        visible_matrix = unit_matrix(visible_profiles)
        visible_power = tuple(
            tuple(1 if row == column else 0 for column in range(len(visible_matrix)))
            for row in range(len(visible_matrix))
        )
        for _ in range(cycle_length):
            visible_power = _integer_matrix_multiply(
                visible_power,
                visible_matrix,
            )
        visible_trace = sum(
            visible_power[index][index]
            for index in range(len(visible_power))
        )
        long_trace = sum(
            long_power[index][index]
            for index in range(len(long_power))
        )
        correction[cycle_length] = Fraction(
            visible_trace - long_trace,
            cycle_length,
        )
    return denominator, tuple(correction)


def _polynomial_value(coefficients: Sequence[complex | float | int], value: complex) -> complex:
    result = 0j
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def _polynomial_derivative(
    coefficients: Sequence[int],
    order: int,
) -> tuple[int, ...]:
    result = tuple(coefficients)
    for _ in range(order):
        result = tuple(
            degree * result[degree]
            for degree in range(1, len(result))
        )
    return result


def _polynomial_multiply(
    left: Sequence[complex],
    right: Sequence[complex],
) -> tuple[complex, ...]:
    result = [0j] * (len(left) + len(right) - 1)
    for left_degree, left_value in enumerate(left):
        for right_degree, right_value in enumerate(right):
            result[left_degree + right_degree] += left_value * right_value
    return tuple(result)


def _binomial_polynomial(order: int) -> tuple[complex, ...]:
    """Return coefficients of ``binomial(n + order - 1, order - 1)``."""

    result: tuple[complex, ...] = (1 + 0j,)
    for offset in range(1, order):
        result = tuple(
            value / offset
            for value in _polynomial_multiply(result, (offset + 0j, 1 + 0j))
        )
    return result


def _local_pole_polynomial(
    denominator: Sequence[int],
    correction: Sequence[Fraction],
    singularity: complex,
    order: int,
) -> tuple[complex, ...]:
    """Return the coefficient polynomial from one pole ``singularity``."""

    required = order - 1

    def compose(coefficients: Sequence[complex | int | Fraction], limit: int) -> list[complex]:
        result = [0j] * (limit + 1)
        for degree, coefficient in enumerate(coefficients):
            converted = complex(float(coefficient))
            for local_degree in range(min(degree, limit) + 1):
                result[local_degree] += (
                    converted
                    * singularity**degree
                    * ((-1) ** local_degree)
                    * factorial(degree)
                    / (factorial(local_degree) * factorial(degree - local_degree))
                )
        return result

    denominator_u = compose(denominator, 2 * order - 1)
    correction_u = compose(correction, required)
    numerator_u = [0j] * (required + 1)
    numerator_u[0] = cmath.exp(correction_u[0])
    for degree in range(1, required + 1):
        numerator_u[degree] = sum(
            local_degree
            * correction_u[local_degree]
            * numerator_u[degree - local_degree]
            for local_degree in range(1, degree + 1)
        ) / degree
    leading_denominator = denominator_u[order]
    if abs(leading_denominator) == 0.0:
        raise ValueError("The local pole order is inconsistent with the denominator")
    quotient = [0j] * (required + 1)
    for degree in range(required + 1):
        quotient[degree] = (
            numerator_u[degree]
            - sum(
                denominator_u[order + offset] * quotient[degree - offset]
                for offset in range(1, degree + 1)
            )
        ) / leading_denominator
    polynomial = [0j] * order
    for pole_order in range(1, order + 1):
        laurent_coefficient = quotient[order - pole_order]
        binomial = _binomial_polynomial(pole_order)
        for degree, coefficient in enumerate(binomial):
            polynomial[degree] += laurent_coefficient * coefficient
    return tuple(polynomial)


def _lcm(left: int, right: int) -> int:
    return left // gcd(left, right) * right


def _solve_labeled_trees_at_x(
    specification: C1FLabeledSpecification,
    x: float,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[float, ...]:
    """Return the minimal analytic tree solution at a fixed positive ``x``."""

    dimension = specification.tree_dimension
    trees = tuple(0.0 for _ in range(dimension))
    for _ in range(24):
        trees = specification.tree_rhs(x, trees)
    for _ in range(max(40, 2 * max_iterations)):
        phi = specification.tree_rhs(x, trees)
        residual = tuple(value - tree for value, tree in zip(phi, trees))
        residual_norm = max((abs(value) for value in residual), default=0.0)
        if residual_norm <= max(1e-13, tolerance * 0.01):
            return trees
        jacobian = specification.tree_jacobian(x, trees)
        operator = tuple(
            tuple(
                (1.0 if row == column else 0.0) - jacobian[row][column]
                for column in range(dimension)
            )
            for row in range(dimension)
        )
        update = _solve_linear_system(operator, residual)
        step = 1.0
        while step >= 2.0**-30:
            candidate = tuple(
                tree + step * delta
                for tree, delta in zip(trees, update)
            )
            if min(candidate, default=0.0) < 0.0:
                step /= 2.0
                continue
            candidate_phi = specification.tree_rhs(x, candidate)
            candidate_norm = max(
                (
                    abs(value - tree)
                    for value, tree in zip(candidate_phi, candidate)
                ),
                default=0.0,
            )
            if candidate_norm < residual_norm:
                trees = candidate
                break
            step /= 2.0
        else:
            raise ValueError("The fixed-x labeled tree solve did not decrease")
    raise ValueError("The fixed-x labeled tree solve did not converge")


def _analyze_analytic_tree_cycle_pole(
    specification: C1FLabeledSpecification,
    *,
    tolerance: float,
    max_iterations: int,
) -> C1FLabeledAsymptotics | None:
    """Find a simple cycle pole reached before the tree subsystem is critical."""

    long_profiles = specification._long_cycle_profiles()
    if not long_profiles or long_profiles == specification.transient_profiles:
        return None

    def point(x: float) -> tuple[tuple[float, ...], float]:
        trees = _solve_labeled_trees_at_x(
            specification,
            x,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        return trees, _perron_eigenvalue(
            specification.long_cycle_matrix(x, trees)
        )

    lower = 0.0
    upper = 1.0
    lower_point: tuple[tuple[float, ...], float] | None = None
    for _ in range(72):
        middle = (lower + upper) / 2.0
        try:
            candidate = point(middle)
        except (OverflowError, ValueError, ZeroDivisionError):
            candidate = None
        if candidate is not None and candidate[1] < 1.0:
            lower = middle
            lower_point = candidate
        else:
            upper = middle
    if lower_point is None:
        return None
    radius = (lower + upper) / 2.0
    try:
        trees = _solve_labeled_trees_at_x(
            specification,
            radius,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    except ValueError:
        trees = lower_point[0]
        radius = lower
    transfer = specification.long_cycle_matrix(radius, trees)
    if abs(_perron_eigenvalue(transfer) - 1.0) > 2e-6:
        return None
    tree_jacobian = specification.tree_jacobian(radius, trees)
    if _perron_eigenvalue(tree_jacobian) >= 1.0 - 2e-6:
        return None

    support = tuple(tuple(value > 0.0 for value in row) for row in transfer)
    components = _strongly_connected_components(support)
    critical_components = tuple(
        component
        for component in components
        if abs(
            _perron_eigenvalue(_principal_submatrix(transfer, component)) - 1.0
        )
        <= 2e-6
    )
    if len(critical_components) != 1:
        raise ValueError(
            "Analytic-tree cycle poles with several critical components require "
            "a higher-order pole expansion"
        )
    critical_component = critical_components[0]
    if _matrix_period(
        _principal_submatrix(transfer, critical_component)
    ) != 1:
        raise ValueError(
            "Periodic analytic-tree cycle poles require residue-class projection"
        )

    epsilon_values = (4e-4, 1e-4, 2.5e-5)
    scaled_determinants = []
    for epsilon in epsilon_values:
        sample_x = radius * (1.0 - epsilon)
        sample_trees = _solve_labeled_trees_at_x(
            specification,
            sample_x,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        scaled_determinants.append(
            specification.long_cycle_determinant(sample_x, sample_trees)
            / epsilon
        )
    eta = 0.0
    for index, value in enumerate(scaled_determinants):
        basis = 1.0
        for other, epsilon in enumerate(epsilon_values):
            if other != index:
                basis *= -epsilon / (epsilon_values[index] - epsilon)
        eta += value * basis
    if eta <= 0.0:
        raise ValueError("The analytic-tree cycle-pole coefficient is not positive")

    right_local = _kernel_vector(
        _principal_submatrix(transfer, critical_component)
    )
    left_local = _kernel_vector(
        _transpose(_principal_submatrix(transfer, critical_component))
    )
    normalization = _dot(left_local, right_local)
    left_local = tuple(value / normalization for value in left_local)
    right = [0.0] * len(transfer)
    left = [0.0] * len(transfer)
    for local, global_index in enumerate(critical_component):
        right[global_index] = right_local[local]
        left[global_index] = left_local[local]

    correction = specification.finite_cycle_correction(radius, trees)
    leading_constant = exp(correction) / eta
    return C1FLabeledAsymptotics(
        radius=radius,
        exponential_growth=1.0 / radius,
        power=0.0,
        leading_constant=leading_constant,
        tree_values=trees,
        tree_sqrt_coefficients=tuple(0.0 for _ in trees),
        finite_cycle_correction=correction,
        denominator_sqrt_coefficient=0.0,
        perron_right=tuple(right),
        perron_left=tuple(left),
        critical_residual=abs(
            specification.long_cycle_determinant(radius, trees)
        ),
        eventual_start=specification.eventual_start,
        regime="analytic-tree-cycle-pole",
        cycle_pole_order=1,
        pole_denominator_coefficient=eta,
        critical_component=critical_component,
        critical_components=(critical_component,),
    )


def _analyze_tree_independent_cycle_pole(
    specification: C1FLabeledSpecification,
    *,
    tolerance: float,
) -> C1FLabeledAsymptotics | None:
    """Analyze the exact exponential-polynomial cycle-only representation."""

    if specification.cycle_dimension == 0:
        return None
    if not specification.cycle_transfers_are_tree_independent():
        return None

    zero = tuple(0.0 for _ in range(specification.tree_dimension))
    unit_transfer = specification.long_cycle_matrix(1.0, zero)
    support = tuple(tuple(value > 0.0 for value in row) for row in unit_transfer)
    components = _strongly_connected_components(support)
    cyclic_components = tuple(
        component
        for component in components
        if len(component) > 1 or support[component[0]][component[0]]
    )
    if not cyclic_components:
        return None
    component_roots = tuple(
        _perron_eigenvalue(_principal_submatrix(unit_transfer, component))
        for component in cyclic_components
    )
    perron_root = max(component_roots)
    if perron_root <= 0.0:
        return None
    critical_components = tuple(
        component
        for component, component_root in zip(cyclic_components, component_roots)
        if abs(component_root - perron_root)
        <= max(1e-9, 100.0 * tolerance) * max(1.0, perron_root)
    )
    component_periods = tuple(
        _matrix_period(_principal_submatrix(unit_transfer, component))
        for component in critical_components
    )
    periodicity = 1
    for period in component_periods:
        periodicity = _lcm(periodicity, period)

    radius = 1.0 / perron_root
    denominator, correction_polynomial = _tree_independent_cycle_polynomials(
        specification
    )
    pole_orders = []
    pole_coefficients: list[complex] = []
    phases = []
    for phase_index in range(periodicity):
        phase = cmath.exp(2j * pi * phase_index / periodicity)
        phases.append(phase)
        order = sum(
            phase_index % (periodicity // period) == 0
            for period in component_periods
        )
        pole_orders.append(order)
        if order == 0:
            pole_coefficients.append(0j)
            continue
        singularity = radius * phase
        derivative = _polynomial_value(
            _polynomial_derivative(denominator, order),
            singularity,
        )
        eta = derivative * ((-singularity) ** order) / factorial(order)
        if abs(eta) <= tolerance:
            raise ValueError(
                "The cycle denominator has a higher multiplicity than its exact "
                "critical-component decomposition predicts"
            )
        correction_at_pole = _polynomial_value(
            correction_polynomial,
            singularity,
        )
        pole_coefficients.append(cmath.exp(correction_at_pole) / eta)

    local_polynomials = tuple(
        tuple()
        if order == 0
        else _local_pole_polynomial(
            denominator,
            correction_polynomial,
            radius * phase,
            order,
        )
        for phase, order in zip(phases, pole_orders)
    )
    residue_powers: list[float | None] = []
    residue_constants: list[float] = []
    residue_polynomials: list[tuple[float, ...]] = []
    for residue in range(periodicity):
        combined = [0j] * max(pole_orders)
        scale = 1.0
        for local_polynomial, phase in zip(local_polynomials, phases):
            phase_factor = phase ** (-residue)
            scale += sum(abs(value) for value in local_polynomial)
            for degree, value in enumerate(local_polynomial):
                combined[degree] += phase_factor * value
        threshold = max(1e-10, 100.0 * tolerance) * scale
        real_coefficients = []
        for coefficient in combined:
            if abs(coefficient.imag) > max(1e-9, 1000.0 * tolerance) * scale:
                raise ValueError("Periodic pole contributions did not combine to a real value")
            real_coefficients.append(
                0.0 if abs(coefficient.real) <= threshold else coefficient.real
            )
        while real_coefficients and real_coefficients[-1] == 0.0:
            real_coefficients.pop()
        if real_coefficients and real_coefficients[-1] < 0.0:
            raise ValueError("A residue-class leading constant is negative")
        selected_power = (
            float(len(real_coefficients) - 1)
            if real_coefficients
            else None
        )
        selected_constant = real_coefficients[-1] if real_coefficients else 0.0
        residue_powers.append(selected_power)
        residue_constants.append(selected_constant)
        residue_polynomials.append(tuple(real_coefficients))

    positive_order = pole_orders[0]
    positive_derivative = _polynomial_value(
        _polynomial_derivative(denominator, positive_order),
        complex(radius),
    ).real
    denominator_coefficient = (
        positive_derivative * ((-radius) ** positive_order) / factorial(positive_order)
    )
    correction = float(_polynomial_value(correction_polynomial, complex(radius)).real)
    leading_constant = residue_constants[0]
    power = residue_powers[0]
    if power is None:
        first_nonzero = next(
            (
                (candidate_power, candidate_constant)
                for candidate_power, candidate_constant in zip(
                    residue_powers,
                    residue_constants,
                )
                if candidate_power is not None
            ),
            None,
        )
        if first_nonzero is None:
            return None
        power, leading_constant = first_nonzero
    root_residual = abs(_polynomial_value(denominator, complex(radius)))
    return C1FLabeledAsymptotics(
        radius=radius,
        exponential_growth=perron_root,
        power=power,
        leading_constant=leading_constant,
        tree_values=tuple(),
        tree_sqrt_coefficients=tuple(),
        finite_cycle_correction=correction,
        denominator_sqrt_coefficient=0.0,
        perron_right=tuple(),
        perron_left=tuple(),
        critical_residual=root_residual,
        eventual_start=specification.eventual_start,
        periodicity=periodicity,
        regime="cycle-pole",
        cycle_pole_order=positive_order,
        pole_denominator_coefficient=denominator_coefficient,
        critical_component=tuple(
            index
            for component in critical_components
            for index in component
        ),
        critical_components=critical_components,
        residue_powers=tuple(residue_powers),
        residue_leading_constants=tuple(residue_constants),
        residue_polynomial_coefficients=tuple(residue_polynomials),
        residue_expansions=tuple(
            tuple(
                C1FAsymptoticTerm(float(degree), coefficient)
                for degree, coefficient in reversed(tuple(enumerate(polynomial)))
                if coefficient != 0.0
            )
            for polynomial in residue_polynomials
        ),
    )


def _locally_active_types(monadic: object) -> tuple[int, ...]:
    predecessor_plan = monadic.predecessor_plan
    local_filters = _entailed_local_filters(monadic.formula)
    return tuple(
        guarded_type.id
        for guarded_type in predecessor_plan.types
        if all(
            _guarded_type_satisfies_local(
                guarded_type,
                local_filter,
                monadic.profiles,
                monadic.predicate_index,
                predecessor_plan.predecessor_index,
            )
            for local_filter in local_filters
        )
    )


def _perturbed_singularity_radius(
    specification: C1FLabeledSpecification,
    *,
    tolerance: float,
    max_iterations: int,
) -> float:
    """Return the positive singularity of a fugacity-weighted specification."""

    if specification.cycle_transfers_are_tree_independent():
        transfer = specification.long_cycle_matrix(
            1.0,
            tuple(0.0 for _ in range(specification.tree_dimension)),
        )
        perron_root = _perron_eigenvalue(transfer)
        if perron_root <= 0.0:
            raise ValueError("The perturbed cycle transfer has no positive pole")
        return 1.0 / perron_root
    return analyze_c1f_labeled_asymptotics(
        specification,
        tolerance=min(tolerance, 1e-12),
        max_iterations=max(max_iterations, 120),
    ).radius


def _analyze_single_global_marker(
    compiled: CompiledC1F,
    *,
    tolerance: float,
    max_iterations: int,
    certify: bool,
    precision_bits: int,
    max_precision_bits: int | None,
) -> C1FLabeledAsymptotics:
    """Analyze one capped global counter through its zero-state perturbation."""

    monadic = compiled.branches[0].monadic
    plan = monadic.cardinality_plan
    if len(plan.caps) != 1:
        raise ValueError(
            "Automatic capped-state asymptotics currently support one dynamic counter"
        )
    active_types = _locally_active_types(monadic)
    if not active_types:
        raise ValueError("The local conditions reject every guarded type")
    cap = plan.caps[0]
    eventual_start = max(_largest_count_threshold(monadic.formula) + 1, cap)
    accepted_states = tuple(
        state
        for state in range(cap + 1)
        if plan.satisfies(monadic.formula, (state,), eventual_start)
    )
    if not accepted_states:
        raise ValueError("The global counter rejects every eventual marker state")

    full_specification = _build_labeled_specification(
        monadic,
        active_types,
        eventual_start=eventual_start,
    )
    if cap in accepted_states:
        result = analyze_c1f_labeled_asymptotics(
            full_specification,
            tolerance=tolerance,
            max_iterations=max_iterations,
            certify=certify,
            precision_bits=precision_bits,
            max_precision_bits=max_precision_bits,
        )
        return replace(
            result,
            regime=f"capped-global-saturated:{result.regime}",
            global_marker_cap=cap,
            accepted_marker_states=accepted_states,
            global_marker_caps=(cap,),
            accepted_global_states=tuple((state,) for state in accepted_states),
        )

    zero_types = tuple(
        type_id
        for type_id in active_types
        if plan.profile_states[type_id] == (0,)
    )
    marked_types = tuple(
        type_id
        for type_id in active_types
        if plan.profile_states[type_id] == (1,)
    )
    if not zero_types or not marked_types:
        raise ValueError(
            "The global counter has no nontrivial zero/marked type decomposition"
        )
    core_specification = _build_labeled_specification(
        monadic,
        zero_types,
        eventual_start=eventual_start,
    )
    core = analyze_c1f_labeled_asymptotics(
        core_specification,
        tolerance=tolerance,
        max_iterations=max_iterations,
        certify=certify,
        precision_bits=precision_bits,
        max_precision_bits=max_precision_bits,
    )

    def perturbed_radius(fugacity: float) -> float:
        scales = tuple(
            fugacity if plan.profile_states[type_id] == (1,) else 1.0
            for type_id in active_types
        )
        specification = _build_labeled_specification(
            monadic,
            active_types,
            eventual_start=eventual_start,
            multiplicity_scales=scales,
        )
        return _perturbed_singularity_radius(
            specification,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    step = 8e-4
    coarse = log(core.radius / perturbed_radius(step)) / step
    fine = log(core.radius / perturbed_radius(step / 2.0)) / (step / 2.0)
    finer = log(core.radius / perturbed_radius(step / 4.0)) / (step / 4.0)
    first = 2.0 * fine - coarse
    second = 2.0 * finer - fine
    marker_derivative = (4.0 * second - first) / 3.0
    if marker_derivative <= max(1e-8, 100.0 * tolerance):
        raise ValueError(
            "The first marker perturbation does not move the dominant singularity; "
            "a higher-order capped-state expansion is required"
        )
    highest_state = max(accepted_states)
    leading_constant = (
        core.leading_constant
        * marker_derivative**highest_state
        / factorial(highest_state)
    )
    certificate = core.certificate
    if certify:
        from .interval_asymptotics import certify_single_global_marker

        certificate = certify_single_global_marker(
            monadic,
            active_types,
            zero_types,
            core,
            highest_state,
            precision_bits=precision_bits,
            max_precision_bits=max_precision_bits,
        )
    return replace(
        core,
        power=core.power + highest_state,
        leading_constant=leading_constant,
        eventual_start=eventual_start,
        regime=f"capped-global-unsaturated:{core.regime}",
        global_marker_cap=cap,
        accepted_marker_states=accepted_states,
        marker_growth_derivative=marker_derivative,
        global_marker_caps=(cap,),
        accepted_global_states=tuple((state,) for state in accepted_states),
        marker_growth_derivatives=(marker_derivative,),
        certificate=certificate,
    )


def _analyze_multiple_global_markers(
    compiled: CompiledC1F,
    *,
    tolerance: float,
    max_iterations: int,
    certify: bool,
    precision_bits: int,
    max_precision_bits: int | None,
    expansion_order: int = 0,
) -> C1FLabeledAsymptotics:
    """Decompose accepted capped states into saturated faces."""

    monadic = compiled.branches[0].monadic
    plan = monadic.cardinality_plan
    caps = plan.caps
    active_types = _locally_active_types(monadic)
    eventual_start = max(
        _largest_count_threshold(monadic.formula) + 1,
        max(caps),
    )
    all_states = tuple(product(*(range(cap + 1) for cap in caps)))
    accepted_states = tuple(
        state
        for state in all_states
        if plan.satisfies(monadic.formula, state, eventual_start)
    )
    if not accepted_states:
        raise ValueError("The global counters reject every eventual state")

    face_results: list[C1FLabeledAsymptotics] = []
    for accepted_state in accepted_states:
        saturated = tuple(
            index
            for index, (value, cap) in enumerate(zip(accepted_state, caps))
            if value == cap
        )
        remaining = tuple(index for index in range(len(caps)) if index not in saturated)
        base_types = tuple(
            type_id
            for type_id in active_types
            if all(plan.profile_states[type_id][index] == 0 for index in remaining)
        )
        if not base_types:
            continue
        if any(
            not any(plan.profile_states[type_id][index] for type_id in base_types)
            for index in saturated
        ):
            continue
        try:
            base_specification = _build_labeled_specification(
                monadic,
                base_types,
                eventual_start=eventual_start,
            )
            base = analyze_c1f_labeled_asymptotics(
                base_specification,
                tolerance=tolerance,
                max_iterations=max_iterations,
                certify=certify,
                precision_bits=precision_bits,
                max_precision_bits=max_precision_bits,
                expansion_order=expansion_order,
            )
        except ValueError:
            continue

        fixed_degree = sum(accepted_state[index] for index in remaining)
        if expansion_order and fixed_degree:
            raise ValueError(
                "Higher-order expansions after fixed global-marker extraction "
                "require the bivariate marker recurrence"
            )
        if (
            fixed_degree
            and base.regime == "cycle-pole"
            and base.cycle_pole_order != 1
        ):
            raise ValueError(
                "Capped-state perturbation of several coincident cycle poles "
                "requires their separate marker derivatives"
            )

        derivatives = [0.0] * len(caps)
        viable = True
        for marker_index in saturated:
            restricted_types = tuple(
                type_id
                for type_id in base_types
                if plan.profile_states[type_id][marker_index] == 0
            )
            if not restricted_types:
                continue
            try:
                restricted = analyze_c1f_labeled_asymptotics(
                    _build_labeled_specification(
                        monadic,
                        restricted_types,
                        eventual_start=eventual_start,
                    ),
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
            except ValueError:
                continue
            if not base.radius < restricted.radius * (
                1.0 - max(1e-8, 100.0 * tolerance)
            ):
                viable = False
                break
        if not viable:
            continue
        for marker_index in remaining:
            if accepted_state[marker_index] == 0:
                continue
            perturbation_types = tuple(
                type_id
                for type_id in active_types
                if all(
                    plan.profile_states[type_id][other] == 0
                    for other in remaining
                    if other != marker_index
                )
            )
            if not any(
                plan.profile_states[type_id][marker_index]
                for type_id in perturbation_types
            ):
                viable = False
                break

            def perturbed_radius(fugacity: float) -> float:
                scales = tuple(
                    fugacity
                    if plan.profile_states[type_id][marker_index]
                    else 1.0
                    for type_id in perturbation_types
                )
                specification = _build_labeled_specification(
                    monadic,
                    perturbation_types,
                    eventual_start=eventual_start,
                    multiplicity_scales=scales,
                )
                return _perturbed_singularity_radius(
                    specification,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )

            step = 8e-4
            coarse = log(base.radius / perturbed_radius(step)) / step
            fine = log(base.radius / perturbed_radius(step / 2.0)) / (step / 2.0)
            finer = log(base.radius / perturbed_radius(step / 4.0)) / (step / 4.0)
            first = 2.0 * fine - coarse
            second = 2.0 * finer - fine
            derivatives[marker_index] = (4.0 * second - first) / 3.0
            if derivatives[marker_index] <= max(1e-8, 100.0 * tolerance):
                viable = False
                break
        if not viable:
            continue

        factor = 1.0
        for marker_index in remaining:
            count = accepted_state[marker_index]
            factor *= derivatives[marker_index] ** count / factorial(count)
        certificate = base.certificate
        if certify and fixed_degree:
            from .interval_asymptotics import certify_global_marker_face

            certificate = certify_global_marker_face(
                monadic,
                active_types,
                base_types,
                remaining,
                accepted_state,
                base,
                precision_bits=precision_bits,
                max_precision_bits=max_precision_bits,
            )
        face_results.append(
            replace(
                base,
                power=base.power + fixed_degree,
                leading_constant=base.leading_constant * factor,
                eventual_start=eventual_start,
                regime=f"capped-global-face:{base.regime}",
                global_marker_caps=caps,
                accepted_global_states=(accepted_state,),
                marker_growth_derivatives=tuple(derivatives),
                residue_powers=tuple(
                    None if power is None else power + fixed_degree
                    for power in base.residue_powers
                ),
                residue_leading_constants=tuple(
                    constant * factor
                    for constant in base.residue_leading_constants
                ),
                residue_polynomial_coefficients=tuple(),
                residue_expansions=_shift_expansions(
                    base.residue_expansions,
                    fixed_degree,
                    factor,
                ),
                certificate=certificate,
            )
        )

    if not face_results:
        raise ValueError(
            "No accepted capped-state face has a supported regular asymptotic"
        )
    if len(face_results) == 1:
        return face_results[0]
    probe = max(100, eventual_start)
    dominant = max(face_results, key=lambda result: result.log_estimate(probe))
    return replace(
        dominant,
        regime="capped-global-face-sum",
        accepted_global_states=accepted_states,
        summands=tuple(face_results),
        certificate=None,
    )


def analyze_c1f_labeled_asymptotics(
    formula: str | C1FFormula | CompiledC1F | C1FLabeledSpecification,
    *,
    function_symbol: str = "f",
    profile_pruning: bool = True,
    twin_compression: bool = True,
    tolerance: float = 1e-10,
    max_iterations: int = 80,
    certify: bool = False,
    precision_bits: int = 192,
    max_precision_bits: int | None = None,
    expansion_order: int = 0,
) -> C1FLabeledAsymptotics:
    """Compute the leading labeled asymptotic in a supported regular regime.

    Complete-compatibility systems use an exact rank-one reduction.  Other
    strongly connected systems use a damped Newton solve for the characteristic
    point, followed by the positive left and right Perron vectors.  The finite
    short-cycle correction supplies the analytic numerator at the singularity.
    Guarded systems whose cycle transfer is independent of all transient trees
    are recognized separately as exact finite cycle poles, including periodic
    and higher-order cases.  Set ``certify=True`` to attach an optional
    Arb/Krawczyk interval certificate to the result.  ``expansion_order``
    requests successive half-orders after the leading term.
    """

    if expansion_order < 0:
        raise ValueError("expansion_order must be non-negative")
    if isinstance(formula, C1FLabeledSpecification):
        specification = formula
    else:
        compiled = (
            formula
            if isinstance(formula, CompiledC1F)
            else compile_c1f_fomc(
                formula,
                function_symbol=function_symbol,
                profile_pruning=profile_pruning,
                twin_compression=twin_compression,
            )
        )
        if len(compiled.branches) != 1:
            summands = []
            for branch in compiled.branches:
                branch_compiled = replace(compiled, branches=(branch,))
                try:
                    summands.append(
                        analyze_c1f_labeled_asymptotics(
                            branch_compiled,
                            tolerance=tolerance,
                            max_iterations=max_iterations,
                            certify=certify,
                            precision_bits=precision_bits,
                            max_precision_bits=max_precision_bits,
                            expansion_order=expansion_order,
                        )
                    )
                except ValueError as exc:
                    if "eventually rejecting" not in str(exc):
                        raise
            if not summands:
                raise ValueError("Every nullary branch is eventually rejecting")
            probe = max(100, *(summand.eventual_start for summand in summands))
            dominant = max(summands, key=lambda summand: summand.log_estimate(probe))
            return replace(
                dominant,
                regime="nullary-branch-sum",
                eventual_start=max(
                    summand.eventual_start for summand in summands
                ),
                summands=tuple(summands),
                certificate=None,
            )
        branch = compiled.branches[0]
        if branch.higher_relation_arities or branch.trace_predicate_count:
            monadic_branch = replace(
                branch,
                higher_relation_arities=tuple(),
                trace_predicate_count=0,
            )
            monadic_compiled = replace(
                compiled,
                has_higher_arity_relations=False,
                branches=(monadic_branch,),
            )
            core = analyze_c1f_labeled_asymptotics(
                monadic_compiled,
                tolerance=tolerance,
                max_iterations=max_iterations,
                certify=certify,
                precision_bits=precision_bits,
                max_precision_bits=max_precision_bits,
                expansion_order=expansion_order,
            )
            certificate = core.certificate
            if certificate is not None:
                certificate = replace(
                    certificate,
                    regime=f"higher-arity-transfer:{certificate.regime}",
                    verified_properties=certificate.verified_properties
                    + ("exact labeled higher-arity multiplicity transfer",),
                )
            return replace(
                core,
                regime=f"higher-arity-transfer:{core.regime}",
                higher_arity_exponents=branch.higher_relation_arities,
                trace_predicate_count=branch.trace_predicate_count,
                certificate=certificate,
            )
        active_marker_states = {
            branch.monadic.cardinality_plan.profile_states[type_id]
            for type_id in _locally_active_types(branch.monadic)
        }
        if (
            branch.monadic.cardinality_plan.caps
            and len(active_marker_states) > 1
        ):
            return _analyze_multiple_global_markers(
                compiled,
                tolerance=tolerance,
                max_iterations=max_iterations,
                certify=certify,
                precision_bits=precision_bits,
                max_precision_bits=max_precision_bits,
                expansion_order=expansion_order,
            )
        specification = compile_c1f_labeled_specification(compiled)

    def finish(result: C1FLabeledAsymptotics) -> C1FLabeledAsymptotics:
        if not certify:
            return result
        from .interval_asymptotics import certify_c1f_labeled_asymptotics

        certificate = certify_c1f_labeled_asymptotics(
            specification,
            result,
            precision_bits=precision_bits,
            max_precision_bits=max_precision_bits,
        )
        return replace(result, certificate=certificate)

    if tolerance <= 0.0:
        raise ValueError("The critical-point tolerance must be positive")
    if max_iterations <= 0:
        raise ValueError("The critical-point iteration limit must be positive")

    cycle_pole = _analyze_tree_independent_cycle_pole(
        specification,
        tolerance=tolerance,
    )
    if cycle_pole is not None:
        return finish(cycle_pole)
    analytic_tree_cycle_pole = _analyze_analytic_tree_cycle_pole(
        specification,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if analytic_tree_cycle_pole is not None:
        return finish(analytic_tree_cycle_pole)
    if specification.tree_dimension == 0:
        raise ValueError(
            "The specification has no productive transient trees and no "
            "supported finite cycle-pole regime"
        )
    long_cycle_profiles = specification._long_cycle_profiles()
    if any(
        profile not in long_cycle_profiles
        for profile in specification.transient_profiles
    ):
        raise ValueError(
            "The regular analyzer requires every productive transient-tree "
            "type to occur in the long-cycle transfer"
        )

    dependencies = specification.dependency_matrix()
    complete_compatibility = (
        not specification.has_guarded_predecessors
        and all(all(row) for row in dependencies)
    )
    if complete_compatibility:
        total_multiplicity = sum(
            specification.multiplicities[profile]
            for profile in specification.transient_profiles
        )
        radius = 1.0 / (total_multiplicity * e)
        tree_values = tuple(
            specification.multiplicities[profile] / total_multiplicity
            for profile in specification.transient_profiles
        )
        solver_residual = 0.0
    else:
        radius, tree_values, solver_residual = _solve_multitype_critical_point(
            specification,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    rhs = specification.tree_rhs(radius, tree_values)
    jacobian = specification.tree_jacobian(radius, tree_values)
    components = _strongly_connected_components(dependencies)
    critical_components = tuple(
        component
        for component in components
        if abs(
            _perron_eigenvalue(_principal_submatrix(jacobian, component)) - 1.0
        )
        <= max(1e-7, 100.0 * tolerance)
    )
    supercritical_components = tuple(
        component
        for component in components
        if _perron_eigenvalue(_principal_submatrix(jacobian, component))
        > 1.0 + max(1e-7, 100.0 * tolerance)
    )
    if supercritical_components:
        raise ValueError(
            "The candidate point has a supercritical dependency component"
        )
    if not critical_components:
        raise ValueError(
            "The regular analyzer did not find a critical dependency component"
        )
    if not _components_are_independent(dependencies, critical_components):
        raise ValueError(
            "Nested nonlinear critical components are outside the regular regime"
        )
    right_directions, left_directions = _critical_kernel_bases(
        jacobian,
        critical_components,
    )
    perron_right = tuple(
        sum(direction[index] for direction in right_directions)
        for index in range(specification.tree_dimension)
    )
    perron_left = tuple(
        sum(direction[index] for direction in left_directions)
        for index in range(specification.tree_dimension)
    )

    x_derivative = specification.tree_x_derivative(radius, tree_values)
    mode_data = []
    eigen_residuals = []
    for right, left in zip(right_directions, left_directions):
        hessian_direction = specification.tree_hessian_direction(
            radius,
            tree_values,
            right,
        )
        numerator = 2.0 * radius * _dot(left, x_derivative)
        denominator = _dot(left, hessian_direction)
        if numerator <= 0.0 or denominator <= 0.0:
            raise ValueError(
                "A critical component fails the positive square-root conditions"
            )
        mode_data.append((right, sqrt(numerator / denominator)))
        eigen_residuals.extend(
            sum(
                jacobian[row][column] * right[column]
                for column in range(specification.tree_dimension)
            )
            - right[row]
            for row in range(specification.tree_dimension)
        )
        eigen_residuals.extend(
            sum(
                left[row] * jacobian[row][column]
                for row in range(specification.tree_dimension)
            )
            - left[column]
            for column in range(specification.tree_dimension)
        )
    critical_residual = max(
        solver_residual,
        *(abs(value - expected) for value, expected in zip(rhs, tree_values)),
        *(abs(value) for value in eigen_residuals),
        abs(specification.long_cycle_determinant(radius, tree_values)),
        specification.jacobian_transfer_residual(radius, tree_values),
    )
    if critical_residual > max(1e-7, 100.0 * tolerance):
        raise ValueError(
            "The regular critical-point and positivity conditions could not be verified"
        )

    tree_sqrt_coefficients = tuple(
        sum(scale * direction[index] for direction, scale in mode_data)
        for index in range(specification.tree_dimension)
    )

    correction = specification.finite_cycle_correction(radius, tree_values)
    critical_count = len(critical_components)
    if critical_count == 1:
        denominator_sqrt_coefficient = (
            sum(tree_sqrt_coefficients)
            if complete_compatibility
            else _denominator_sqrt_coefficient(
                specification,
                radius,
                tree_values,
                tree_sqrt_coefficients,
            )
        )
        denominator_error = 0.0
    else:
        denominator_sqrt_coefficient, denominator_error = (
            _denominator_direction_coefficient(
                specification,
                radius,
                tree_values,
                tree_sqrt_coefficients,
                critical_count,
            )
        )
        critical_residual = max(critical_residual, denominator_error)
    if denominator_sqrt_coefficient <= 0.0:
        raise ValueError(
            "The determinant does not have the expected positive critical term"
        )
    singular_exponent = critical_count / 2.0
    leading_constant = (
        exp(correction)
        / (denominator_sqrt_coefficient * gamma(singular_exponent))
    )
    result = C1FLabeledAsymptotics(
            radius=radius,
            exponential_growth=1.0 / radius,
            power=singular_exponent - 1.0,
            leading_constant=leading_constant,
            tree_values=tree_values,
            tree_sqrt_coefficients=tree_sqrt_coefficients,
            finite_cycle_correction=correction,
            denominator_sqrt_coefficient=denominator_sqrt_coefficient,
            perron_right=perron_right,
            perron_left=perron_left,
            critical_residual=critical_residual,
            eventual_start=specification.eventual_start,
            regime=(
                "multiple-tree-critical"
                if critical_count > 1
                else (
                    "guarded-tree-critical"
                    if specification.has_guarded_predecessors
                    else (
                        "scalar-tree-critical"
                        if specification.tree_dimension == 1
                        else (
                            "rank-one-tree-critical"
                            if complete_compatibility
                            else "multitype-tree-critical"
                        )
                    )
                )
            ),
            critical_component=tuple(
                index
                for component in critical_components
                for index in component
            ),
            critical_components=critical_components,
            critical_directions=right_directions,
            critical_left_directions=left_directions,
        )
    return finish(
        _project_periodic_labeled_orbit(
            specification,
            result,
            expansion_order=expansion_order,
        )
    )


def c1f_labeled_asymptotics(
    formula: str | C1FFormula | CompiledC1F | C1FLabeledSpecification,
    *,
    function_symbol: str = "f",
    profile_pruning: bool = True,
    twin_compression: bool = True,
    tolerance: float = 1e-10,
    max_iterations: int = 80,
    certify: bool = False,
    precision_bits: int = 192,
    max_precision_bits: int | None = None,
    expansion_order: int = 0,
) -> C1FLabeledAsymptotics:
    """Return the leading labeled asymptotic for a C1F sentence.

    This is the concise public entry point.  The result represents

    ``leading_constant * n! * exponential_growth**n * n**power``.

    Unsupported or uncertified singular regimes raise ``ValueError`` rather
    than returning a heuristic constant.  With ``certify=True``, the optional
    interval backend attaches rigorous local enclosures in ``certificate``.
    """

    return analyze_c1f_labeled_asymptotics(
        formula,
        function_symbol=function_symbol,
        profile_pruning=profile_pruning,
        twin_compression=twin_compression,
        tolerance=tolerance,
        max_iterations=max_iterations,
        certify=certify,
        precision_bits=precision_bits,
        max_precision_bits=max_precision_bits,
        expansion_order=expansion_order,
    )


def _totient(value: int) -> int:
    result = value
    divisor = 2
    remaining = value
    while divisor * divisor <= remaining:
        if remaining % divisor == 0:
            while remaining % divisor == 0:
                remaining //= divisor
            result -= result // divisor
        divisor += 1
    if remaining > 1:
        result -= result // remaining
    return result


def _unlabeled_child_distribution(
    specification: C1FLabeledSpecification,
    root_profile: int,
    levels: Sequence[tuple[float, ...] | None],
    index: int,
) -> dict[tuple[int, ...], float]:
    """Evaluate the Pólya child multiset at ``x**index``."""

    grouped: dict[tuple[int, ...], float] = {}
    repeat_limit = (len(levels) - 1) // index
    for repeat in range(1, repeat_limit + 1):
        repeated_values = levels[index * repeat]
        assert repeated_values is not None
        for child_index, child_profile in enumerate(
            specification.transient_profiles
        ):
            if not specification.successor[child_profile][root_profile]:
                continue
            contribution = tuple(
                min(cap, repeat * value)
                for cap, value in zip(
                    specification.local_caps,
                    specification._local_contribution(child_profile),
                )
            )
            grouped[contribution] = grouped.get(contribution, 0.0) + (
                repeated_values[child_index] / repeat
            )

    zero = tuple(0 for _ in specification.local_caps)
    distribution = {zero: 1.0}
    for contribution, value in grouped.items():
        distribution = _distribution_multiply(
            distribution,
            _capped_exponential_factor(
                value,
                contribution,
                specification.local_caps,
            ),
            specification.local_caps,
        )
    return distribution


def _unlabeled_tree_map(
    specification: C1FLabeledSpecification,
    x: float,
    levels: list[tuple[float, ...] | None],
    index: int,
    trees: Sequence[float],
    profile_fugacities: Sequence[float] | None = None,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    levels[index] = tuple(trees)
    distributions = tuple(
        _unlabeled_child_distribution(
            specification,
            root_profile,
            levels,
            index,
        )
        for root_profile in specification.transient_profiles
    )
    atom = x**index
    phi = tuple(
        atom
        * specification.multiplicities[root_profile]
        * (
            1.0
            if profile_fugacities is None
            else profile_fugacities[root_profile] ** index
        )
        * specification._projected_child_value(
            root_profile,
            distributions[row],
        )
        for row, root_profile in enumerate(specification.transient_profiles)
    )
    jacobian = tuple(
        tuple(
            0.0
            if not specification.successor[child_profile][root_profile]
            else atom
            * specification.multiplicities[root_profile]
            * (
                1.0
                if profile_fugacities is None
                else profile_fugacities[root_profile] ** index
            )
            * specification._projected_child_value(
                root_profile,
                distributions[row],
                specification._local_contribution(child_profile),
            )
            for child_profile in specification.transient_profiles
        )
        for row, root_profile in enumerate(specification.transient_profiles)
    )
    return phi, jacobian


def _solve_unlabeled_tree_levels(
    specification: C1FLabeledSpecification,
    x: float,
    tail_terms: int,
    *,
    tolerance: float,
    max_iterations: int,
    profile_fugacities: Sequence[float] | None = None,
) -> list[tuple[float, ...] | None]:
    dimension = specification.tree_dimension
    levels: list[tuple[float, ...] | None] = [None] * (tail_terms + 1)
    for index in range(tail_terms, 0, -1):
        trees = tuple(0.0 for _ in range(dimension))
        for _ in range(12):
            phi, _ = _unlabeled_tree_map(
                specification,
                x,
                levels,
                index,
                trees,
                profile_fugacities,
            )
            trees = phi

        residual_norm = float("inf")
        for _ in range(max(40, 2 * max_iterations)):
            phi, jacobian = _unlabeled_tree_map(
                specification,
                x,
                levels,
                index,
                trees,
                profile_fugacities,
            )
            residual = tuple(
                value - tree for value, tree in zip(phi, trees)
            )
            residual_norm = max((abs(value) for value in residual), default=0.0)
            if residual_norm <= max(1e-13, tolerance * 0.01):
                break
            operator = tuple(
                tuple(
                    (1.0 if row == column else 0.0) - jacobian[row][column]
                    for column in range(dimension)
                )
                for row in range(dimension)
            )
            try:
                update = _solve_linear_system(operator, residual)
            except ValueError as exc:
                raise ValueError(
                    "The truncated Pólya tree system reached a singular Jacobian"
                ) from exc
            step = 1.0
            while step >= 2.0**-30:
                candidate = tuple(
                    tree + step * delta
                    for tree, delta in zip(trees, update)
                )
                if min(candidate, default=0.0) < 0.0:
                    step /= 2.0
                    continue
                candidate_phi, _ = _unlabeled_tree_map(
                    specification,
                    x,
                    levels,
                    index,
                    candidate,
                    profile_fugacities,
                )
                candidate_norm = max(
                    (
                        abs(value - tree)
                        for value, tree in zip(candidate_phi, candidate)
                    ),
                    default=0.0,
                )
                if candidate_norm < residual_norm:
                    trees = candidate
                    break
                step /= 2.0
            else:
                raise ValueError(
                    "The truncated Pólya tree Newton iteration did not decrease"
                )
        else:
            raise ValueError(
                "The truncated Pólya tree system did not converge"
            )
        levels[index] = tuple(trees)
    return levels


def _unlabeled_cycle_matrix(
    specification: C1FLabeledSpecification,
    active_profiles: Sequence[int],
    x: float,
    levels: list[tuple[float, ...] | None],
    index: int,
    profile_fugacities: Sequence[float] | None = None,
) -> tuple[tuple[float, ...], ...]:
    distributions = {
        profile: _unlabeled_child_distribution(
            specification,
            profile,
            levels,
            index,
        )
        for profile in active_profiles
    }
    atom = x**index
    return tuple(
        tuple(
            0.0
            if not specification.successor[previous][current]
            else atom
            * specification.multiplicities[current]
            * (
                1.0
                if profile_fugacities is None
                else profile_fugacities[current] ** index
            )
            * specification._projected_child_value(
                current,
                distributions[current],
                specification._local_contribution(previous),
            )
            for current in active_profiles
        )
        for previous in active_profiles
    )


def _unlabeled_log_ogf(
    specification: C1FLabeledSpecification,
    x: float,
    levels: list[tuple[float, ...] | None],
    profile_fugacities: Sequence[float] | None = None,
) -> float:
    """Evaluate the truncated logarithm of the unlabeled OGF.

    The logarithm is the sum of ``C(x^k) / k`` over ``k``, where ``C`` is
    the series of the connected components (see the UnaryFOMC
    documentation, docs/unlabeled-counting.md).
    """

    tail_terms = len(levels) - 1
    long_profiles = specification._long_cycle_profiles()
    visible_profiles = tuple(
        tuple(
            profile
            for profile, visible_length in enumerate(
                specification.visible_cycle_lengths
            )
            if visible_length == cycle_length
        )
        for cycle_length in range(1, specification.depth + 1)
    )
    matrix_cache: dict[tuple[tuple[int, ...], int], tuple[tuple[float, ...], ...]] = {}

    def matrix(profiles: tuple[int, ...], index: int) -> tuple[tuple[float, ...], ...]:
        key = profiles, index
        cached = matrix_cache.get(key)
        if cached is None:
            cached = _unlabeled_cycle_matrix(
                specification,
                profiles,
                x,
                levels,
                index,
                profile_fugacities,
            )
            matrix_cache[key] = cached
        return cached

    def connected(index: int) -> float:
        value = 0.0
        for adams in range(1, tail_terms // index + 1):
            transfer = matrix(long_profiles, index * adams)
            operator = tuple(
                tuple(
                    (1.0 if row == column else 0.0)
                    - transfer[row][column]
                    for column in range(len(transfer))
                )
                for row in range(len(transfer))
            )
            determinant = _matrix_determinant(operator)
            if determinant <= 0.0:
                raise ValueError(
                    "A substituted Pólya cycle determinant is not positive"
                )
            value -= _totient(adams) * log(determinant) / adams

        for cycle_length, profiles in enumerate(visible_profiles, 1):
            correction = 0.0
            for adams in range(1, cycle_length + 1):
                if cycle_length % adams or index * adams > tail_terms:
                    continue
                exponent = cycle_length // adams
                visible = matrix(profiles, index * adams)
                long = matrix(long_profiles, index * adams)
                correction += _totient(adams) * (
                    _matrix_trace(_matrix_power(visible, exponent))
                    - _matrix_trace(_matrix_power(long, exponent))
                )
            value += correction / cycle_length
        return value

    return sum(
        connected(index) / index
        for index in range(1, tail_terms + 1)
    )


def _complex_capped_exponential_factor(
    value: complex,
    contribution: tuple[int, ...],
    caps: tuple[int, ...],
) -> dict[tuple[int, ...], complex]:
    zero = tuple(0 for _ in caps)
    if not any(contribution):
        return {zero: cmath.exp(value)}
    saturation_count = max(
        (cap + increment - 1) // increment
        for cap, increment in zip(caps, contribution)
        if increment
    )
    result: dict[tuple[int, ...], complex] = {}
    term = 1 + 0j
    partial = 0j
    for count in range(saturation_count):
        state = tuple(
            min(cap, count * increment)
            for cap, increment in zip(caps, contribution)
        )
        result[state] = result.get(state, 0j) + term
        partial += term
        term *= value / (count + 1)
    saturated = tuple(
        cap if increment else 0
        for cap, increment in zip(caps, contribution)
    )
    result[saturated] = result.get(saturated, 0j) + cmath.exp(value) - partial
    return result


def _complex_distribution_multiply(
    left: dict[tuple[int, ...], complex],
    right: dict[tuple[int, ...], complex],
    caps: tuple[int, ...],
) -> dict[tuple[int, ...], complex]:
    result: dict[tuple[int, ...], complex] = {}
    for left_state, left_value in left.items():
        for right_state, right_value in right.items():
            state = _capped_add(left_state, right_state, caps)
            result[state] = result.get(state, 0j) + left_value * right_value
    return result


def _complex_unlabeled_child_distribution(
    specification: C1FLabeledSpecification,
    root_profile: int,
    levels: Sequence[tuple[complex, ...] | None],
    index: int,
) -> dict[tuple[int, ...], complex]:
    grouped: dict[tuple[int, ...], complex] = {}
    repeat_limit = (len(levels) - 1) // index
    for repeat in range(1, repeat_limit + 1):
        repeated_values = levels[index * repeat]
        assert repeated_values is not None
        for child_index, child_profile in enumerate(
            specification.transient_profiles
        ):
            if not specification.successor[child_profile][root_profile]:
                continue
            contribution = tuple(
                min(cap, repeat * value)
                for cap, value in zip(
                    specification.local_caps,
                    specification._local_contribution(child_profile),
                )
            )
            grouped[contribution] = grouped.get(contribution, 0j) + (
                repeated_values[child_index] / repeat
            )
    zero = tuple(0 for _ in specification.local_caps)
    distribution = {zero: 1 + 0j}
    for contribution, value in grouped.items():
        distribution = _complex_distribution_multiply(
            distribution,
            _complex_capped_exponential_factor(
                value,
                contribution,
                specification.local_caps,
            ),
            specification.local_caps,
        )
    return distribution


def _complex_projected_child_value(
    specification: C1FLabeledSpecification,
    root_profile: int,
    distribution: dict[tuple[int, ...], complex],
    extra: tuple[int, ...] | None = None,
) -> complex:
    accepted = specification._accepted_states(root_profile)
    return sum(
        (
            value
            for state, value in distribution.items()
            if (
                state
                if extra is None
                else _capped_add(state, extra, specification.local_caps)
            ) in accepted
        ),
        0j,
    )


def _complex_unlabeled_cycle_matrix(
    specification: C1FLabeledSpecification,
    active_profiles: Sequence[int],
    x: complex,
    levels: Sequence[tuple[complex, ...] | None],
    index: int,
) -> tuple[tuple[complex, ...], ...]:
    distributions = {
        profile: _complex_unlabeled_child_distribution(
            specification,
            profile,
            levels,
            index,
        )
        for profile in active_profiles
    }
    atom = x**index
    return tuple(
        tuple(
            0j
            if not specification.successor[previous][current]
            else atom
            * specification.multiplicities[current]
            * _complex_projected_child_value(
                specification,
                current,
                distributions[current],
                specification._local_contribution(previous),
            )
            for current in active_profiles
        )
        for previous in active_profiles
    )


def _complex_unlabeled_log_ogf(
    specification: C1FLabeledSpecification,
    x: complex,
    levels: Sequence[tuple[complex, ...] | None],
) -> complex:
    """Evaluate the retained Pólya logarithm at a conjugate point."""

    tail_terms = len(levels) - 1
    long_profiles = specification._long_cycle_profiles()
    visible_profiles = tuple(
        tuple(
            profile
            for profile, visible_length in enumerate(
                specification.visible_cycle_lengths
            )
            if visible_length == cycle_length
        )
        for cycle_length in range(1, specification.depth + 1)
    )
    cache: dict[tuple[tuple[int, ...], int], tuple[tuple[complex, ...], ...]] = {}

    def matrix(
        profiles: tuple[int, ...],
        index: int,
    ) -> tuple[tuple[complex, ...], ...]:
        key = profiles, index
        if key not in cache:
            cache[key] = _complex_unlabeled_cycle_matrix(
                specification,
                profiles,
                x,
                levels,
                index,
            )
        return cache[key]

    def connected(index: int) -> complex:
        value = 0j
        for adams in range(1, tail_terms // index + 1):
            transfer = matrix(long_profiles, index * adams)
            operator = tuple(
                tuple(
                    (1 if row == column else 0) - transfer[row][column]
                    for column in range(len(transfer))
                )
                for row in range(len(transfer))
            )
            determinant = _matrix_determinant(operator)
            if abs(determinant) <= 1e-14:
                raise ValueError("A conjugate Pólya determinant is numerically zero")
            value -= _totient(adams) * cmath.log(determinant) / adams
        for cycle_length, profiles in enumerate(visible_profiles, 1):
            correction = 0j
            for adams in range(1, cycle_length + 1):
                if cycle_length % adams or index * adams > tail_terms:
                    continue
                exponent = cycle_length // adams
                visible = matrix(profiles, index * adams)
                long = matrix(long_profiles, index * adams)
                correction += _totient(adams) * (
                    _matrix_trace(_matrix_power(visible, exponent))
                    - _matrix_trace(_matrix_power(long, exponent))
                )
            value += correction / cycle_length
        return value

    return sum(
        (connected(index) / index for index in range(1, tail_terms + 1)),
        0j,
    )


def _analyze_subcritical_unlabeled_specification(
    specification: C1FLabeledSpecification,
    *,
    tolerance: float,
    max_iterations: int,
    tail_terms: int,
    profile_fugacities: Sequence[float] | None = None,
    expansion_order: int = 0,
) -> C1FUnlabeledAsymptotics:
    """Numerically analyze the general finite subcritical Pólya system."""

    if tail_terms < 16:
        raise ValueError("polya_tail_terms must be at least 16")
    lower = 0.0
    upper = 1.0 - 1e-8
    lower_levels: list[tuple[float, ...] | None] | None = None
    for _ in range(64):
        middle = (lower + upper) / 2.0
        try:
            levels = _solve_unlabeled_tree_levels(
                specification,
                middle,
                tail_terms,
                tolerance=tolerance,
                max_iterations=max_iterations,
                profile_fugacities=profile_fugacities,
            )
            transfer = _unlabeled_cycle_matrix(
                specification,
                specification._long_cycle_profiles(),
                middle,
                levels,
                1,
                profile_fugacities,
            )
            spectral_radius = _perron_eigenvalue(transfer)
        except (OverflowError, ValueError, ZeroDivisionError):
            spectral_radius = float("inf")
            levels = None
        if spectral_radius < 1.0:
            lower = middle
            lower_levels = levels
        else:
            upper = middle
    radius = (lower + upper) / 2.0
    if radius >= 1.0 - 1e-6 or lower_levels is None:
        raise ValueError(
            "The monadic unlabeled system is not certified subcritical below radius one"
        )

    probe_x = radius * (1.0 - 1e-7)
    probe_levels = _solve_unlabeled_tree_levels(
        specification,
        probe_x,
        tail_terms,
        tolerance=tolerance,
        max_iterations=max_iterations,
        profile_fugacities=profile_fugacities,
    )
    long_profiles = specification._long_cycle_profiles()
    transfer = _unlabeled_cycle_matrix(
        specification,
        long_profiles,
        probe_x,
        probe_levels,
        1,
        profile_fugacities,
    )
    support = tuple(tuple(value > 0.0 for value in row) for row in transfer)
    components = _strongly_connected_components(support)
    component_roots = tuple(
        _perron_eigenvalue(_principal_submatrix(transfer, component))
        if len(component) > 1 or support[component[0]][component[0]]
        else 0.0
        for component in components
    )
    largest = max(component_roots, default=0.0)
    critical_components = tuple(
        component
        for component, root in zip(components, component_roots)
        if largest - root <= 2e-3
    )
    component_periods = tuple(
        _matrix_period(_principal_submatrix(transfer, component))
        for component in critical_components
    )
    periodicity = 1
    for period in component_periods:
        periodicity = _lcm(periodicity, period)
    pure_irreducible_periodic = (
        specification.tree_dimension == 0
        and len(critical_components) == 1
        and critical_components[0] == tuple(range(len(long_profiles)))
        and all(
            visible_length is None
            for visible_length in specification.visible_cycle_lengths
        )
    )
    if periodicity > 1 and not pure_irreducible_periodic:
        # For implicit tree systems the matrix period alone is not the size
        # period.  Exact coefficient congruences are projected below.
        periodicity = 1
    transient_position = {
        profile: index
        for index, profile in enumerate(specification.transient_profiles)
    }
    _, tree_jacobian = _unlabeled_tree_map(
        specification,
        probe_x,
        probe_levels,
        1,
        probe_levels[1] or tuple(),
        profile_fugacities,
    )
    singular_exponent = 0.0
    for component in critical_components:
        profiles = tuple(long_profiles[index] for index in component)
        if all(profile in transient_position for profile in profiles):
            tree_component = tuple(transient_position[profile] for profile in profiles)
            tree_root = _perron_eigenvalue(
                _principal_submatrix(tree_jacobian, tree_component)
            )
        else:
            tree_root = 0.0
        singular_exponent += 0.5 if tree_root >= 1.0 - 2e-3 else 1.0
    if singular_exponent <= 0.0:
        raise ValueError("No regular subcritical Pólya singular factor was found")

    epsilon_values = (4e-4, 1e-4, 2.5e-5)
    scaled_values = []
    roots = []
    for epsilon in epsilon_values:
        sample_x = radius * (1.0 - epsilon)
        levels = _solve_unlabeled_tree_levels(
            specification,
            sample_x,
            tail_terms,
            tolerance=tolerance,
            max_iterations=max_iterations,
            profile_fugacities=profile_fugacities,
        )
        log_ogf = _unlabeled_log_ogf(
            specification,
            sample_x,
            levels,
            profile_fugacities,
        )
        scaled_values.append(
            exp(log_ogf + singular_exponent * log(epsilon))
        )
        roots.append(sqrt(epsilon))
    singular_factor = 0.0
    for index, value in enumerate(scaled_values):
        basis = 1.0
        for other, root in enumerate(roots):
            if other != index:
                basis *= -root / (roots[index] - root)
        singular_factor += value * basis
    leading_constant = singular_factor / gamma(singular_exponent)
    if leading_constant <= 0.0 or not all(
        value > 0.0 for value in scaled_values
    ):
        raise ValueError("The projected Pólya leading constant is not positive")
    residue_powers: tuple[float | None, ...] = tuple()
    residue_constants: tuple[float, ...] = tuple()
    if periodicity > 1:
        leading_constant *= periodicity
        residue_powers = (singular_exponent - 1.0,) + (None,) * (
            periodicity - 1
        )
        residue_constants = (leading_constant,) + (0.0,) * (
            periodicity - 1
        )
    result = C1FUnlabeledAsymptotics(
        radius=radius,
        exponential_growth=1.0 / radius,
        power=singular_exponent - 1.0,
        leading_constant=leading_constant,
        periodicity=periodicity,
        regime="subcritical-polya-multitype",
        tree_values=probe_levels[1] or tuple(),
        critical_components=critical_components,
        residue_powers=residue_powers,
        residue_leading_constants=residue_constants,
    )
    if specification.tree_dimension and critical_components:
        tree_positions = {
            profile: index
            for index, profile in enumerate(specification.transient_profiles)
        }
        critical_tree_components = tuple(
            tuple(
                tree_positions[long_profiles[index]]
                for index in component
                if long_profiles[index] in tree_positions
            )
            for component in critical_components
        )
        critical_tree_components = tuple(
            component for component in critical_tree_components if component
        )
        if critical_tree_components:
            result = _project_periodic_unlabeled_orbit(
                specification,
                result,
                critical_tree_components,
                tail_terms=tail_terms,
                tolerance=tolerance,
                max_iterations=max_iterations,
                expansion_order=expansion_order,
            )
    return result


def _analyze_unlabeled_global_markers(
    compiled: CompiledC1F,
    *,
    tolerance: float,
    max_iterations: int,
    tail_terms: int,
    certify: bool,
    precision_bits: int,
    max_precision_bits: int | None,
    expansion_order: int = 0,
) -> C1FUnlabeledAsymptotics:
    """Analyze the subcritical Pólya system under global counters.

    The numerical Pólya system is analyzed separately for every accepted
    state of the capped counters.
    """

    monadic = compiled.branches[0].monadic
    plan = monadic.cardinality_plan
    caps = plan.caps
    active_types = _locally_active_types(monadic)
    eventual_start = max(
        _largest_count_threshold(monadic.formula) + 1,
        max(caps),
    )
    accepted_states = tuple(
        state
        for state in product(*(range(cap + 1) for cap in caps))
        if plan.satisfies(monadic.formula, state, eventual_start)
    )
    face_results: list[C1FUnlabeledAsymptotics] = []
    for accepted_state in accepted_states:
        saturated = tuple(
            index
            for index, (value, cap) in enumerate(zip(accepted_state, caps))
            if value == cap
        )
        remaining = tuple(
            index for index in range(len(caps)) if index not in saturated
        )
        base_types = tuple(
            type_id
            for type_id in active_types
            if all(plan.profile_states[type_id][index] == 0 for index in remaining)
        )
        if not base_types:
            continue
        try:
            base_specification = _build_labeled_specification(
                monadic,
                base_types,
                eventual_start=eventual_start,
            )
            base = _analyze_subcritical_unlabeled_specification(
                base_specification,
                tolerance=tolerance,
                max_iterations=max_iterations,
                tail_terms=tail_terms,
                expansion_order=expansion_order,
            )
        except ValueError:
            continue
        if len(base.critical_components) != 1:
            raise ValueError(
                "Capped-state Pólya lifting currently requires one regular "
                "critical component per face"
            )
        fixed_degree = sum(accepted_state[index] for index in remaining)
        if expansion_order and fixed_degree:
            raise ValueError(
                "Higher-order Pólya expansions after fixed global-marker "
                "extraction require the bivariate marker recurrence"
            )

        viable = True
        for marker_index in saturated:
            restricted_types = tuple(
                type_id
                for type_id in base_types
                if plan.profile_states[type_id][marker_index] == 0
            )
            if not restricted_types:
                continue
            try:
                restricted = _analyze_subcritical_unlabeled_specification(
                    _build_labeled_specification(
                        monadic,
                        restricted_types,
                        eventual_start=eventual_start,
                    ),
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    tail_terms=tail_terms,
                )
            except ValueError:
                continue
            if not base.radius < restricted.radius * (
                1.0 - max(1e-7, 100.0 * tolerance)
            ):
                viable = False
                break
        if not viable:
            continue

        derivatives = [0.0] * len(caps)
        for marker_index in remaining:
            count = accepted_state[marker_index]
            if count == 0:
                continue
            perturbation_types = tuple(
                type_id
                for type_id in active_types
                if all(
                    plan.profile_states[type_id][other] == 0
                    for other in remaining
                    if other != marker_index
                )
            )
            if not any(
                plan.profile_states[type_id][marker_index]
                for type_id in perturbation_types
            ):
                viable = False
                break

            def perturbed_radius(fugacity: float) -> float:
                profile_fugacities = tuple(
                    fugacity
                    if plan.profile_states[type_id][marker_index]
                    else 1.0
                    for type_id in perturbation_types
                )
                return _analyze_subcritical_unlabeled_specification(
                    _build_labeled_specification(
                        monadic,
                        perturbation_types,
                        eventual_start=eventual_start,
                    ),
                    tolerance=min(tolerance, 1e-11),
                    max_iterations=max_iterations,
                    tail_terms=tail_terms,
                    profile_fugacities=profile_fugacities,
                ).radius

            step = 8e-4
            coarse = log(base.radius / perturbed_radius(step)) / step
            fine = log(base.radius / perturbed_radius(step / 2.0)) / (step / 2.0)
            finer = log(base.radius / perturbed_radius(step / 4.0)) / (step / 4.0)
            first = 2.0 * fine - coarse
            second = 2.0 * finer - fine
            derivatives[marker_index] = (4.0 * second - first) / 3.0
            if derivatives[marker_index] <= max(1e-7, 100.0 * tolerance):
                viable = False
                break
        if not viable:
            continue

        factor = 1.0
        for marker_index in remaining:
            count = accepted_state[marker_index]
            factor *= derivatives[marker_index] ** count / factorial(count)
        certificate = None
        if certify:
            from .interval_asymptotics import certify_unlabeled_global_marker_face

            certificate = certify_unlabeled_global_marker_face(
                monadic,
                active_types,
                base_types,
                remaining,
                accepted_state,
                replace(base, global_marker_caps=caps),
                precision_bits=precision_bits,
                max_precision_bits=max_precision_bits,
                tail_terms=tail_terms,
            )
        face_results.append(
            replace(
                base,
                power=base.power + fixed_degree,
                leading_constant=base.leading_constant * factor,
                regime=f"capped-global-face:{base.regime}",
                global_marker_caps=caps,
                accepted_global_states=(accepted_state,),
                marker_growth_derivatives=tuple(derivatives),
                residue_powers=tuple(
                    None if power is None else power + fixed_degree
                    for power in base.residue_powers
                ),
                residue_leading_constants=tuple(
                    constant * factor
                    for constant in base.residue_leading_constants
                ),
                residue_expansions=_shift_expansions(
                    base.residue_expansions,
                    fixed_degree,
                    factor,
                ),
                certificate=certificate,
            )
        )

    if not face_results:
        raise ValueError(
            "No accepted capped-state face has a supported subcritical Pólya asymptotic"
        )
    if len(face_results) == 1:
        return face_results[0]
    probe = 100
    dominant = max(face_results, key=lambda result: result.log_estimate(probe))
    return replace(
        dominant,
        regime="capped-global-face-sum:subcritical-polya",
        accepted_global_states=accepted_states,
        summands=tuple(face_results),
        certificate=None,
    )


def c1f_unlabeled_asymptotics(
    formula: str | C1FFormula | CompiledC1F | C1FLabeledSpecification,
    *,
    function_symbol: str = "f",
    profile_pruning: bool = True,
    twin_compression: bool = True,
    tolerance: float = 1e-10,
    max_iterations: int = 80,
    certify: bool = False,
    precision_bits: int = 192,
    max_precision_bits: int | None = None,
    polya_tail_terms: int = 128,
    expansion_order: int = 0,
) -> C1FUnlabeledAsymptotics:
    """Analyze subcritical monadic or higher-arity unlabeled asymptotics.

    Monadic inputs use the Pólya system of the exact unlabeled construction.
    Higher-arity inputs use the rigidity transfer from the labeled problem
    and are divided by ``n!`` only after the exact free-relation factor has
    been restored (see the UnaryFOMC documentation,
    docs/unlabeled-asymptotics.md).
    ``expansion_order`` requests successive half-orders after the leading
    term.
    """

    if expansion_order < 0:
        raise ValueError("expansion_order must be non-negative")
    specification_input = isinstance(formula, C1FLabeledSpecification)
    compiled = None if specification_input else (
        formula
        if isinstance(formula, CompiledC1F)
        else compile_c1f_fomc(
            formula,
            function_symbol=function_symbol,
            profile_pruning=profile_pruning,
            twin_compression=twin_compression,
        )
    )
    if compiled is not None and compiled.has_higher_arity_relations:
        labeled = analyze_c1f_labeled_asymptotics(
            compiled,
            tolerance=tolerance,
            max_iterations=max_iterations,
            certify=certify,
            precision_bits=precision_bits,
            max_precision_bits=max_precision_bits,
            expansion_order=expansion_order,
        )
        return C1FUnlabeledAsymptotics(
            radius=labeled.radius,
            exponential_growth=labeled.exponential_growth,
            power=labeled.power,
            leading_constant=labeled.leading_constant,
            periodicity=labeled.periodicity,
            regime="higher-arity-rigidity-transfer",
            labeled=labeled,
            residue_powers=labeled.residue_powers,
            residue_leading_constants=labeled.residue_leading_constants,
            residue_expansions=labeled.residue_expansions,
            certificate=labeled.certificate,
        )
    if compiled is not None and len(compiled.branches) != 1:
        summands = tuple(
            c1f_unlabeled_asymptotics(
                replace(compiled, branches=(branch,)),
                tolerance=tolerance,
                max_iterations=max_iterations,
                certify=certify,
                precision_bits=precision_bits,
                max_precision_bits=max_precision_bits,
                polya_tail_terms=polya_tail_terms,
                expansion_order=expansion_order,
            )
            for branch in compiled.branches
        )
        dominant = max(summands, key=lambda result: result.log_estimate(100))
        return replace(
            dominant,
            regime="nullary-branch-sum:subcritical-polya",
            summands=summands,
            certificate=None,
        )
    if compiled is not None:
        monadic = compiled.branches[0].monadic
        active_marker_states = {
            monadic.cardinality_plan.profile_states[type_id]
            for type_id in _locally_active_types(monadic)
        }
        if monadic.cardinality_plan.caps and len(active_marker_states) > 1:
            return _analyze_unlabeled_global_markers(
                compiled,
                tolerance=tolerance,
                max_iterations=max_iterations,
                tail_terms=polya_tail_terms,
                certify=certify,
                precision_bits=precision_bits,
                max_precision_bits=max_precision_bits,
                expansion_order=expansion_order,
            )
        specification = compile_c1f_labeled_specification(compiled)
    else:
        assert isinstance(formula, C1FLabeledSpecification)
        specification = formula
    dependencies = specification.dependency_matrix()
    complete_compatibility = (
        not specification.has_guarded_predecessors
        and specification.transient_profiles
        == specification._long_cycle_profiles()
        and all(all(row) for row in dependencies)
        and all(
            visible_length is None
            for visible_length in specification.visible_cycle_lengths
        )
    )
    if not complete_compatibility:
        result = _analyze_subcritical_unlabeled_specification(
            specification,
            tolerance=tolerance,
            max_iterations=max_iterations,
            tail_terms=polya_tail_terms,
            expansion_order=expansion_order,
        )
        if certify:
            if specification.tree_dimension == 0:
                from .interval_asymptotics import certify_cycle_pole_unlabeled

                certificate = certify_cycle_pole_unlabeled(
                    specification,
                    result,
                    precision_bits=precision_bits,
                    max_precision_bits=max_precision_bits,
                    tail_terms=polya_tail_terms,
                )
                return replace(result, certificate=certificate)
            if specification.tree_dimension == specification.cycle_dimension == 1:
                from .interval_asymptotics import certify_scalar_guarded_unlabeled

                certificate = certify_scalar_guarded_unlabeled(
                    specification,
                    result,
                    precision_bits=precision_bits,
                    max_precision_bits=max_precision_bits,
                    tail_terms=polya_tail_terms,
                )
                return replace(result, certificate=certificate)
            from .interval_asymptotics import certify_multitype_unlabeled

            certificate = certify_multitype_unlabeled(
                specification,
                result,
                precision_bits=precision_bits,
                max_precision_bits=max_precision_bits,
                tail_terms=polya_tail_terms,
            )
            return replace(result, certificate=certificate)
        return result
    if polya_tail_terms < 16:
        raise ValueError("polya_tail_terms must be at least 16")

    multiplicity = sum(
        specification.multiplicities[profile]
        for profile in specification.transient_profiles
    )

    def tail_values(radius: float) -> tuple[list[float], list[float]]:
        values = [0.0] * (polya_tail_terms + 1)
        derivatives = [0.0] * (polya_tail_terms + 1)
        for index in range(polya_tail_terms, 1, -1):
            tail = sum(
                values[index * repeat] / repeat
                for repeat in range(2, polya_tail_terms // index + 1)
            )
            atom = multiplicity * radius**index * exp(tail)
            value = atom
            for _ in range(30):
                updated = value - (value - atom * exp(value)) / (
                    1.0 - atom * exp(value)
                )
                if abs(updated - value) <= 1e-15 * max(1.0, abs(updated)):
                    value = updated
                    break
                value = updated
            values[index] = value
            derivative_tail = sum(
                derivatives[index * repeat] / repeat
                for repeat in range(2, polya_tail_terms // index + 1)
            )
            derivatives[index] = value * (
                index / radius + derivative_tail
            ) / (1.0 - value)
        return values, derivatives

    lower = 0.2 / max(1, multiplicity)
    upper = min(0.49, 0.5 / max(1, multiplicity) + 0.2)
    for _ in range(100):
        middle = (lower + upper) / 2.0
        values, _ = tail_values(middle)
        characteristic = (
            log(multiplicity * middle)
            + 1.0
            + sum(
                values[index] / index
                for index in range(2, polya_tail_terms + 1)
            )
        )
        if characteristic < 0.0:
            lower = middle
        else:
            upper = middle
    radius = (lower + upper) / 2.0
    values, derivatives = tail_values(radius)
    values[1] = 1.0
    characteristic_derivative = 1.0 / radius + sum(
        derivatives[index] / index
        for index in range(2, polya_tail_terms + 1)
    )
    square_root_coefficient = sqrt(2.0 * radius * characteristic_derivative)
    analytic_factor = exp(
        sum(
            -log(1.0 - values[index])
            for index in range(2, polya_tail_terms + 1)
        )
        - sum(
            _totient(adams)
            * values[outer * adams] ** (cycle_length // adams)
            / (outer * cycle_length)
            for outer in range(1, polya_tail_terms + 1)
            for cycle_length in range(1, specification.depth + 1)
            for adams in range(1, cycle_length + 1)
            if cycle_length % adams == 0
            and outer * adams <= polya_tail_terms
        )
    )
    leading_constant = analytic_factor / (
        square_root_coefficient * sqrt(pi)
    )
    result = C1FUnlabeledAsymptotics(
        radius=radius,
        exponential_growth=1.0 / radius,
        power=-0.5,
        leading_constant=leading_constant,
        regime="subcritical-polya-complete-compatibility",
    )
    if expansion_order:
        expanded = _analyze_subcritical_unlabeled_specification(
            specification,
            tolerance=tolerance,
            max_iterations=max_iterations,
            tail_terms=polya_tail_terms,
            expansion_order=expansion_order,
        )
        expansion_scale = leading_constant / expanded.leading_constant
        scaled_expansions = _shift_expansions(
            expanded.residue_expansions,
            0.0,
            expansion_scale,
        )
        result = replace(
            result,
            power=expanded.power,
            periodicity=expanded.periodicity,
            residue_powers=expanded.residue_powers,
            residue_leading_constants=tuple(
                residue[0].constant if residue else 0.0
                for residue in scaled_expansions
            ),
            residue_expansions=scaled_expansions,
        )
    if not certify:
        return result
    from .interval_asymptotics import certify_complete_compatibility_unlabeled

    certificate = certify_complete_compatibility_unlabeled(
        multiplicity,
        result,
        precision_bits=precision_bits,
        max_precision_bits=max_precision_bits,
        tail_terms=polya_tail_terms,
        forbidden_cycle_depth=specification.depth,
    )
    return replace(result, certificate=certificate)
