"""Formal Puiseux-series machinery for regular C1F specifications."""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from fractions import Fraction
from math import gamma
from typing import Any, Callable, Sequence

from .specification import C1FLabeledSpecification, _capped_add


def _scalar_exp(value: Any) -> Any:
    if isinstance(value, complex):
        return cmath.exp(value)
    method = getattr(value, "exp", None)
    if method is not None:
        return method()
    from math import exp

    return exp(value)


def _scalar_log(value: Any) -> Any:
    if isinstance(value, complex):
        return cmath.log(value)
    method = getattr(value, "log", None)
    if method is not None:
        return method()
    from math import log

    return log(value)


@dataclass(frozen=True)
class TruncatedSeries:
    """A dense power series truncated after a fixed non-negative degree."""

    coefficients: tuple[Any, ...]

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    @classmethod
    def constant(cls, value: Any, order: int) -> "TruncatedSeries":
        zero = value * 0
        return cls((value,) + (zero,) * order)

    def coefficient(self, degree: int) -> Any:
        if degree < 0 or degree > self.order:
            return self.coefficients[0] * 0
        return self.coefficients[degree]

    def truncate(self, order: int) -> "TruncatedSeries":
        if order > self.order:
            raise ValueError("Cannot increase a formal-series truncation order")
        return TruncatedSeries(self.coefficients[: order + 1])

    def _coerce(self, other: Any) -> "TruncatedSeries":
        if isinstance(other, TruncatedSeries):
            if other.order != self.order:
                raise ValueError("Formal-series truncation orders do not agree")
            return other
        return TruncatedSeries.constant(other, self.order)

    def __add__(self, other: Any) -> "TruncatedSeries":
        other = self._coerce(other)
        return TruncatedSeries(
            tuple(a + b for a, b in zip(self.coefficients, other.coefficients))
        )

    __radd__ = __add__

    def __neg__(self) -> "TruncatedSeries":
        return TruncatedSeries(tuple(-value for value in self.coefficients))

    def __sub__(self, other: Any) -> "TruncatedSeries":
        return self + (-self._coerce(other))

    def __rsub__(self, other: Any) -> "TruncatedSeries":
        return self._coerce(other) - self

    def __mul__(self, other: Any) -> "TruncatedSeries":
        other = self._coerce(other)
        zero = self.coefficients[0] * 0
        result = [zero for _ in range(self.order + 1)]
        for left_degree, left_value in enumerate(self.coefficients):
            for right_degree in range(self.order - left_degree + 1):
                result[left_degree + right_degree] += (
                    left_value * other.coefficients[right_degree]
                )
        return TruncatedSeries(tuple(result))

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "TruncatedSeries":
        other = self._coerce(other)
        zero = self.coefficients[0] * 0
        result = [zero for _ in range(self.order + 1)]
        result[0] = self.coefficients[0] / other.coefficients[0]
        for degree in range(1, self.order + 1):
            result[degree] = (
                self.coefficients[degree]
                - sum(
                    (
                        other.coefficients[offset]
                        * result[degree - offset]
                        for offset in range(1, degree + 1)
                    ),
                    zero,
                )
            ) / other.coefficients[0]
        return TruncatedSeries(tuple(result))

    def __rtruediv__(self, other: Any) -> "TruncatedSeries":
        return self._coerce(other) / self

    def __pow__(self, exponent: int) -> "TruncatedSeries":
        if exponent < 0:
            return TruncatedSeries.constant(1, self.order) / (self ** (-exponent))
        result = TruncatedSeries.constant(1, self.order)
        factor = self
        remaining = exponent
        while remaining:
            if remaining & 1:
                result = result * factor
            remaining >>= 1
            if remaining:
                factor = factor * factor
        return result

    def exp(self) -> "TruncatedSeries":
        zero = self.coefficients[0] * 0
        result = [zero for _ in range(self.order + 1)]
        result[0] = _scalar_exp(self.coefficients[0])
        for degree in range(1, self.order + 1):
            result[degree] = sum(
                (
                    offset
                    * self.coefficients[offset]
                    * result[degree - offset]
                    for offset in range(1, degree + 1)
                ),
                zero,
            ) / degree
        return TruncatedSeries(tuple(result))

    def log(self) -> "TruncatedSeries":
        zero = self.coefficients[0] * 0
        result = [zero for _ in range(self.order + 1)]
        result[0] = _scalar_log(self.coefficients[0])
        for degree in range(1, self.order + 1):
            result[degree] = (
                degree * self.coefficients[degree]
                - sum(
                    (
                        offset
                        * result[offset]
                        * self.coefficients[degree - offset]
                        for offset in range(1, degree)
                    ),
                    zero,
                )
            ) / (degree * self.coefficients[0])
        return TruncatedSeries(tuple(result))


def _series_distribution_multiply(
    left: dict[tuple[int, ...], TruncatedSeries],
    right: dict[tuple[int, ...], TruncatedSeries],
    caps: tuple[int, ...],
) -> dict[tuple[int, ...], TruncatedSeries]:
    result: dict[tuple[int, ...], TruncatedSeries] = {}
    for left_state, left_value in left.items():
        for right_state, right_value in right.items():
            state = _capped_add(left_state, right_state, caps)
            result[state] = result.get(
                state,
                TruncatedSeries.constant(0, left_value.order),
            ) + left_value * right_value
    return result


def _series_exponential_factor(
    value: TruncatedSeries,
    contribution: tuple[int, ...],
    caps: tuple[int, ...],
) -> dict[tuple[int, ...], TruncatedSeries]:
    zero_state = tuple(0 for _ in caps)
    if not any(contribution):
        return {zero_state: value.exp()}
    saturation_count = max(
        (cap + increment - 1) // increment
        for cap, increment in zip(caps, contribution)
        if increment
    )
    zero = TruncatedSeries.constant(0, value.order)
    result: dict[tuple[int, ...], TruncatedSeries] = {}
    term = TruncatedSeries.constant(1, value.order)
    partial = zero
    for count in range(saturation_count):
        state = tuple(
            min(cap, count * increment)
            for cap, increment in zip(caps, contribution)
        )
        result[state] = result.get(state, zero) + term
        partial += term
        term = term * value / (count + 1)
    saturated = tuple(
        cap if increment else 0
        for cap, increment in zip(caps, contribution)
    )
    result[saturated] = result.get(saturated, zero) + value.exp() - partial
    return result


def _series_child_distribution(
    specification: C1FLabeledSpecification,
    root_profile: int,
    trees: Sequence[TruncatedSeries],
) -> dict[tuple[int, ...], TruncatedSeries]:
    order = trees[0].order if trees else 0
    zero = TruncatedSeries.constant(0, order)
    grouped: dict[tuple[int, ...], TruncatedSeries] = {}
    for coordinate, child_profile in enumerate(specification.transient_profiles):
        if not specification.successor[child_profile][root_profile]:
            continue
        contribution = specification._local_contribution(child_profile)
        grouped[contribution] = grouped.get(contribution, zero) + trees[coordinate]
    zero_state = tuple(0 for _ in specification.local_caps)
    distribution = {zero_state: TruncatedSeries.constant(1, order)}
    for contribution, value in grouped.items():
        distribution = _series_distribution_multiply(
            distribution,
            _series_exponential_factor(
                value,
                contribution,
                specification.local_caps,
            ),
            specification.local_caps,
        )
    return distribution


def _series_projected_value(
    specification: C1FLabeledSpecification,
    root_profile: int,
    distribution: dict[tuple[int, ...], TruncatedSeries],
    order: int,
    extra: tuple[int, ...] | None = None,
) -> TruncatedSeries:
    result = TruncatedSeries.constant(0, order)
    accepted = specification._accepted_states(root_profile)
    for state, value in distribution.items():
        final_state = (
            state
            if extra is None
            else _capped_add(state, extra, specification.local_caps)
        )
        if final_state in accepted:
            result += value
    return result


def formal_tree_rhs(
    specification: C1FLabeledSpecification,
    x: TruncatedSeries,
    trees: Sequence[TruncatedSeries],
) -> tuple[TruncatedSeries, ...]:
    return tuple(
        x
        * specification.multiplicities[root_profile]
        * _series_projected_value(
            specification,
            root_profile,
            _series_child_distribution(specification, root_profile, trees),
            x.order,
        )
        for root_profile in specification.transient_profiles
    )


def _series_cycle_matrix(
    specification: C1FLabeledSpecification,
    active_profiles: Sequence[int],
    x: TruncatedSeries,
    trees: Sequence[TruncatedSeries],
) -> tuple[tuple[TruncatedSeries, ...], ...]:
    zero = TruncatedSeries.constant(0, x.order)
    distributions = {
        profile: _series_child_distribution(specification, profile, trees)
        for profile in active_profiles
    }
    return tuple(
        tuple(
            zero
            if not specification.successor[previous][current]
            else x
            * specification.multiplicities[current]
            * _series_projected_value(
                specification,
                current,
                distributions[current],
                x.order,
                specification._local_contribution(previous),
            )
            for current in active_profiles
        )
        for previous in active_profiles
    )


def _series_matrix_multiply(
    left: Sequence[Sequence[TruncatedSeries]],
    right: Sequence[Sequence[TruncatedSeries]],
) -> tuple[tuple[TruncatedSeries, ...], ...]:
    size = len(left)
    order = left[0][0].order if size else 0
    zero = TruncatedSeries.constant(0, order)
    return tuple(
        tuple(
            sum(
                (left[row][middle] * right[middle][column] for middle in range(size)),
                zero,
            )
            for column in range(size)
        )
        for row in range(size)
    )


def _series_matrix_power(
    matrix: Sequence[Sequence[TruncatedSeries]],
    exponent: int,
) -> tuple[tuple[TruncatedSeries, ...], ...]:
    size = len(matrix)
    order = matrix[0][0].order if size else 0
    result = tuple(
        tuple(
            TruncatedSeries.constant(1 if row == column else 0, order)
            for column in range(size)
        )
        for row in range(size)
    )
    factor = tuple(tuple(row) for row in matrix)
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _series_matrix_multiply(result, factor)
        remaining >>= 1
        if remaining:
            factor = _series_matrix_multiply(factor, factor)
    return result


def _series_determinant(
    matrix: Sequence[Sequence[TruncatedSeries]],
    order: int,
) -> TruncatedSeries:
    """Compute a determinant by subset dynamic programming without division."""

    size = len(matrix)
    if size == 0:
        return TruncatedSeries.constant(1, order)
    zero = TruncatedSeries.constant(0, order)
    states = {0: TruncatedSeries.constant(1, order)}
    for row in range(size):
        updated: dict[int, TruncatedSeries] = {}
        for mask, value in states.items():
            for column in range(size):
                bit = 1 << column
                if mask & bit:
                    continue
                inversions = sum(
                    1
                    for previous in range(column + 1, size)
                    if mask & (1 << previous)
                )
                term = value * matrix[row][column]
                if inversions & 1:
                    term = -term
                target = mask | bit
                updated[target] = updated.get(target, zero) + term
        states = updated
    return states[(1 << size) - 1]


def formal_labeled_numerator_denominator(
    specification: C1FLabeledSpecification,
    x: TruncatedSeries,
    trees: Sequence[TruncatedSeries],
) -> tuple[TruncatedSeries, TruncatedSeries]:
    long_matrix = _series_cycle_matrix(
        specification,
        specification._long_cycle_profiles(),
        x,
        trees,
    )
    size = len(long_matrix)
    operator = tuple(
        tuple(
            TruncatedSeries.constant(1 if row == column else 0, x.order)
            - long_matrix[row][column]
            for column in range(size)
        )
        for row in range(size)
    )
    denominator = _series_determinant(operator, x.order)
    correction = TruncatedSeries.constant(0, x.order)
    for cycle_length in range(1, specification.depth + 1):
        visible_profiles = tuple(
            profile
            for profile, visible_length in enumerate(
                specification.visible_cycle_lengths
            )
            if visible_length == cycle_length
        )
        visible = _series_cycle_matrix(
            specification,
            visible_profiles,
            x,
            trees,
        )
        visible_power = _series_matrix_power(visible, cycle_length)
        long_power = _series_matrix_power(long_matrix, cycle_length)
        correction += (
            sum(
                (visible_power[index][index] for index in range(len(visible_power))),
                TruncatedSeries.constant(0, x.order),
            )
            - sum(
                (long_power[index][index] for index in range(len(long_power))),
                TruncatedSeries.constant(0, x.order),
            )
        ) / cycle_length
    return correction.exp(), denominator


def compute_tree_puiseux_coefficients(
    specification: C1FLabeledSpecification,
    radius: Any,
    tree_values: Sequence[Any],
    first_coefficients: Sequence[Any],
    right_directions: Sequence[Sequence[Any]],
    left_directions: Sequence[Sequence[Any]],
    order: int,
    solve_linear: Callable[[Sequence[Sequence[Any]], Sequence[Any]], Sequence[Any]],
    rhs_builder: Callable[
        [TruncatedSeries, Sequence[TruncatedSeries]],
        Sequence[TruncatedSeries],
    ] | None = None,
) -> tuple[tuple[Any, ...], ...]:
    """Continue a regular implicit solution in powers of ``X`` through ``order``."""

    dimension = len(tree_values)
    critical_count = len(right_directions)
    if critical_count == 0 or len(left_directions) != critical_count:
        raise ValueError("Puiseux continuation requires paired critical directions")
    coefficients: list[tuple[Any, ...]] = [
        tuple(tree_values),
        tuple(first_coefficients),
    ]
    zero = radius * 0
    one = zero + 1
    if rhs_builder is None:
        rhs_builder = lambda x, trees: formal_tree_rhs(specification, x, trees)

    def residual_pair(
        degree: int,
        trial: Sequence[Any] | None = None,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        truncation = degree + 1
        x_coefficients = [zero for _ in range(truncation + 1)]
        x_coefficients[0] = radius
        if truncation >= 2:
            x_coefficients[2] = -radius
        x = TruncatedSeries(tuple(x_coefficients))
        tree_series = []
        for coordinate in range(dimension):
            values = [zero for _ in range(truncation + 1)]
            for known_degree, known in enumerate(coefficients):
                if known_degree <= truncation:
                    values[known_degree] = known[coordinate]
            if trial is not None:
                values[degree] = trial[coordinate]
            tree_series.append(TruncatedSeries(tuple(values)))
        rhs = rhs_builder(x, tree_series)
        residual = tuple(tree_series[index] - rhs[index] for index in range(dimension))
        return (
            tuple(value.coefficient(degree) for value in residual),
            tuple(value.coefficient(degree + 1) for value in residual),
        )

    for degree in range(2, order + 1):
        base, next_base = residual_pair(degree)
        top_columns: list[tuple[Any, ...]] = []
        next_columns: list[tuple[Any, ...]] = []
        for coordinate in range(dimension):
            trial = [zero for _ in range(dimension)]
            trial[coordinate] = one
            top, following = residual_pair(degree, trial)
            top_columns.append(tuple(value - base[row] for row, value in enumerate(top)))
            next_columns.append(
                tuple(value - next_base[row] for row, value in enumerate(following))
            )
        augmented = []
        rhs_values = []
        for row in range(dimension):
            augmented.append(
                [top_columns[column][row] for column in range(dimension)]
                + [right_directions[index][row] for index in range(critical_count)]
            )
            rhs_values.append(-base[row])
        for left in left_directions:
            augmented.append(
                [
                    sum(
                        (
                            left[row] * next_columns[column][row]
                            for row in range(dimension)
                        ),
                        zero,
                    )
                    for column in range(dimension)
                ]
                + [zero for _ in range(critical_count)]
            )
            rhs_values.append(
                -sum(
                    (left[row] * next_base[row] for row in range(dimension)),
                    zero,
                )
            )
        solution = tuple(solve_linear(augmented, rhs_values))
        if len(solution) != dimension + critical_count:
            raise ValueError("The Puiseux augmented solve returned the wrong dimension")
        coefficients.append(solution[:dimension])
    return tuple(coefficients)


def labeled_local_laurent_coefficients(
    specification: C1FLabeledSpecification,
    radius: Any,
    tree_coefficients: Sequence[Sequence[Any]],
    *,
    denominator_valuation: int,
    half_order: int,
    phase: Any = 1,
    tree_residues: Sequence[int] | None = None,
) -> dict[int, Any]:
    """Expand ``exp(H)/det(I-D)`` at one root-of-unity conjugate."""

    if tree_residues is None:
        tree_residues = (0,) * len(tree_coefficients[0])
    series_order = denominator_valuation + half_order
    zero = radius * phase * 0
    x_coefficients = [zero for _ in range(series_order + 1)]
    x_coefficients[0] = phase * radius
    if series_order >= 2:
        x_coefficients[2] = -phase * radius
    x = TruncatedSeries(tuple(x_coefficients))
    trees = []
    for coordinate, residue in enumerate(tree_residues):
        values = [zero for _ in range(series_order + 1)]
        phase_factor = phase**residue
        for degree in range(min(series_order, len(tree_coefficients) - 1) + 1):
            values[degree] = phase_factor * tree_coefficients[degree][coordinate]
        trees.append(TruncatedSeries(tuple(values)))
    numerator, denominator = formal_labeled_numerator_denominator(
        specification,
        x,
        trees,
    )
    leading_denominator = denominator.coefficient(denominator_valuation)
    quotient: list[Any] = []
    for offset in range(half_order + 1):
        value = numerator.coefficient(offset)
        for denominator_offset in range(1, offset + 1):
            value -= (
                denominator.coefficient(denominator_valuation + denominator_offset)
                * quotient[offset - denominator_offset]
            )
        quotient.append(value / leading_denominator)
    return {
        -denominator_valuation + offset: value
        for offset, value in enumerate(quotient)
    }


def _formal_unlabeled_child_distribution(
    specification: C1FLabeledSpecification,
    root_profile: int,
    levels: Sequence[Sequence[TruncatedSeries] | None],
    index: int,
    omitted_tolls: Sequence[dict[tuple[int, ...], TruncatedSeries] | None] | None = None,
) -> dict[tuple[int, ...], TruncatedSeries]:
    current = levels[index]
    assert current is not None
    order = current[0].order if current else 0
    zero = TruncatedSeries.constant(0, order)
    grouped: dict[tuple[int, ...], TruncatedSeries] = {}
    repeat_limit = (len(levels) - 1) // index
    for repeat in range(1, repeat_limit + 1):
        repeated = levels[index * repeat]
        assert repeated is not None
        for coordinate, child_profile in enumerate(specification.transient_profiles):
            if not specification.successor[child_profile][root_profile]:
                continue
            contribution = tuple(
                min(cap, repeat * value)
                for cap, value in zip(
                    specification.local_caps,
                    specification._local_contribution(child_profile),
                )
            )
            grouped[contribution] = grouped.get(contribution, zero) + (
                repeated[coordinate] / repeat
            )
    if omitted_tolls is not None and omitted_tolls[index] is not None:
        for contribution, value in omitted_tolls[index].items():
            grouped[contribution] = grouped.get(contribution, zero) + value
    zero_state = tuple(0 for _ in specification.local_caps)
    distribution = {zero_state: TruncatedSeries.constant(1, order)}
    for contribution, value in grouped.items():
        distribution = _series_distribution_multiply(
            distribution,
            _series_exponential_factor(
                value,
                contribution,
                specification.local_caps,
            ),
            specification.local_caps,
        )
    return distribution


def formal_unlabeled_tree_map(
    specification: C1FLabeledSpecification,
    x: TruncatedSeries,
    levels: list[Sequence[TruncatedSeries] | None],
    index: int,
    trees: Sequence[TruncatedSeries],
    omitted_tolls: Sequence[dict[tuple[int, ...], TruncatedSeries] | None] | None = None,
) -> tuple[TruncatedSeries, ...]:
    levels[index] = tuple(trees)
    atom = x**index
    return tuple(
        atom
        * specification.multiplicities[root_profile]
        * _series_projected_value(
            specification,
            root_profile,
            _formal_unlabeled_child_distribution(
                specification,
                root_profile,
                levels,
                index,
                omitted_tolls,
            ),
            x.order,
        )
        for root_profile in specification.transient_profiles
    )


def compute_unlabeled_tree_hierarchy(
    specification: C1FLabeledSpecification,
    radius: Any,
    level_values: Sequence[Sequence[Any] | None],
    first_coefficients: Sequence[Any],
    right_directions: Sequence[Sequence[Any]],
    left_directions: Sequence[Sequence[Any]],
    order: int,
    solve_linear: Callable[[Sequence[Sequence[Any]], Sequence[Any]], Sequence[Any]],
    omitted_tolls: Sequence[dict[tuple[int, ...], TruncatedSeries] | None] | None = None,
) -> tuple[tuple[tuple[Any, ...], ...] | None, ...]:
    """Expand the retained Pólya hierarchy in the common local parameter."""

    tail_terms = len(level_values) - 1
    dimension = specification.tree_dimension
    zero = radius * 0
    one = zero + 1
    analytic_order = order + 1
    x_values = [zero for _ in range(analytic_order + 1)]
    x_values[0] = radius
    if order >= 2:
        x_values[2] = -radius
    x = TruncatedSeries(tuple(x_values))
    levels: list[tuple[TruncatedSeries, ...] | None] = [None] * (tail_terms + 1)

    for index in range(tail_terms, 1, -1):
        values = level_values[index]
        assert values is not None
        coefficients: list[tuple[Any, ...]] = [tuple(values)]
        for degree in range(1, analytic_order + 1):
            def residual(trial: Sequence[Any] | None = None) -> tuple[Any, ...]:
                current = []
                for coordinate in range(dimension):
                    series_values = [zero for _ in range(analytic_order + 1)]
                    for known_degree, known in enumerate(coefficients):
                        series_values[known_degree] = known[coordinate]
                    if trial is not None:
                        series_values[degree] = trial[coordinate]
                    current.append(TruncatedSeries(tuple(series_values)))
                levels[index] = tuple(current)
                phi = formal_unlabeled_tree_map(
                    specification,
                    x,
                    levels,
                    index,
                    current,
                    omitted_tolls,
                )
                return tuple(
                    (current[row] - phi[row]).coefficient(degree)
                    for row in range(dimension)
                )

            base = residual()
            columns = []
            for coordinate in range(dimension):
                trial = [zero for _ in range(dimension)]
                trial[coordinate] = one
                candidate = residual(trial)
                columns.append(
                    tuple(candidate[row] - base[row] for row in range(dimension))
                )
            matrix = tuple(
                tuple(columns[column][row] for column in range(dimension))
                for row in range(dimension)
            )
            coefficients.append(tuple(solve_linear(matrix, tuple(-v for v in base))))
        levels[index] = tuple(
            TruncatedSeries(
                tuple(coefficients[degree][coordinate] for degree in range(order + 1))
                + (coefficients[analytic_order][coordinate],)
            )
            for coordinate in range(dimension)
        )

    def critical_rhs(
        local_x: TruncatedSeries,
        trees: Sequence[TruncatedSeries],
    ) -> Sequence[TruncatedSeries]:
        local_levels: list[tuple[TruncatedSeries, ...] | None] = [
            None
        ] * (tail_terms + 1)
        for index in range(2, tail_terms + 1):
            expanded = levels[index]
            assert expanded is not None
            local_levels[index] = tuple(
                value.truncate(local_x.order) for value in expanded
            )
        return formal_unlabeled_tree_map(
            specification,
            local_x,
            local_levels,
            1,
            trees,
            None
            if omitted_tolls is None
            else tuple(
                None
                if toll is None
                else {
                    state: value.truncate(local_x.order)
                    for state, value in toll.items()
                }
                for toll in omitted_tolls
            ),
        )

    critical = compute_tree_puiseux_coefficients(
        specification,
        radius,
        level_values[1] or tuple(),
        first_coefficients,
        right_directions,
        left_directions,
        order,
        solve_linear,
        rhs_builder=critical_rhs,
    )
    levels[1] = tuple(
        TruncatedSeries(
            tuple(critical[degree][coordinate] for degree in range(order + 1))
        )
        for coordinate in range(dimension)
    )
    return tuple(
        None
        if level is None
        else tuple(tuple(series.coefficients) for series in level)
        for level in levels
    )


def _formal_unlabeled_cycle_matrix(
    specification: C1FLabeledSpecification,
    active_profiles: Sequence[int],
    x: TruncatedSeries,
    levels: list[Sequence[TruncatedSeries] | None],
    index: int,
    omitted_tolls: Sequence[dict[tuple[int, ...], TruncatedSeries] | None] | None = None,
) -> tuple[tuple[TruncatedSeries, ...], ...]:
    zero = TruncatedSeries.constant(0, x.order)
    distributions = {
        profile: _formal_unlabeled_child_distribution(
            specification,
            profile,
            levels,
            index,
            omitted_tolls,
        )
        for profile in active_profiles
    }
    atom = x**index
    return tuple(
        tuple(
            zero
            if not specification.successor[previous][current]
            else atom
            * specification.multiplicities[current]
            * _series_projected_value(
                specification,
                current,
                distributions[current],
                x.order,
                specification._local_contribution(previous),
            )
            for current in active_profiles
        )
        for previous in active_profiles
    )


def _totient(value: int) -> int:
    result = value
    remaining = value
    divisor = 2
    while divisor * divisor <= remaining:
        if remaining % divisor == 0:
            while remaining % divisor == 0:
                remaining //= divisor
            result -= result // divisor
        divisor += 1
    if remaining > 1:
        result -= result // remaining
    return result


def unlabeled_local_laurent_coefficients(
    specification: C1FLabeledSpecification,
    radius: Any,
    hierarchy: Sequence[Sequence[Sequence[Any]] | None],
    *,
    denominator_valuation: int,
    half_order: int,
    phase: Any = 1,
    tree_residues: Sequence[int] | None = None,
    omitted_tolls: Sequence[dict[tuple[int, ...], TruncatedSeries] | None] | None = None,
    omitted_log: TruncatedSeries | None = None,
) -> dict[int, Any]:
    """Expand the retained subcritical Pólya OGF at one conjugate."""

    if tree_residues is None:
        tree_residues = (0,) * specification.tree_dimension
    series_order = denominator_valuation + half_order
    zero = radius * phase * 0
    x_values = [zero for _ in range(series_order + 1)]
    x_values[0] = phase * radius
    if series_order >= 2:
        x_values[2] = -phase * radius
    x = TruncatedSeries(tuple(x_values))
    local_omitted_tolls = (
        None
        if omitted_tolls is None
        else tuple(
            None
            if toll is None
            else {
                state: value.truncate(series_order)
                for state, value in toll.items()
            }
            for toll in omitted_tolls
        )
    )
    levels: list[tuple[TruncatedSeries, ...] | None] = [None] * len(hierarchy)
    for index in range(1, len(hierarchy)):
        level = hierarchy[index]
        assert level is not None
        levels[index] = tuple(
            TruncatedSeries(
                tuple(
                    phase ** (index * tree_residues[coordinate]) * coefficient
                    for coefficient in coordinate_coefficients[: series_order + 1]
                )
            )
            for coordinate, coordinate_coefficients in enumerate(level)
        )

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
    cache: dict[tuple[tuple[int, ...], int], tuple[tuple[TruncatedSeries, ...], ...]] = {}

    def matrix(
        profiles: tuple[int, ...],
        index: int,
    ) -> tuple[tuple[TruncatedSeries, ...], ...]:
        key = profiles, index
        if key not in cache:
            cache[key] = _formal_unlabeled_cycle_matrix(
                specification,
                profiles,
                x,
                levels,
                index,
                local_omitted_tolls,
            )
        return cache[key]

    analytic_log = TruncatedSeries.constant(0, series_order)
    if omitted_log is not None:
        analytic_log += omitted_log.truncate(series_order)
    critical_denominator: TruncatedSeries | None = None
    tail_terms = len(levels) - 1
    for outer in range(1, tail_terms + 1):
        for adams in range(1, tail_terms // outer + 1):
            transfer = matrix(long_profiles, outer * adams)
            operator = tuple(
                tuple(
                    TruncatedSeries.constant(
                        1 if row == column else 0,
                        series_order,
                    )
                    - transfer[row][column]
                    for column in range(len(transfer))
                )
                for row in range(len(transfer))
            )
            determinant = _series_determinant(operator, series_order)
            if outer == 1 and adams == 1:
                critical_denominator = determinant
            else:
                analytic_log -= (
                    determinant.log() * _totient(adams) / (outer * adams)
                )
        for cycle_length, profiles in enumerate(visible_profiles, 1):
            correction = TruncatedSeries.constant(0, series_order)
            for adams in range(1, cycle_length + 1):
                if cycle_length % adams or outer * adams > tail_terms:
                    continue
                exponent = cycle_length // adams
                visible = _series_matrix_power(matrix(profiles, outer * adams), exponent)
                long = _series_matrix_power(matrix(long_profiles, outer * adams), exponent)
                correction += _totient(adams) * (
                    sum(
                        (visible[index][index] for index in range(len(visible))),
                        TruncatedSeries.constant(0, series_order),
                    )
                    - sum(
                        (long[index][index] for index in range(len(long))),
                        TruncatedSeries.constant(0, series_order),
                    )
                )
            analytic_log += correction / (outer * cycle_length)
    if critical_denominator is None:
        raise ValueError("The Pólya hierarchy has no primary cycle determinant")
    numerator = analytic_log.exp()
    leading_denominator = critical_denominator.coefficient(denominator_valuation)
    quotient: list[Any] = []
    for offset in range(half_order + 1):
        value = numerator.coefficient(offset)
        for denominator_offset in range(1, offset + 1):
            value -= (
                critical_denominator.coefficient(
                    denominator_valuation + denominator_offset
                )
                * quotient[offset - denominator_offset]
            )
        quotient.append(value / leading_denominator)
    return {
        -denominator_valuation + offset: value
        for offset, value in enumerate(quotient)
    }


def _binomial_series(exponent: Fraction, order: int) -> tuple[Fraction, ...]:
    result = [Fraction(1)]
    for degree in range(1, order + 1):
        result.append(
            result[-1] * (exponent - degree + 1) / degree
        )
    return tuple(result)


def gamma_ratio_coefficients(beta: Fraction, order: int) -> tuple[Fraction, ...]:
    """Return the asymptotic series of Gamma(n+beta)/Gamma(n+1)."""

    rhs_base = list(_binomial_series(-beta, order + 1))
    rhs = rhs_base[:]
    for degree in range(1, order + 2):
        rhs[degree] += beta * rhs_base[degree - 1]
    coefficients = [Fraction(1)]
    for selected in range(1, order + 1):
        target = selected + 1

        def equation(candidate: Fraction) -> Fraction:
            values = coefficients + [candidate]
            composed = Fraction(0)
            for degree, value in enumerate(values):
                if degree <= target:
                    composed += value * _binomial_series(
                        Fraction(-degree),
                        target - degree,
                    )[target - degree]
            product_value = sum(
                (
                    rhs[offset] * values[target - offset]
                    for offset in range(target + 1)
                    if 0 <= target - offset < len(values)
                ),
                Fraction(0),
            )
            return composed - product_value

        base = equation(Fraction(0))
        slope = equation(Fraction(1)) - base
        coefficients.append(-base / slope)
    return tuple(coefficients)


def project_complex_laurent_orbit(
    local_coefficients: Sequence[dict[int, complex]],
    phases: Sequence[complex],
    *,
    period: int,
    admitted_residues: Sequence[int],
    leading_power: float,
    half_order: int,
    tolerance: float,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Transfer and Fourier-project local Laurent-Puiseux coefficients."""

    minimum_power = leading_power - half_order / 2.0
    admitted = frozenset(admitted_residues)
    result: list[tuple[tuple[float, float], ...]] = []
    for residue in range(period):
        if residue not in admitted:
            result.append(tuple())
            continue
        by_half_step: dict[int, complex] = {}
        scale: dict[int, float] = {}
        for phase_index, (phase, coefficients) in enumerate(
            zip(phases, local_coefficients)
        ):
            phase_factor = phase ** (-residue)
            for local_exponent, local_coefficient in coefficients.items():
                beta = Fraction(-local_exponent, 2)
                if beta.denominator == 1 and beta <= 0:
                    continue
                base_power = float(beta) - 1.0
                if base_power < minimum_power - 1e-12:
                    continue
                correction_order = int((base_power - minimum_power) + 1e-9)
                gamma_corrections = gamma_ratio_coefficients(beta, correction_order)
                gamma_value = gamma(float(beta))
                for correction, ratio in enumerate(gamma_corrections):
                    power = base_power - correction
                    if power < minimum_power - 1e-12:
                        continue
                    half_step = int(round(2.0 * (leading_power - power)))
                    contribution = (
                        phase_factor
                        * local_coefficient
                        * float(ratio)
                        / gamma_value
                    )
                    by_half_step[half_step] = (
                        by_half_step.get(half_step, 0j) + contribution
                    )
                    scale[half_step] = scale.get(half_step, 0.0) + abs(contribution)
        terms = []
        for half_step in sorted(by_half_step):
            value = by_half_step[half_step]
            threshold = tolerance * max(1.0, scale[half_step])
            if abs(value.imag) > 100.0 * threshold:
                raise ValueError(
                    "Periodic Puiseux contributions did not project to a real value"
                )
            if abs(value.real) <= threshold:
                continue
            terms.append((leading_power - half_step / 2.0, value.real))
        result.append(tuple(terms))
    return tuple(result)
