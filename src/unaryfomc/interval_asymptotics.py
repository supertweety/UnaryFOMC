"""Optional Arb certificates for labeled and unlabeled C1F asymptotics.

The ordinary analyzer supplies a floating-point seed.  This module rebuilds
the finite guarded specification with Arb balls and proves a nearby critical
solution using a Krawczyk inclusion.  It is imported only when certification
is requested.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from math import factorial, gcd, log
from typing import Any, Callable, Sequence

from .asymptotics import (
    C1FAsymptoticCertificate,
    C1FAsymptoticCertificationError,
    C1FCertifiedAsymptoticTerm,
    C1FRealInterval,
)
from .puiseux import (
    TruncatedSeries,
    compute_tree_puiseux_coefficients,
    compute_unlabeled_tree_hierarchy,
    gamma_ratio_coefficients,
    labeled_local_laurent_coefficients,
    unlabeled_local_laurent_coefficients,
)
from .specification import C1FLabeledSpecification

try:
    from flint import acb, acb_mat, arb, arb_mat, ctx, fmpz_poly
except ImportError:  # pragma: no cover - exercised on Python 3.10 installations
    acb = None
    acb_mat = None
    arb = None
    arb_mat = None
    ctx = None
    fmpz_poly = None


@dataclass(frozen=True)
class _Jet:
    value: Any
    gradient: tuple[Any, ...]

    @classmethod
    def constant(cls, value: Any, dimension: int) -> "_Jet":
        return cls(_as_arb(value), tuple(arb(0) for _ in range(dimension)))

    @classmethod
    def variable(cls, value: Any, index: int, dimension: int) -> "_Jet":
        gradient = [arb(0) for _ in range(dimension)]
        gradient[index] = arb(1)
        return cls(_as_arb(value), tuple(gradient))

    def _coerce(self, other: Any) -> "_Jet":
        if isinstance(other, _Jet):
            if len(other.gradient) != len(self.gradient):
                raise ValueError("Jet dimensions do not agree")
            return other
        return _Jet.constant(other, len(self.gradient))

    def __add__(self, other: Any) -> "_Jet":
        other = self._coerce(other)
        return _Jet(
            self.value + other.value,
            tuple(a + b for a, b in zip(self.gradient, other.gradient)),
        )

    def __radd__(self, other: Any) -> "_Jet":
        return self + other

    def __neg__(self) -> "_Jet":
        return _Jet(-self.value, tuple(-value for value in self.gradient))

    def __sub__(self, other: Any) -> "_Jet":
        return self + (-self._coerce(other))

    def __rsub__(self, other: Any) -> "_Jet":
        return self._coerce(other) - self

    def __mul__(self, other: Any) -> "_Jet":
        other = self._coerce(other)
        return _Jet(
            self.value * other.value,
            tuple(
                a * other.value + self.value * b
                for a, b in zip(self.gradient, other.gradient)
            ),
        )

    def __rmul__(self, other: Any) -> "_Jet":
        return self * other

    def __truediv__(self, other: Any) -> "_Jet":
        other = self._coerce(other)
        denominator = other.value * other.value
        return _Jet(
            self.value / other.value,
            tuple(
                (a * other.value - self.value * b) / denominator
                for a, b in zip(self.gradient, other.gradient)
            ),
        )

    def __pow__(self, exponent: int) -> "_Jet":
        if exponent < 0:
            return _Jet.constant(1, len(self.gradient)) / (self ** (-exponent))
        result = _Jet.constant(1, len(self.gradient))
        factor = self
        remaining = exponent
        while remaining:
            if remaining & 1:
                result = result * factor
            remaining >>= 1
            if remaining:
                factor = factor * factor
        return result

    def exp(self) -> "_Jet":
        value = self.value.exp()
        return _Jet(value, tuple(value * item for item in self.gradient))

    def log(self) -> "_Jet":
        return _Jet(
            self.value.log(),
            tuple(item / self.value for item in self.gradient),
        )


@dataclass(frozen=True)
class _AcbDirectional:
    """An Acb ball with one formal directional derivative."""

    value: Any
    derivative: Any

    @classmethod
    def constant(cls, value: Any) -> "_AcbDirectional":
        return cls(value if isinstance(value, acb) else acb(value), acb(0))

    def _coerce(self, other: Any) -> "_AcbDirectional":
        return other if isinstance(other, _AcbDirectional) else self.constant(other)

    def __add__(self, other: Any) -> "_AcbDirectional":
        other = self._coerce(other)
        return _AcbDirectional(
            self.value + other.value,
            self.derivative + other.derivative,
        )

    __radd__ = __add__

    def __neg__(self) -> "_AcbDirectional":
        return _AcbDirectional(-self.value, -self.derivative)

    def __sub__(self, other: Any) -> "_AcbDirectional":
        return self + (-self._coerce(other))

    def __rsub__(self, other: Any) -> "_AcbDirectional":
        return self._coerce(other) - self

    def __mul__(self, other: Any) -> "_AcbDirectional":
        other = self._coerce(other)
        return _AcbDirectional(
            self.value * other.value,
            self.derivative * other.value + self.value * other.derivative,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "_AcbDirectional":
        other = self._coerce(other)
        return _AcbDirectional(
            self.value / other.value,
            (
                self.derivative * other.value
                - self.value * other.derivative
            ) / (other.value * other.value),
        )

    def exp(self) -> "_AcbDirectional":
        value = self.value.exp()
        return _AcbDirectional(value, value * self.derivative)


def _require_arb() -> None:
    if arb is None or arb_mat is None or ctx is None:
        raise RuntimeError(
            "Interval asymptotic certification requires python-flint with Arb support"
        )


def _integer_totient(value: int) -> int:
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


def _as_arb(value: Any):
    if isinstance(value, arb):
        return value
    return arb(value)


def _capped_add(
    left: tuple[int, ...],
    right: tuple[int, ...],
    caps: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        min(cap, left_value + right_value)
        for cap, left_value, right_value in zip(caps, left, right)
    )


def _distribution_multiply(
    left: dict[tuple[int, ...], _Jet],
    right: dict[tuple[int, ...], _Jet],
    caps: tuple[int, ...],
) -> dict[tuple[int, ...], _Jet]:
    result: dict[tuple[int, ...], _Jet] = {}
    for left_state, left_value in left.items():
        for right_state, right_value in right.items():
            state = _capped_add(left_state, right_state, caps)
            result[state] = result.get(
                state,
                _Jet.constant(0, len(left_value.gradient)),
            ) + left_value * right_value
    return result


def _capped_exponential_factor(
    value: _Jet,
    contribution: tuple[int, ...],
    caps: tuple[int, ...],
) -> dict[tuple[int, ...], _Jet]:
    zero = tuple(0 for _ in caps)
    if not any(contribution):
        return {zero: value.exp()}

    saturation_count = max(
        (cap + increment - 1) // increment
        for cap, increment in zip(caps, contribution)
        if increment
    )
    result: dict[tuple[int, ...], _Jet] = {}
    term = _Jet.constant(1, len(value.gradient))
    partial = _Jet.constant(0, len(value.gradient))
    for count in range(saturation_count):
        state = tuple(
            min(cap, count * increment)
            for cap, increment in zip(caps, contribution)
        )
        result[state] = result.get(
            state,
            _Jet.constant(0, len(value.gradient)),
        ) + term
        partial = partial + term
        term = term * value / (count + 1)

    saturated = tuple(
        cap if increment else 0
        for cap, increment in zip(caps, contribution)
    )
    result[saturated] = result.get(
        saturated,
        _Jet.constant(0, len(value.gradient)),
    ) + value.exp() - partial
    return result


def _local_contribution(
    specification: C1FLabeledSpecification,
    profile: int,
) -> tuple[int, ...]:
    if not specification.local_contributions:
        return tuple()
    return specification.local_contributions[profile]


def _accepted_states(
    specification: C1FLabeledSpecification,
    profile: int,
) -> frozenset[tuple[int, ...]]:
    if not specification.local_acceptance:
        return frozenset((tuple(),))
    return specification.local_acceptance[profile]


def _child_distribution(
    specification: C1FLabeledSpecification,
    root_profile: int,
    trees: Sequence[_Jet],
    dimension: int | None = None,
) -> dict[tuple[int, ...], _Jet]:
    if dimension is None:
        dimension = len(trees[0].gradient) if trees else 0
    grouped: dict[tuple[int, ...], _Jet] = {}
    for child_index, child_profile in enumerate(specification.transient_profiles):
        if not specification.successor[child_profile][root_profile]:
            continue
        contribution = _local_contribution(specification, child_profile)
        grouped[contribution] = grouped.get(
            contribution,
            _Jet.constant(0, dimension),
        ) + trees[child_index]

    zero = tuple(0 for _ in specification.local_caps)
    distribution = {zero: _Jet.constant(1, dimension)}
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


def _projected_value(
    specification: C1FLabeledSpecification,
    root_profile: int,
    distribution: dict[tuple[int, ...], _Jet],
    dimension: int,
    extra: tuple[int, ...] | None = None,
) -> _Jet:
    accepted = _accepted_states(specification, root_profile)
    total = _Jet.constant(0, dimension)
    for state, value in distribution.items():
        final_state = (
            state
            if extra is None
            else _capped_add(state, extra, specification.local_caps)
        )
        if final_state in accepted:
            total = total + value
    return total


def _tree_phi(
    specification: C1FLabeledSpecification,
    x: _Jet,
    trees: Sequence[_Jet],
) -> tuple[_Jet, ...]:
    dimension = len(x.gradient)
    return tuple(
        x
        * specification.multiplicities[root_profile]
        * _projected_value(
            specification,
            root_profile,
            _child_distribution(
                specification,
                root_profile,
                trees,
                dimension,
            ),
            dimension,
        )
        for root_profile in specification.transient_profiles
    )


def _cycle_matrix(
    specification: C1FLabeledSpecification,
    active_profiles: Sequence[int],
    x: _Jet,
    trees: Sequence[_Jet],
) -> tuple[tuple[_Jet, ...], ...]:
    dimension = len(x.gradient)
    distributions = {
        current: _child_distribution(
            specification,
            current,
            trees,
            dimension,
        )
        for current in active_profiles
    }
    return tuple(
        tuple(
            _Jet.constant(0, dimension)
            if not specification.successor[previous][current]
            else x
            * specification.multiplicities[current]
            * _projected_value(
                specification,
                current,
                distributions[current],
                dimension,
                _local_contribution(specification, previous),
            )
            for current in active_profiles
        )
        for previous in active_profiles
    )


def _long_profiles(
    specification: C1FLabeledSpecification,
) -> tuple[int, ...]:
    if specification.long_cycle_profiles is None:
        return specification.transient_profiles
    return specification.long_cycle_profiles


def _sum_jets(values: Sequence[_Jet], dimension: int) -> _Jet:
    total = _Jet.constant(0, dimension)
    for value in values:
        total = total + value
    return total


def _regular_system(
    specification: C1FLabeledSpecification,
    values: Sequence[Any],
    critical_component: Sequence[int] | None = None,
) -> tuple[_Jet, ...]:
    tree_dimension = specification.tree_dimension
    if critical_component is None:
        critical_component = tuple(range(tree_dimension))
    critical_component = tuple(critical_component)
    critical_dimension = len(critical_component)
    dimension = tree_dimension + 2 * critical_dimension + 1
    variables = tuple(
        _Jet.variable(value, index, dimension)
        for index, value in enumerate(values)
    )
    x = variables[0].exp()
    trees = tuple(variables[1 + index].exp() for index in range(tree_dimension))
    right_offset = 1 + tree_dimension
    right = tuple(
        variables[right_offset + index].exp()
        for index in range(critical_dimension)
    )
    left_offset = right_offset + critical_dimension
    left = tuple(
        variables[left_offset + index].exp()
        for index in range(critical_dimension)
    )

    phi = _tree_phi(specification, x, trees)
    transfer = _cycle_matrix(
        specification,
        _long_profiles(specification),
        x,
        trees,
    )
    long_positions = {
        profile: index
        for index, profile in enumerate(_long_profiles(specification))
    }
    transient = specification.transient_profiles
    jacobian = tuple(
        tuple(
            transfer[
                long_positions[transient[critical_component[column]]]
            ][long_positions[transient[critical_component[row]]]]
            for column in range(critical_dimension)
        )
        for row in range(critical_dimension)
    )

    equations = [
        variables[1 + index] - phi[index].log()
        for index in range(tree_dimension)
    ]
    equations.extend(
        _sum_jets(
            tuple(
                jacobian[row][column] * right[column]
                for column in range(critical_dimension)
            ),
            dimension,
        )
        - right[row]
        for row in range(critical_dimension)
    )
    equations.append(_sum_jets(right, dimension) - 1)
    equations.extend(
        _sum_jets(
            tuple(
                left[row] * jacobian[row][column]
                for row in range(critical_dimension)
            ),
            dimension,
        )
        - left[column]
        for column in range(critical_dimension - 1)
    )
    equations.append(
        _sum_jets(
            tuple(a * b for a, b in zip(left, right)),
            dimension,
        )
        - 1
    )
    if len(equations) != dimension:
        raise C1FAsymptoticCertificationError(
            "Internal error: the regular certification system is not square"
        )
    return tuple(equations)


def _full_tree_jacobian(
    specification: C1FLabeledSpecification,
    radius: Any,
    trees: Sequence[Any],
):
    tree_dimension = specification.tree_dimension
    x = _Jet.constant(radius, 0)
    tree_jets = tuple(_Jet.constant(value, 0) for value in trees)
    transfer = _cycle_matrix(
        specification,
        _long_profiles(specification),
        x,
        tree_jets,
    )
    long_positions = {
        profile: index
        for index, profile in enumerate(_long_profiles(specification))
    }
    transient = specification.transient_profiles
    return arb_mat(
        tree_dimension,
        tree_dimension,
        tuple(
            transfer[long_positions[transient[column]]][
                long_positions[transient[row]]
            ].value
            for row in range(tree_dimension)
            for column in range(tree_dimension)
        ),
    )


def _normalized_full_kernel_vectors(
    jacobian: Any,
    critical_component: Sequence[int],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Enclose full right and left kernels using a critical-block gauge."""

    dimension = jacobian.nrows()
    pivot = critical_component[-1]
    operator = _arb_identity(dimension) - jacobian
    constrained_right = arb_mat(operator)
    for column in range(dimension):
        constrained_right[pivot, column] = (
            1 if column in critical_component else 0
        )
    rhs = arb_mat(dimension, 1)
    rhs[pivot, 0] = 1
    try:
        right_column = constrained_right.solve(rhs)
    except ZeroDivisionError as exc:
        raise C1FAsymptoticCertificationError(
            "The full right critical direction could not be isolated"
        ) from exc

    transpose = arb_mat(
        dimension,
        dimension,
        tuple(
            operator[column, row]
            for row in range(dimension)
            for column in range(dimension)
        ),
    )
    constrained_left = arb_mat(transpose)
    for column in range(dimension):
        constrained_left[pivot, column] = (
            1 if column in critical_component else 0
        )
    try:
        left_column = constrained_left.solve(rhs)
    except ZeroDivisionError as exc:
        raise C1FAsymptoticCertificationError(
            "The full left critical direction could not be isolated"
        ) from exc

    right = tuple(right_column[index, 0] for index in range(dimension))
    raw_left = tuple(left_column[index, 0] for index in range(dimension))
    normalization = sum(
        (a * b for a, b in zip(raw_left, right)),
        arb(0),
    )
    _require_positive("left-right Perron normalization", normalization)
    left = tuple(value / normalization for value in raw_left)
    return right, left


def _cycle_pole_system(
    specification: C1FLabeledSpecification,
    values: Sequence[Any],
    active_profiles: Sequence[int] | None = None,
) -> tuple[_Jet, ...]:
    if active_profiles is None:
        active_profiles = _long_profiles(specification)
    cycle_dimension = len(active_profiles)
    dimension = 2 * cycle_dimension + 1
    variables = tuple(
        _Jet.variable(value, index, dimension)
        for index, value in enumerate(values)
    )
    x = variables[0].exp()
    right = tuple(
        variables[1 + index].exp()
        for index in range(cycle_dimension)
    )
    left_offset = 1 + cycle_dimension
    left = tuple(
        variables[left_offset + index].exp()
        for index in range(cycle_dimension)
    )
    zero_trees = tuple(
        _Jet.constant(0, dimension)
        for _ in range(specification.tree_dimension)
    )
    transfer = _cycle_matrix(
        specification,
        active_profiles,
        x,
        zero_trees,
    )

    equations = [
        _sum_jets(
            tuple(transfer[row][column] * right[column] for column in range(cycle_dimension)),
            dimension,
        )
        - right[row]
        for row in range(cycle_dimension)
    ]
    equations.append(_sum_jets(right, dimension) - 1)
    equations.extend(
        _sum_jets(
            tuple(left[row] * transfer[row][column] for row in range(cycle_dimension)),
            dimension,
        )
        - left[column]
        for column in range(cycle_dimension - 1)
    )
    equations.append(
        _sum_jets(
            tuple(a * b for a, b in zip(left, right)),
            dimension,
        )
        - 1
    )
    if len(equations) != dimension:
        raise C1FAsymptoticCertificationError(
            "Internal error: the cycle-pole certification system is not square"
        )
    return tuple(equations)


def _analytic_tree_cycle_pole_system(
    specification: C1FLabeledSpecification,
    critical_component: Sequence[int],
    values: Sequence[Any],
) -> tuple[_Jet, ...]:
    """Joint analytic-tree and cycle-Perron equations in log coordinates."""

    tree_dimension = specification.tree_dimension
    critical_component = tuple(critical_component)
    critical_dimension = len(critical_component)
    dimension = 1 + tree_dimension + 2 * critical_dimension
    variables = tuple(
        _Jet.variable(value, index, dimension)
        for index, value in enumerate(values)
    )
    x = variables[0].exp()
    trees = tuple(
        variables[1 + index].exp()
        for index in range(tree_dimension)
    )
    right_offset = 1 + tree_dimension
    right = tuple(
        variables[right_offset + index].exp()
        for index in range(critical_dimension)
    )
    left_offset = right_offset + critical_dimension
    left = tuple(
        variables[left_offset + index].exp()
        for index in range(critical_dimension)
    )

    phi = _tree_phi(specification, x, trees)
    full_transfer = _cycle_matrix(
        specification,
        _long_profiles(specification),
        x,
        trees,
    )
    transfer = tuple(
        tuple(
            full_transfer[row][column]
            for column in critical_component
        )
        for row in critical_component
    )
    equations = [
        variables[1 + index] - phi[index].log()
        for index in range(tree_dimension)
    ]
    equations.extend(
        _sum_jets(
            tuple(
                transfer[row][column] * right[column]
                for column in range(critical_dimension)
            ),
            dimension,
        )
        - right[row]
        for row in range(critical_dimension)
    )
    equations.append(_sum_jets(right, dimension) - 1)
    equations.extend(
        _sum_jets(
            tuple(
                left[row] * transfer[row][column]
                for row in range(critical_dimension)
            ),
            dimension,
        )
        - left[column]
        for column in range(critical_dimension - 1)
    )
    equations.append(
        _sum_jets(
            tuple(a * b for a, b in zip(left, right)),
            dimension,
        )
        - 1
    )
    if len(equations) != dimension:
        raise C1FAsymptoticCertificationError(
            "Internal error: the analytic cycle-pole system is not square"
        )
    return tuple(equations)


def _arb_column(values: Sequence[Any]):
    return arb_mat(len(values), 1, tuple(values))


def _arb_identity(size: int):
    matrix = arb_mat(size, size)
    for index in range(size):
        matrix[index, index] = 1
    return matrix


def _system_values_and_jacobian(
    system: Callable[[Sequence[Any]], tuple[_Jet, ...]],
    values: Sequence[Any],
):
    equations = system(values)
    dimension = len(equations)
    return (
        _arb_column(tuple(equation.value for equation in equations)),
        arb_mat(
            dimension,
            dimension,
            tuple(
                derivative
                for equation in equations
                for derivative in equation.gradient
            ),
        ),
    )


def _strictly_inside(inner: Any, outer: Any) -> bool:
    return bool(inner.lower() > outer.lower() and inner.upper() < outer.upper())


def _krawczyk_box(
    system: Callable[[Sequence[Any]], tuple[_Jet, ...]],
    centers: Sequence[float],
) -> tuple[Any, ...]:
    center = tuple(arb(value) for value in centers)
    for _ in range(12):
        values, jacobian = _system_values_and_jacobian(system, center)
        try:
            correction = jacobian.mid().solve(values)
        except ZeroDivisionError as exc:
            raise C1FAsymptoticCertificationError(
                "The midpoint Jacobian of the certification system is singular"
            ) from exc
        center = tuple(
            (value - correction[index, 0]).mid()
            for index, value in enumerate(center)
        )
        if max(
            float(correction[index, 0].abs_upper())
            for index in range(len(center))
        ) < 1e-30:
            break

    center_values, center_jacobian = _system_values_and_jacobian(system, center)
    try:
        inverse = center_jacobian.mid().inv().mid()
    except ZeroDivisionError as exc:
        raise C1FAsymptoticCertificationError(
            "The midpoint Jacobian of the certification system is singular"
        ) from exc

    identity = _arb_identity(len(center))
    center_column = _arb_column(center)
    radius_exponents = (48, 40, 32, 24, 18, 14, 12, 10, 8, 6, 4)
    for exponent in radius_exponents:
        radius_text = f"1e-{exponent}"
        box = tuple(arb(value, radius_text) for value in center)
        _, interval_jacobian = _system_values_and_jacobian(system, box)
        displacement = _arb_column(
            tuple(value - midpoint for value, midpoint in zip(box, center))
        )
        image = (
            center_column
            - inverse * center_values
            + (identity - inverse * interval_jacobian) * displacement
        )
        if all(
            _strictly_inside(image[index, 0], box[index])
            for index in range(len(center))
        ):
            return tuple(image[index, 0] for index in range(len(center)))
    raise C1FAsymptoticCertificationError(
        "Arb could not prove a Krawczyk inclusion around the numerical critical point"
    )


def _arb_matrix(entries: Sequence[Sequence[_Jet]]):
    size = len(entries)
    return arb_mat(
        size,
        size,
        tuple(entry.value for row in entries for entry in row),
    )


def _minor_matrix(entries: Sequence[Sequence[_Jet]], skip_row: int, skip_col: int):
    reduced = [
        [entry.value for column, entry in enumerate(row) if column != skip_col]
        for row_index, row in enumerate(entries)
        if row_index != skip_row
    ]
    if not reduced:
        return arb_mat(0, 0)
    return arb_mat(reduced)


def _determinant_jet(entries: Sequence[Sequence[_Jet]]) -> _Jet:
    size = len(entries)
    if size == 0:
        return _Jet.constant(1, 0)
    dimension = len(entries[0][0].gradient)
    value = _arb_matrix(entries).det()
    cofactors = [
        [
            ((-1) ** (row + column))
            * _minor_matrix(entries, row, column).det()
            for column in range(size)
        ]
        for row in range(size)
    ]
    gradient = tuple(
        sum(
            (
                cofactors[row][column] * entries[row][column].gradient[index]
                for row in range(size)
                for column in range(size)
            ),
            arb(0),
        )
        for index in range(dimension)
    )
    return _Jet(value, gradient)


def _identity_minus(entries: Sequence[Sequence[_Jet]]) -> tuple[tuple[_Jet, ...], ...]:
    dimension = len(entries[0][0].gradient) if entries else 0
    return tuple(
        tuple(
            _Jet.constant(1 if row == column else 0, dimension)
            - entries[row][column]
            for column in range(len(entries))
        )
        for row in range(len(entries))
    )


def _finite_cycle_correction(
    specification: C1FLabeledSpecification,
    x: Any,
    trees: Sequence[Any],
):
    constant_x = _Jet.constant(x, 0)
    constant_trees = tuple(_Jet.constant(value, 0) for value in trees)
    long_matrix = _arb_matrix(
        _cycle_matrix(
            specification,
            _long_profiles(specification),
            constant_x,
            constant_trees,
        )
    )
    correction = arb(0)
    for cycle_length in range(1, specification.depth + 1):
        visible_profiles = tuple(
            profile
            for profile, visible_length in enumerate(
                specification.visible_cycle_lengths
            )
            if visible_length == cycle_length
        )
        visible_matrix = _arb_matrix(
            _cycle_matrix(
                specification,
                visible_profiles,
                constant_x,
                constant_trees,
            )
        )
        correction += (
            (visible_matrix ** cycle_length).trace()
            - (long_matrix ** cycle_length).trace()
        ) / cycle_length
    return correction


def _acb_distribution_multiply(
    left: dict[tuple[int, ...], _AcbDirectional],
    right: dict[tuple[int, ...], _AcbDirectional],
    caps: tuple[int, ...],
) -> dict[tuple[int, ...], _AcbDirectional]:
    result: dict[tuple[int, ...], _AcbDirectional] = {}
    for left_state, left_value in left.items():
        for right_state, right_value in right.items():
            state = _capped_add(left_state, right_state, caps)
            result[state] = result.get(
                state,
                _AcbDirectional.constant(0),
            ) + left_value * right_value
    return result


def _acb_exponential_factor(
    value: _AcbDirectional,
    contribution: tuple[int, ...],
    caps: tuple[int, ...],
) -> dict[tuple[int, ...], _AcbDirectional]:
    zero = tuple(0 for _ in caps)
    if not any(contribution):
        return {zero: value.exp()}
    saturation_count = max(
        (cap + increment - 1) // increment
        for cap, increment in zip(caps, contribution)
        if increment
    )
    result: dict[tuple[int, ...], _AcbDirectional] = {}
    term = _AcbDirectional.constant(1)
    partial = _AcbDirectional.constant(0)
    for count in range(saturation_count):
        state = tuple(
            min(cap, count * increment)
            for cap, increment in zip(caps, contribution)
        )
        result[state] = result.get(
            state,
            _AcbDirectional.constant(0),
        ) + term
        partial += term
        term = term * value / (count + 1)
    saturated = tuple(
        cap if increment else 0
        for cap, increment in zip(caps, contribution)
    )
    result[saturated] = (
        result.get(saturated, _AcbDirectional.constant(0))
        + value.exp()
        - partial
    )
    return result


def _acb_child_distribution(
    specification: C1FLabeledSpecification,
    root_profile: int,
    trees: Sequence[_AcbDirectional],
) -> dict[tuple[int, ...], _AcbDirectional]:
    grouped: dict[tuple[int, ...], _AcbDirectional] = {}
    for coordinate, child_profile in enumerate(specification.transient_profiles):
        if not specification.successor[child_profile][root_profile]:
            continue
        contribution = _local_contribution(specification, child_profile)
        grouped[contribution] = grouped.get(
            contribution,
            _AcbDirectional.constant(0),
        ) + trees[coordinate]
    zero = tuple(0 for _ in specification.local_caps)
    distribution = {zero: _AcbDirectional.constant(1)}
    for contribution, value in grouped.items():
        distribution = _acb_distribution_multiply(
            distribution,
            _acb_exponential_factor(
                value,
                contribution,
                specification.local_caps,
            ),
            specification.local_caps,
        )
    return distribution


def _acb_projected_value(
    specification: C1FLabeledSpecification,
    root_profile: int,
    distribution: dict[tuple[int, ...], _AcbDirectional],
    extra: tuple[int, ...] | None = None,
) -> _AcbDirectional:
    total = _AcbDirectional.constant(0)
    accepted = _accepted_states(specification, root_profile)
    for state, value in distribution.items():
        final_state = (
            state
            if extra is None
            else _capped_add(state, extra, specification.local_caps)
        )
        if final_state in accepted:
            total += value
    return total


def _acb_cycle_matrix(
    specification: C1FLabeledSpecification,
    active_profiles: Sequence[int],
    x: _AcbDirectional,
    trees: Sequence[_AcbDirectional],
) -> tuple[tuple[_AcbDirectional, ...], ...]:
    distributions = {
        profile: _acb_child_distribution(specification, profile, trees)
        for profile in active_profiles
    }
    return tuple(
        tuple(
            _AcbDirectional.constant(0)
            if not specification.successor[previous][current]
            else x
            * specification.multiplicities[current]
            * _acb_projected_value(
                specification,
                current,
                distributions[current],
                _local_contribution(specification, previous),
            )
            for current in active_profiles
        )
        for previous in active_profiles
    )


def _acb_determinant_direction(
    entries: Sequence[Sequence[_AcbDirectional]],
) -> tuple[Any, Any]:
    size = len(entries)
    if size == 0:
        return acb(1), acb(0)
    values = acb_mat(
        size,
        size,
        tuple(entry.value for row in entries for entry in row),
    )
    determinant = values.det()
    derivative = acb(0)
    for row in range(size):
        for column in range(size):
            minor_values = [
                entries[other_row][other_column].value
                for other_row in range(size)
                if other_row != row
                for other_column in range(size)
                if other_column != column
            ]
            cofactor = (
                acb(1)
                if size == 1
                else (-1) ** (row + column)
                * acb_mat(size - 1, size - 1, tuple(minor_values)).det()
            )
            derivative += cofactor * entries[row][column].derivative
    return determinant, derivative


def _acb_finite_cycle_correction(
    specification: C1FLabeledSpecification,
    x: _AcbDirectional,
    trees: Sequence[_AcbDirectional],
) -> _AcbDirectional:
    long_matrix = _acb_cycle_matrix(
        specification,
        _long_profiles(specification),
        x,
        trees,
    )
    correction = _AcbDirectional.constant(0)
    for cycle_length in range(1, specification.depth + 1):
        visible_profiles = tuple(
            profile
            for profile, visible_length in enumerate(
                specification.visible_cycle_lengths
            )
            if visible_length == cycle_length
        )
        visible_matrix = _acb_cycle_matrix(
            specification,
            visible_profiles,
            x,
            trees,
        )

        def trace_power(
            entries: Sequence[Sequence[_AcbDirectional]],
            exponent: int,
        ) -> _AcbDirectional:
            size = len(entries)
            result = tuple(
                tuple(
                    _AcbDirectional.constant(1 if row == column else 0)
                    for column in range(size)
                )
                for row in range(size)
            )
            factor = tuple(tuple(row) for row in entries)
            for _ in range(exponent):
                result = tuple(
                    tuple(
                        sum(
                            (
                                result[row][middle] * factor[middle][column]
                                for middle in range(size)
                            ),
                            _AcbDirectional.constant(0),
                        )
                        for column in range(size)
                    )
                    for row in range(size)
                )
            return sum(
                (result[index][index] for index in range(size)),
                _AcbDirectional.constant(0),
            )

        correction += (
            trace_power(visible_matrix, cycle_length)
            - trace_power(long_matrix, cycle_length)
        ) / cycle_length
    return correction


def _certify_periodic_labeled_residues(
    specification: C1FLabeledSpecification,
    orbit: Any,
    analysis: Any,
    radius: Any,
    trees: Sequence[Any],
    tree_sqrt: Sequence[Any],
    precision_bits: int,
) -> tuple[tuple[C1FRealInterval | None, ...], C1FRealInterval]:
    """Evaluate and Fourier-project all conjugate square-root constants."""

    root = (acb(0, 2 * arb.pi() / orbit.period)).exp()
    local_constants = []
    for phase_index in range(orbit.period):
        phase = root**phase_index
        phased_trees = tuple(
            _AcbDirectional(
                phase**residue * value,
                -(phase**residue) * coefficient,
            )
            for residue, value, coefficient in zip(
                orbit.tree_residues,
                trees,
                tree_sqrt,
            )
        )
        x = _AcbDirectional.constant(phase * radius)
        transfer = _acb_cycle_matrix(
            specification,
            _long_profiles(specification),
            x,
            phased_trees,
        )
        operator = tuple(
            tuple(
                _AcbDirectional.constant(1 if row == column else 0) - entry
                for column, entry in enumerate(matrix_row)
            )
            for row, matrix_row in enumerate(transfer)
        )
        _, delta = _acb_determinant_direction(operator)
        if delta.contains(0):
            raise C1FAsymptoticCertificationError(
                "A periodic determinant coefficient contains zero"
            )
        correction = _acb_finite_cycle_correction(
            specification,
            x,
            phased_trees,
        ).value
        local_constants.append(
            correction.exp() / (delta * arb.pi().sqrt())
        )

    admitted = frozenset(orbit.object_residues)
    residue_constants: list[C1FRealInterval | None] = []
    for residue in range(orbit.period):
        if residue not in admitted:
            residue_constants.append(None)
            continue
        combined = sum(
            (
                constant * root ** (-phase_index * residue)
                for phase_index, constant in enumerate(local_constants)
            ),
            acb(0),
        )
        if not combined.imag.contains(0):
            raise C1FAsymptoticCertificationError(
                "A periodic residue constant could not be proved real"
            )
        _require_positive(
            f"periodic residue-{residue} leading constant",
            combined.real,
        )
        residue_constants.append(
            _decimal_interval(combined.real, precision_bits)
        )
    leading = next(
        value for value in residue_constants if value is not None
    )
    return tuple(residue_constants), leading


def _certify_labeled_puiseux_expansions(
    specification: C1FLabeledSpecification,
    analysis: Any,
    radius: Any,
    trees: Sequence[Any],
    tree_sqrt: Sequence[Any],
    right_directions: Sequence[Sequence[Any]],
    left_directions: Sequence[Sequence[Any]],
    precision_bits: int,
) -> tuple[tuple[C1FCertifiedAsymptoticTerm, ...], ...]:
    """Enclose every displayed Laurent-Puiseux/Fourier coefficient."""

    if not analysis.residue_expansions:
        return tuple()
    critical_count = len(right_directions)
    nominal_power = critical_count / 2.0 - 1.0
    if any(
        residue and residue[0].power < nominal_power - 1e-10
        for residue in analysis.residue_expansions
    ):
        raise C1FAsymptoticCertificationError(
            "A vanished leading Fourier coefficient requires an exact "
            "structural cancellation witness or Constant Oracle"
        )
    half_order = max(
        (
            int(round(2.0 * (nominal_power - term.power)))
            for residue in analysis.residue_expansions
            for term in residue
        ),
        default=0,
    )
    tree_order = max(1, critical_count + half_order)

    def solve_linear(
        matrix: Sequence[Sequence[Any]],
        rhs: Sequence[Any],
    ) -> tuple[Any, ...]:
        size = len(matrix)
        entries = arb_mat(
            size,
            size,
            tuple(matrix[row][column] for row in range(size) for column in range(size)),
        )
        vector = arb_mat(size, 1, tuple(rhs))
        try:
            solution = entries.solve(vector)
        except ZeroDivisionError as exc:
            raise C1FAsymptoticCertificationError(
                "The augmented Puiseux recurrence could not be interval-inverted"
            ) from exc
        return tuple(solution[index, 0] for index in range(size))

    coefficients = compute_tree_puiseux_coefficients(
        specification,
        radius,
        trees,
        tuple(-value for value in tree_sqrt),
        right_directions,
        left_directions,
        tree_order,
        solve_linear,
    )
    from .asymptotics import _exact_periodic_orbit

    orbit = _exact_periodic_orbit(
        specification,
        analysis.critical_component,
        unlabeled=False,
    )
    if analysis.periodicity > 1 and (
        orbit is None or orbit.period != analysis.periodicity
    ):
        raise C1FAsymptoticCertificationError(
            "The numerical Puiseux period has no matching exact support orbit"
        )
    period = analysis.periodicity
    tree_residues = (
        (0,) * specification.tree_dimension
        if orbit is None
        else orbit.tree_residues
    )
    root = (acb(0, 2 * arb.pi() / period)).exp()
    phases = tuple(root**index for index in range(period))
    local = tuple(
        labeled_local_laurent_coefficients(
            specification,
            acb(radius),
            coefficients,
            denominator_valuation=critical_count,
            half_order=half_order,
            phase=phase,
            tree_residues=tree_residues,
        )
        for phase in phases
    )

    result: list[tuple[C1FCertifiedAsymptoticTerm, ...]] = []
    for residue, numerical_terms in enumerate(analysis.residue_expansions):
        certified_terms = []
        for numerical_term in numerical_terms:
            combined = acb(0)
            for phase, coefficients_at_phase in zip(phases, local):
                phase_factor = phase ** (-residue)
                for local_exponent, local_coefficient in coefficients_at_phase.items():
                    beta = Fraction(-local_exponent, 2)
                    if beta.denominator == 1 and beta <= 0:
                        continue
                    base_power = float(beta) - 1.0
                    correction = int(round(base_power - numerical_term.power))
                    if correction < 0 or abs(
                        base_power - correction - numerical_term.power
                    ) > 1e-8:
                        continue
                    ratios = gamma_ratio_coefficients(beta, correction)
                    beta_ball = arb(beta.numerator) / beta.denominator
                    combined += (
                        phase_factor
                        * local_coefficient
                        * arb(ratios[correction].numerator)
                        / ratios[correction].denominator
                        / beta_ball.gamma()
                    )
            if not combined.imag.contains(0):
                raise C1FAsymptoticCertificationError(
                    "A Puiseux/Fourier coefficient could not be proved real"
                )
            midpoint = float(combined.real.mid())
            if abs(midpoint - numerical_term.constant) > 1e-7 * max(
                1.0,
                abs(midpoint),
                abs(numerical_term.constant),
            ):
                raise C1FAsymptoticCertificationError(
                    "A Puiseux/Fourier interval missed its numerical seed: "
                    f"{combined.real} versus {numerical_term.constant}"
                )
            certified_terms.append(
                C1FCertifiedAsymptoticTerm(
                    numerical_term.power,
                    _decimal_interval(combined.real, precision_bits),
                )
            )
        result.append(tuple(certified_terms))
    return tuple(result)


def _replace_expansion_leading_intervals(
    expansions: Sequence[Sequence[C1FCertifiedAsymptoticTerm]],
    leading_intervals: Sequence[C1FRealInterval | None],
) -> tuple[tuple[C1FCertifiedAsymptoticTerm, ...], ...]:
    """Keep the sharper dedicated leading-term enclosure when available."""

    if len(expansions) != len(leading_intervals):
        return tuple(tuple(residue) for residue in expansions)
    result = []
    for terms, leading in zip(expansions, leading_intervals):
        terms = tuple(terms)
        if not terms or leading is None:
            result.append(terms)
            continue
        result.append(
            (C1FCertifiedAsymptoticTerm(terms[0].power, leading),) + terms[1:]
        )
    return tuple(result)


def _certify_unlabeled_puiseux_expansions(
    specification: C1FLabeledSpecification,
    analysis: Any,
    radius: Any,
    level_values: Sequence[Sequence[Any] | None],
    first_coefficients: Sequence[Any],
    right_directions: Sequence[Sequence[Any]],
    left_directions: Sequence[Sequence[Any]],
    precision_bits: int,
    *,
    omitted_tolls: Sequence[
        dict[tuple[int, ...], TruncatedSeries] | None
    ] | None = None,
    omitted_log: TruncatedSeries | None = None,
) -> tuple[tuple[C1FCertifiedAsymptoticTerm, ...], ...]:
    """Enclose displayed Pólya coefficients, including certified tail balls."""

    if not analysis.residue_expansions:
        return tuple()
    critical_count = len(right_directions)
    nominal_power = critical_count / 2.0 - 1.0
    if any(
        residue and residue[0].power < nominal_power - 1e-10
        for residue in analysis.residue_expansions
    ):
        raise C1FAsymptoticCertificationError(
            "A vanished leading Pólya Fourier coefficient requires an exact "
            "structural cancellation witness or Constant Oracle"
        )
    denominator_valuation = critical_count
    half_order = max(
        (
            int(round(2.0 * (nominal_power - term.power)))
            for residue in analysis.residue_expansions
            for term in residue
        ),
        default=0,
    )

    def solve_linear(
        matrix: Sequence[Sequence[Any]],
        rhs: Sequence[Any],
    ) -> tuple[Any, ...]:
        size = len(matrix)
        entries = arb_mat(
            size,
            size,
            tuple(matrix[row][column] for row in range(size) for column in range(size)),
        )
        vector = arb_mat(size, 1, tuple(rhs))
        try:
            solution = entries.solve(vector)
        except ZeroDivisionError as exc:
            raise C1FAsymptoticCertificationError(
                "The Pólya Puiseux recurrence could not be interval-inverted"
            ) from exc
        return tuple(solution[index, 0] for index in range(size))

    hierarchy = compute_unlabeled_tree_hierarchy(
        specification,
        radius,
        level_values,
        first_coefficients,
        right_directions,
        left_directions,
        max(2, denominator_valuation + half_order),
        solve_linear,
        omitted_tolls=omitted_tolls,
    )
    from .asymptotics import _exact_periodic_orbit

    orbit = _exact_periodic_orbit(
        specification,
        analysis.critical_component
        if hasattr(analysis, "critical_component")
        else tuple(index for component in analysis.critical_components for index in component),
        unlabeled=True,
    )
    period = analysis.periodicity
    if period > 1 and (orbit is None or orbit.period != period):
        raise C1FAsymptoticCertificationError(
            "The Pólya expansion has no matching exact periodic orbit"
        )
    tree_residues = (
        (0,) * specification.tree_dimension
        if orbit is None
        else orbit.tree_residues
    )
    root = (acb(0, 2 * arb.pi() / period)).exp()
    phases = tuple(root**index for index in range(period))
    local = tuple(
        unlabeled_local_laurent_coefficients(
            specification,
            acb(radius),
            hierarchy,
            denominator_valuation=denominator_valuation,
            half_order=half_order,
            phase=phase,
            tree_residues=tree_residues,
            omitted_tolls=omitted_tolls,
            omitted_log=omitted_log,
        )
        for phase in phases
    )
    result = []
    for residue, numerical_terms in enumerate(analysis.residue_expansions):
        certified_terms = []
        for numerical_term in numerical_terms:
            combined = acb(0)
            for phase, coefficients_at_phase in zip(phases, local):
                for local_exponent, local_coefficient in coefficients_at_phase.items():
                    beta = Fraction(-local_exponent, 2)
                    if beta.denominator == 1 and beta <= 0:
                        continue
                    base_power = float(beta) - 1.0
                    correction = int(round(base_power - numerical_term.power))
                    if correction < 0 or abs(
                        base_power - correction - numerical_term.power
                    ) > 1e-8:
                        continue
                    ratio = gamma_ratio_coefficients(beta, correction)[correction]
                    beta_ball = arb(beta.numerator) / beta.denominator
                    combined += (
                        phase ** (-residue)
                        * local_coefficient
                        * arb(ratio.numerator)
                        / ratio.denominator
                        / beta_ball.gamma()
                    )
            if not combined.imag.contains(0):
                raise C1FAsymptoticCertificationError(
                    "A Pólya Puiseux/Fourier coefficient could not be proved real"
                )
            midpoint = float(combined.real.mid())
            if (
                not combined.real.contains(numerical_term.constant)
                and abs(midpoint - numerical_term.constant) > 5e-4 * max(
                    1.0,
                    abs(midpoint),
                    abs(numerical_term.constant),
                )
            ):
                raise C1FAsymptoticCertificationError(
                    "A Pólya Puiseux interval missed its numerical seed: "
                    f"{combined.real} versus {numerical_term.constant}"
                )
            certified_terms.append(
                C1FCertifiedAsymptoticTerm(
                    numerical_term.power,
                    _decimal_interval(combined.real, precision_bits),
                )
            )
        result.append(tuple(certified_terms))
    return tuple(result)


def _polya_puiseux_tail_balls(
    specification: C1FLabeledSpecification,
    radius: Any,
    tail_terms: int,
    series_order: int,
) -> tuple[
    tuple[dict[tuple[int, ...], TruncatedSeries] | None, ...],
    TruncatedSeries,
]:
    """Return Cauchy balls for every omitted analytic Pólya coefficient."""

    cauchy_radius = arb(1) / 4
    majorant_radius = arb(radius.upper()) * (1 + cauchy_radius**2)
    if not majorant_radius.upper() < 1:
        raise C1FAsymptoticCertificationError(
            "The Pólya Cauchy disk is not contained in the unit disk"
        )
    multiplicity = sum(specification.multiplicities)
    tolls: list[dict[tuple[int, ...], TruncatedSeries] | None] = [None]
    possible_states = {
        tuple(
            cap if increment else 0
            for cap, increment in zip(
                specification.local_caps,
                specification._local_contribution(profile),
            )
        )
        for profile in specification.transient_profiles
    }
    if not possible_states:
        possible_states = {tuple()}
    for index in range(1, tail_terms + 1):
        first_repeat = tail_terms // index + 1
        first_power = majorant_radius ** (index * first_repeat)
        separation = 1 - 4 * multiplicity * first_power
        _require_positive("Pólya Puiseux tail separation", separation)
        bound = (
            multiplicity
            * first_power
            / (
                first_repeat
                * (1 - majorant_radius**index)
                * separation
            )
        )
        uncertainty = TruncatedSeries(
            tuple(
                arb(0, bound / cauchy_radius**degree)
                for degree in range(series_order + 1)
            )
        )
        tolls.append({state: uncertainty for state in possible_states})

    first_outer = majorant_radius ** (tail_terms + 1)
    outer_separation = 1 - 4 * multiplicity * first_outer
    _require_positive("outer Pólya Puiseux tail separation", outer_separation)
    outer_bound = (
        multiplicity
        * first_outer
        / ((1 - majorant_radius) * outer_separation)
    )
    omitted_log = TruncatedSeries(
        tuple(
            arb(0, outer_bound / cauchy_radius**degree)
            for degree in range(series_order + 1)
        )
    )
    return tuple(tolls), omitted_log


def _complex_error_ball(bound: Any) -> Any:
    radius = arb(bound.upper() if hasattr(bound, "upper") else bound)
    return acb(arb(0, radius), arb(0, radius))


def _periodic_polya_distributions(
    specification: C1FLabeledSpecification,
    radius: Any,
    levels: Sequence[Sequence[Any]],
    tree_sqrt: Sequence[Any],
    orbit: Any,
    phase_index: int,
    level: int,
    total_multiplicity: int,
) -> dict[int, dict[tuple[int, ...], _AcbDirectional]]:
    """Evaluate one Adams level with exact coordinate phases and tail balls."""

    truncation = len(levels) - 1
    root = (acb(0, 2 * arb.pi() / orbit.period)).exp()
    phase = root**phase_index
    repeat_limit = truncation // level
    result = {}
    for root_profile in range(specification.profile_count):
        grouped: dict[tuple[int, ...], _AcbDirectional] = {}
        omitted_states: set[tuple[int, ...]] = set()
        for coordinate, child_profile in enumerate(
            specification.transient_profiles
        ):
            if not specification.successor[child_profile][root_profile]:
                continue
            contribution = _local_contribution(specification, child_profile)
            residue = orbit.tree_residues[coordinate]
            for repeat in range(1, repeat_limit + 1):
                state = tuple(
                    min(cap, repeat * increment)
                    for cap, increment in zip(
                        specification.local_caps,
                        contribution,
                    )
                )
                phase_factor = phase ** (level * repeat * residue)
                derivative = (
                    -phase_factor * tree_sqrt[coordinate]
                    if level == repeat == 1
                    else acb(0)
                )
                value = _AcbDirectional(
                    phase_factor * levels[level * repeat][coordinate] / repeat,
                    derivative / repeat,
                )
                grouped[state] = grouped.get(
                    state,
                    _AcbDirectional.constant(0),
                ) + value
            omitted_states.add(
                tuple(
                    cap if increment else 0
                    for cap, increment in zip(
                        specification.local_caps,
                        contribution,
                    )
                )
            )
        if omitted_states:
            first_repeat = repeat_limit + 1
            first_power = radius ** (level * first_repeat)
            separation = 1 - 4 * total_multiplicity * first_power
            _require_positive("periodic Pólya omitted-tail separation", separation)
            omitted = (
                total_multiplicity
                * first_power
                / (
                    first_repeat
                    * (1 - radius**level)
                    * separation
                )
            )
            for state in omitted_states:
                grouped[state] = grouped.get(
                    state,
                    _AcbDirectional.constant(0),
                ) + _AcbDirectional(_complex_error_ball(omitted), acb(0))
        zero = tuple(0 for _ in specification.local_caps)
        distribution = {zero: _AcbDirectional.constant(1)}
        for state, value in grouped.items():
            distribution = _acb_distribution_multiply(
                distribution,
                _acb_exponential_factor(
                    value,
                    state,
                    specification.local_caps,
                ),
                specification.local_caps,
            )
        result[root_profile] = distribution
    return result


def _periodic_polya_cycle_matrix(
    specification: C1FLabeledSpecification,
    profiles: Sequence[int],
    radius: Any,
    levels: Sequence[Sequence[Any]],
    tree_sqrt: Sequence[Any],
    orbit: Any,
    phase_index: int,
    level: int,
    total_multiplicity: int,
) -> tuple[tuple[_AcbDirectional, ...], ...]:
    distributions = _periodic_polya_distributions(
        specification,
        radius,
        levels,
        tree_sqrt,
        orbit,
        phase_index,
        level,
        total_multiplicity,
    )
    root = (acb(0, 2 * arb.pi() / orbit.period)).exp()
    atom = (root**phase_index * radius) ** level
    return tuple(
        tuple(
            _AcbDirectional.constant(0)
            if not specification.successor[previous][current]
            else atom
            * specification.multiplicities[current]
            * _acb_projected_value(
                specification,
                current,
                distributions[current],
                _local_contribution(specification, previous),
            )
            for current in profiles
        )
        for previous in profiles
    )


def _certify_periodic_unlabeled_residues(
    specification: C1FLabeledSpecification,
    orbit: Any,
    analysis: Any,
    radius: Any,
    levels: Sequence[Sequence[Any]],
    tree_sqrt: Sequence[Any],
    total_multiplicity: int,
    precision_bits: int,
) -> tuple[tuple[C1FRealInterval | None, ...], C1FRealInterval]:
    """Certify retained Pólya factors at every root-of-unity conjugate."""

    truncation = len(levels) - 1
    long_profiles = _long_profiles(specification)
    root = (acb(0, 2 * arb.pi() / orbit.period)).exp()
    local_constants = []
    for phase_index in range(orbit.period):
        matrix_cache: dict[
            tuple[tuple[int, ...], int],
            tuple[tuple[_AcbDirectional, ...], ...],
        ] = {}

        def matrix(
            profiles: tuple[int, ...],
            level: int,
        ) -> tuple[tuple[_AcbDirectional, ...], ...]:
            key = profiles, level
            if key not in matrix_cache:
                matrix_cache[key] = _periodic_polya_cycle_matrix(
                    specification,
                    profiles,
                    radius,
                    levels,
                    tree_sqrt,
                    orbit,
                    phase_index,
                    level,
                    total_multiplicity,
                )
            return matrix_cache[key]

        first_transfer = matrix(long_profiles, 1)
        first_operator = tuple(
            tuple(
                _AcbDirectional.constant(1 if row == column else 0) - entry
                for column, entry in enumerate(matrix_row)
            )
            for row, matrix_row in enumerate(first_transfer)
        )
        _, determinant_coefficient = _acb_determinant_direction(first_operator)
        if determinant_coefficient.contains(0):
            raise C1FAsymptoticCertificationError(
                "A periodic Pólya determinant coefficient contains zero"
            )

        log_factor = acb(0)
        for outer in range(1, truncation + 1):
            for adams in range(1, truncation // outer + 1):
                if outer == 1 and adams == 1:
                    continue
                entries = matrix(long_profiles, outer * adams)
                size = len(entries)
                operator = acb_mat(
                    size,
                    size,
                    tuple(
                        (1 if row == column else 0) - entries[row][column].value
                        for row in range(size)
                        for column in range(size)
                    ),
                )
                determinant = operator.det()
                if determinant.contains(0):
                    raise C1FAsymptoticCertificationError(
                        "A periodic substituted Pólya determinant contains zero"
                    )
                log_factor -= (
                    _integer_totient(adams)
                    * determinant.log()
                    / (outer * adams)
                )

        for outer in range(1, truncation + 1):
            for cycle_length in range(1, specification.depth + 1):
                visible_profiles = tuple(
                    profile
                    for profile, visible_length in enumerate(
                        specification.visible_cycle_lengths
                    )
                    if visible_length == cycle_length
                )
                for adams in range(1, cycle_length + 1):
                    if cycle_length % adams or outer * adams > truncation:
                        continue
                    exponent = cycle_length // adams
                    visible_entries = matrix(
                        visible_profiles,
                        outer * adams,
                    )
                    long_entries = matrix(long_profiles, outer * adams)
                    visible = acb_mat(
                        len(visible_entries),
                        len(visible_entries),
                        tuple(
                            entry.value
                            for row in visible_entries
                            for entry in row
                        ),
                    )
                    long = acb_mat(
                        len(long_entries),
                        len(long_entries),
                        tuple(
                            entry.value
                            for row in long_entries
                            for entry in row
                        ),
                    )
                    log_factor += (
                        _integer_totient(adams)
                        * (
                            (visible**exponent).trace()
                            - (long**exponent).trace()
                        )
                        / (outer * cycle_length)
                    )

        first = radius ** (truncation + 1)
        separation = 1 - 5 * total_multiplicity * first
        _require_positive("periodic Pólya factor-tail separation", separation)
        analytic_tail = (
            total_multiplicity
            * first
            / ((1 - radius) * separation)
        )
        log_factor += _complex_error_ball(analytic_tail)
        local_constants.append(
            log_factor.exp()
            / (determinant_coefficient * arb.pi().sqrt())
        )

    admitted = frozenset(orbit.object_residues)
    residue_constants: list[C1FRealInterval | None] = []
    for residue in range(orbit.period):
        if residue not in admitted:
            residue_constants.append(None)
            continue
        combined = sum(
            (
                constant * root ** (-phase_index * residue)
                for phase_index, constant in enumerate(local_constants)
            ),
            acb(0),
        )
        if not combined.imag.contains(0):
            raise C1FAsymptoticCertificationError(
                "A periodic Pólya residue constant could not be proved real"
            )
        _require_positive(
            f"periodic Pólya residue-{residue} leading constant",
            combined.real,
        )
        residue_constants.append(
            _decimal_interval(combined.real, precision_bits)
        )
    leading = next(value for value in residue_constants if value is not None)
    return tuple(residue_constants), leading


def _verify_tree_independent_cycle_data(
    specification: C1FLabeledSpecification,
) -> None:
    """Prove that every transfer used by the cycle EGF ignores tree values."""

    tree_dimension = specification.tree_dimension
    if tree_dimension == 0:
        return
    x = _Jet.constant(1, tree_dimension)
    trees = tuple(
        _Jet.variable(arb(1), index, tree_dimension)
        for index in range(tree_dimension)
    )
    profile_sets = [_long_profiles(specification)]
    profile_sets.extend(
        tuple(
            profile
            for profile, visible_length in enumerate(
                specification.visible_cycle_lengths
            )
            if visible_length == cycle_length
        )
        for cycle_length in range(1, specification.depth + 1)
    )
    for profiles in profile_sets:
        transfer = _cycle_matrix(specification, profiles, x, trees)
        if any(
            not derivative.is_zero()
            for row in transfer
            for entry in row
            for derivative in entry.gradient
        ):
            raise C1FAsymptoticCertificationError(
                "Arb could not prove that the cycle-pole transfer and finite "
                "correction are independent of transient trees"
            )


def _tree_hessian_direction(
    specification: C1FLabeledSpecification,
    x: Any,
    trees: Sequence[Any],
    direction: Sequence[Any],
) -> tuple[Any, ...]:
    constant_trees = tuple(_Jet.constant(value, 0) for value in trees)
    result = []
    for root_profile in specification.transient_profiles:
        distribution = _child_distribution(
            specification,
            root_profile,
            constant_trees,
            0,
        )
        pointed: dict[tuple[int, ...], Any] = {}
        for child_index, child_profile in enumerate(specification.transient_profiles):
            if not specification.successor[child_profile][root_profile]:
                continue
            contribution = _local_contribution(specification, child_profile)
            pointed[contribution] = pointed.get(contribution, arb(0)) + direction[
                child_index
            ]
        double_pointed: dict[tuple[int, ...], Any] = {}
        for left_state, left_value in pointed.items():
            for right_state, right_value in pointed.items():
                state = _capped_add(
                    left_state,
                    right_state,
                    specification.local_caps,
                )
                double_pointed[state] = (
                    double_pointed.get(state, arb(0))
                    + left_value * right_value
                )
        value = arb(0)
        for contribution, coefficient in double_pointed.items():
            value += coefficient * _projected_value(
                specification,
                root_profile,
                distribution,
                0,
                contribution,
            ).value
        result.append(x * specification.multiplicities[root_profile] * value)
    return tuple(result)


def _strictly_positive(value: Any) -> bool:
    return bool(value.lower() > 0)


def _require_positive(name: str, value: Any) -> None:
    if not _strictly_positive(value):
        raise C1FAsymptoticCertificationError(
            f"The interval for {name} does not prove strict positivity: {value}"
        )


def _certify_nonnegative_matrix_subcritical(matrix: Any, name: str) -> None:
    """Prove ``rho(matrix) < 1`` using a positive Collatz supersolution."""

    size = matrix.nrows()
    if size == 0:
        return
    operator = _arb_identity(size) - matrix
    ones = arb_mat(size, 1, tuple(arb(1) for _ in range(size)))
    try:
        weights = operator.mid().solve(ones)
    except ZeroDivisionError as exc:
        raise C1FAsymptoticCertificationError(
            f"The midpoint operator for {name} is singular"
        ) from exc
    slack = weights - matrix * weights
    for index in range(size):
        _require_positive(f"{name} Collatz weight {index}", weights[index, 0])
        _require_positive(f"{name} Collatz slack {index}", slack[index, 0])


def _exact_tree_aperiodicity_witness(
    specification: C1FLabeledSpecification,
    critical_component: Sequence[int],
    *,
    degree_limit: int = 16,
    unlabeled: bool,
) -> tuple[int, tuple[int, ...]] | None:
    """Find an exact span-one coefficient witness in a critical tree series."""

    from .c1f import (
        _atom_times_series,
        _series_add_in_place,
        _series_append_state,
        _series_filter_local_state,
        _series_exp,
        _series_multiset,
        _zero_series,
    )

    multiplicities = []
    for value in specification.multiplicities:
        integer = int(round(value))
        if abs(value - integer) > 1e-12:
            return None
        multiplicities.append(integer)
    transient = specification.transient_profiles
    collection = _series_multiset if unlabeled else _series_exp
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
                multiplicities[root_profile],
                valid,
                tuple(),
                degree_limit,
            )
        trees = next_trees

    for coordinate in critical_component:
        profile = transient[coordinate]
        support = tuple(
            degree
            for degree, coefficient in enumerate(trees[profile])
            if coefficient
        )
        if len(support) < 2:
            continue
        span = 0
        anchor = support[0]
        for degree in support[1:]:
            span = gcd(span, degree - anchor)
        if span == 1:
            return profile, support
    return None


def _exact_unlabeled_tree_aperiodicity_witness(
    specification: C1FLabeledSpecification,
    critical_component: Sequence[int],
) -> tuple[int, tuple[int, ...]] | None:
    return _exact_tree_aperiodicity_witness(
        specification,
        critical_component,
        unlabeled=True,
    )


def _decimal_interval(value: Any, precision_bits: int) -> C1FRealInterval:
    digits = max(18, int(precision_bits * 0.30103) - 8)
    midpoint_text = value.mid().str(20, radius=False)
    midpoint = Decimal(midpoint_text)
    magnitude = midpoint.adjusted() if midpoint else 0
    display_guard = f"1e{magnitude - digits + 2:+d}"
    outward = arb(value, display_guard)
    return C1FRealInterval(
        lower=outward.lower().str(digits, radius=False),
        upper=outward.upper().str(digits, radius=False),
        enclosure=outward.str(digits),
    )


def _regular_certificate(
    specification: C1FLabeledSpecification,
    analysis: Any,
    precision_bits: int,
) -> C1FAsymptoticCertificate:
    if any(
        profile not in _long_profiles(specification)
        for profile in specification.transient_profiles
    ):
        raise C1FAsymptoticCertificationError(
            "Regular certification requires every productive tree type in the "
            "long-cycle transfer"
        )
    from .asymptotics import (
        _kernel_vector,
        _principal_submatrix,
        _strongly_connected_components,
        _transpose,
    )

    critical_component = (
        analysis.critical_component
        if analysis.critical_component
        else tuple(range(specification.tree_dimension))
    )
    aperiodicity_witness = _exact_tree_aperiodicity_witness(
        specification,
        critical_component,
        unlabeled=False,
    )
    from .asymptotics import _exact_periodic_orbit

    periodic_orbit = _exact_periodic_orbit(
        specification,
        critical_component,
        unlabeled=False,
    )
    periodic_projection = bool(
        periodic_orbit is not None
        and analysis.periodicity == periodic_orbit.period
    )
    numerical_jacobian = specification.tree_jacobian(
        analysis.radius,
        analysis.tree_values,
    )
    critical_jacobian = _principal_submatrix(
        numerical_jacobian,
        critical_component,
    )
    critical_right = _kernel_vector(critical_jacobian)
    raw_critical_left = _kernel_vector(_transpose(critical_jacobian))
    critical_normalization = sum(
        a * b for a, b in zip(raw_critical_left, critical_right)
    )
    critical_left = tuple(
        value / critical_normalization for value in raw_critical_left
    )
    centers = (
        log(analysis.radius),
        *(log(value) for value in analysis.tree_values),
        *(log(value) for value in critical_right),
        *(log(value) for value in critical_left),
    )
    root_box = _krawczyk_box(
        lambda values: _regular_system(
            specification,
            values,
            critical_component,
        ),
        centers,
    )
    tree_dimension = specification.tree_dimension
    critical_dimension = len(critical_component)
    radius = root_box[0].exp()
    trees = tuple(root_box[1 + index].exp() for index in range(tree_dimension))
    right_offset = 1 + tree_dimension
    certified_critical_right = tuple(
        root_box[right_offset + index].exp()
        for index in range(critical_dimension)
    )
    left_offset = right_offset + critical_dimension
    certified_critical_left = tuple(
        root_box[left_offset + index].exp()
        for index in range(critical_dimension)
    )
    jacobian = _full_tree_jacobian(specification, radius, trees)
    right, left = _normalized_full_kernel_vectors(
        jacobian,
        critical_component,
    )
    for index, value in enumerate(certified_critical_right):
        _require_positive(f"critical right Perron coordinate {index}", value)
    for index, value in enumerate(certified_critical_left):
        _require_positive(f"critical left Perron coordinate {index}", value)

    dominance_checks_succeeded = (
        aperiodicity_witness is not None or periodic_projection
    )
    if dominance_checks_succeeded:
        try:
            components = _strongly_connected_components(
                specification.dependency_matrix()
            )
            if tuple(critical_component) not in components:
                raise C1FAsymptoticCertificationError(
                    "The selected labeled critical component is not an exact SCC"
                )
            for component in components:
                if component == tuple(critical_component):
                    continue
                block = arb_mat(
                    len(component),
                    len(component),
                    tuple(
                        jacobian[row, column]
                        for row in component
                        for column in component
                    ),
                )
                _certify_nonnegative_matrix_subcritical(
                    block,
                    "noncritical labeled tree block",
                )
        except C1FAsymptoticCertificationError:
            dominance_checks_succeeded = False

    constant_x = _Jet.constant(radius, 0)
    constant_trees = tuple(_Jet.constant(value, 0) for value in trees)
    phi = tuple(
        value.value
        for value in _tree_phi(specification, constant_x, constant_trees)
    )
    x_derivative = tuple(value / radius for value in phi)
    hessian = _tree_hessian_direction(
        specification,
        radius,
        trees,
        right,
    )
    numerator = 2 * radius * sum(
        (a * b for a, b in zip(left, x_derivative)),
        arb(0),
    )
    denominator = sum(
        (a * b for a, b in zip(left, hessian)),
        arb(0),
    )
    _require_positive("square-root numerator", numerator)
    _require_positive("square-root denominator", denominator)
    kappa = (numerator / denominator).sqrt()
    tree_sqrt = tuple(kappa * value for value in right)

    directional_x = _Jet.constant(radius, 1)
    directional_trees = tuple(
        _Jet(value, (-coefficient,))
        for value, coefficient in zip(trees, tree_sqrt)
    )
    transfer = _cycle_matrix(
        specification,
        _long_profiles(specification),
        directional_x,
        directional_trees,
    )
    delta = _determinant_jet(_identity_minus(transfer)).gradient[0]
    _require_positive("determinant square-root coefficient", delta)
    correction = _finite_cycle_correction(specification, radius, trees)
    local_leading_constant = correction.exp() / (delta * arb.pi().sqrt())
    _require_positive("leading constant", local_leading_constant)
    convert = lambda value: _decimal_interval(value, precision_bits)
    residue_constants: tuple[C1FRealInterval | None, ...] = tuple()
    certified_leading_constant = convert(local_leading_constant)
    if periodic_projection and periodic_orbit is not None:
        residue_constants, certified_leading_constant = (
            _certify_periodic_labeled_residues(
                specification,
                periodic_orbit,
                analysis,
                radius,
                trees,
                tree_sqrt,
                precision_bits,
            )
        )
    certified_expansions = _certify_labeled_puiseux_expansions(
        specification,
        analysis,
        radius,
        trees,
        tree_sqrt,
        (right,),
        (left,),
        precision_bits,
    )
    if certified_expansions:
        residue_constants = tuple(
            residue[0].constant if residue else None
            for residue in certified_expansions
        )
        certified_leading_constant = next(
            residue[0].constant for residue in certified_expansions if residue
        )
    return C1FAsymptoticCertificate(
        precision_bits=precision_bits,
        method="arb-krawczyk",
        regime=analysis.regime,
        radius=convert(radius),
        exponential_growth=convert(1 / radius),
        leading_constant=certified_leading_constant,
        finite_cycle_correction=convert(correction),
        denominator_coefficient=convert(delta),
        tree_values=tuple(convert(value) for value in trees),
        tree_sqrt_coefficients=tuple(convert(value) for value in tree_sqrt),
        perron_right=tuple(convert(value) for value in right),
        perron_left=tuple(convert(value) for value in left),
        verified_properties=(
            "unique critical point in the displayed Krawczyk box",
            "positive normalized Perron vectors on the critical component",
            "unique full non-negative left and right critical directions",
            "simple critical Perron root",
            "positive quadratic square-root scale",
            "positive determinant and leading coefficients",
        ) + (
            (
                "exact span-one critical-tree coefficient witness",
                "monotone majorant excludes interior singularities",
                "phase rigidity excludes nonpositive boundary singularities",
                "all noncritical labeled blocks have spectral radius below one",
            )
            if dominance_checks_succeeded and aperiodicity_witness is not None
            else (
                "exact periodic critical-tree support congruences",
                "complete root-of-unity dominant orbit",
                "exact residue-class support projection",
                "monotone majorant excludes interior singularities",
                "phase rigidity excludes unlisted boundary singularities",
                "all noncritical labeled blocks have spectral radius below one",
            )
            if dominance_checks_succeeded and periodic_projection
            else tuple()
        ) + (
            (
                "recursive interval Laurent-Puiseux expansion",
                "interval coefficient transfer and Fourier projection",
            )
            if certified_expansions
            else tuple()
        ),
        dominance_certified=dominance_checks_succeeded,
        residue_powers=(
            analysis.residue_powers if certified_expansions or periodic_projection else tuple()
        ),
        residue_leading_constants=(
            residue_constants
            if certified_expansions or periodic_projection
            else tuple()
        ),
        residue_expansions=certified_expansions,
    )


def _repeated_critical_block_specification(
    specification: C1FLabeledSpecification,
    components: Sequence[Sequence[int]],
) -> C1FLabeledSpecification:
    """Extract one block after proving all critical blocks are exact copies."""

    covered = sorted(index for component in components for index in component)
    if covered != list(range(specification.tree_dimension)):
        raise C1FAsymptoticCertificationError(
            "Multiple critical SCC certification currently requires the critical "
            "blocks to cover all transient coordinates"
        )
    first = tuple(components[0])
    if any(len(component) != len(first) for component in components):
        raise C1FAsymptoticCertificationError(
            "The independent critical blocks are not exact copies"
        )
    transient = specification.transient_profiles

    def signature(component: Sequence[int]) -> tuple[Any, ...]:
        profiles = tuple(transient[index] for index in component)
        profile_set = frozenset(profiles)
        if any(
            specification.successor[source][target]
            for source in profiles
            for target in range(specification.profile_count)
            if target not in profile_set
        ) or any(
            specification.successor[source][target]
            for source in range(specification.profile_count)
            if source not in profile_set
            for target in profiles
        ):
            raise C1FAsymptoticCertificationError(
                "A repeated critical block is coupled to another profile"
            )
        return (
            tuple(specification.multiplicities[profile] for profile in profiles),
            tuple(specification.visible_cycle_lengths[profile] for profile in profiles),
            tuple(
                tuple(specification.successor[source][target] for target in profiles)
                for source in profiles
            ),
            tuple(_local_contribution(specification, profile) for profile in profiles),
            tuple(_accepted_states(specification, profile) for profile in profiles),
        )

    first_signature = signature(first)
    if any(signature(component) != first_signature for component in components[1:]):
        raise C1FAsymptoticCertificationError(
            "The independent critical blocks are not exact copies"
        )
    profiles = tuple(transient[index] for index in first)
    return C1FLabeledSpecification(
        depth=specification.depth,
        successor=tuple(
            tuple(specification.successor[source][target] for target in profiles)
            for source in profiles
        ),
        multiplicities=tuple(
            specification.multiplicities[profile] for profile in profiles
        ),
        visible_cycle_lengths=tuple(
            specification.visible_cycle_lengths[profile] for profile in profiles
        ),
        transient_profiles=tuple(range(len(profiles))),
        eventual_start=specification.eventual_start,
        long_cycle_profiles=tuple(range(len(profiles))),
        local_caps=specification.local_caps,
        local_contributions=tuple(
            _local_contribution(specification, profile) for profile in profiles
        ) if specification.local_caps else tuple(),
        local_acceptance=tuple(
            _accepted_states(specification, profile) for profile in profiles
        ) if specification.local_caps else tuple(),
    )


def _multiple_regular_certificate(
    specification: C1FLabeledSpecification,
    analysis: Any,
    precision_bits: int,
) -> C1FAsymptoticCertificate:
    from .asymptotics import analyze_c1f_labeled_asymptotics

    components = analysis.critical_components
    block = _repeated_critical_block_specification(specification, components)
    block_analysis = analyze_c1f_labeled_asymptotics(block)
    block_certificate = _regular_certificate(
        block,
        block_analysis,
        precision_bits,
    )
    copies = len(components)

    def interval(value: C1FRealInterval) -> Any:
        return arb(value.enclosure)

    radius = interval(block_certificate.radius)
    block_trees = tuple(interval(value) for value in block_certificate.tree_values)
    block_sqrt = tuple(
        interval(value) for value in block_certificate.tree_sqrt_coefficients
    )
    block_right = tuple(interval(value) for value in block_certificate.perron_right)
    block_left = tuple(interval(value) for value in block_certificate.perron_left)
    trees_by_index: list[Any] = [None] * specification.tree_dimension
    sqrt_by_index: list[Any] = [None] * specification.tree_dimension
    right_by_index: list[Any] = [None] * specification.tree_dimension
    left_by_index: list[Any] = [None] * specification.tree_dimension
    for component in components:
        for local_index, global_index in enumerate(component):
            trees_by_index[global_index] = block_trees[local_index]
            sqrt_by_index[global_index] = block_sqrt[local_index]
            right_by_index[global_index] = block_right[local_index]
            left_by_index[global_index] = block_left[local_index]
    trees = tuple(trees_by_index)
    tree_sqrt = tuple(sqrt_by_index)
    right = tuple(right_by_index)
    left = tuple(left_by_index)
    zero = arb(0)
    right_directions = tuple(
        tuple(right[index] if index in component else zero for index in range(len(right)))
        for component in components
    )
    left_directions = tuple(
        tuple(left[index] if index in component else zero for index in range(len(left)))
        for component in components
    )
    correction = copies * interval(block_certificate.finite_cycle_correction)
    denominator = interval(block_certificate.denominator_coefficient) ** copies
    singular_exponent = arb(copies) / 2
    leading_constant = correction.exp() / (
        denominator * singular_exponent.gamma()
    )
    _require_positive("multiple-critical denominator coefficient", denominator)
    _require_positive("multiple-critical leading constant", leading_constant)

    certified_expansions = _certify_labeled_puiseux_expansions(
        specification,
        analysis,
        radius,
        trees,
        tree_sqrt,
        right_directions,
        left_directions,
        precision_bits,
    )
    residue_constants = tuple(
        residue[0].constant if residue else None
        for residue in certified_expansions
    )
    if certified_expansions:
        leading_constant_interval = next(
            residue[0].constant for residue in certified_expansions if residue
        )
    else:
        leading_constant_interval = None

    convert = lambda value: _decimal_interval(value, precision_bits)
    return C1FAsymptoticCertificate(
        precision_bits=precision_bits,
        method="arb-krawczyk-repeated-block",
        regime=analysis.regime,
        radius=convert(radius),
        exponential_growth=convert(1 / radius),
        leading_constant=(
            leading_constant_interval
            if leading_constant_interval is not None
            else convert(leading_constant)
        ),
        finite_cycle_correction=convert(correction),
        denominator_coefficient=convert(denominator),
        tree_values=tuple(convert(value) for value in trees),
        tree_sqrt_coefficients=tuple(convert(value) for value in tree_sqrt),
        perron_right=tuple(convert(value) for value in right),
        perron_left=tuple(convert(value) for value in left),
        verified_properties=(
            "exact decomposition into uncoupled isomorphic critical blocks",
            "unique critical point for the representative block",
            "positive square-root scale in every critical block",
            "positive product denominator and leading coefficients",
        ) + (
            (
                "exact periodic support for every repeated critical block",
                "complete interacting root-of-unity orbit",
                "recursive interval Laurent-Puiseux expansion",
                "interval coefficient transfer and Fourier projection",
            )
            if analysis.periodicity > 1
            else (
                "recursive interval Laurent-Puiseux expansion",
            )
            if certified_expansions
            else tuple()
        ),
        dominance_certified=block_certificate.dominance_certified,
        residue_powers=analysis.residue_powers if certified_expansions else tuple(),
        residue_leading_constants=residue_constants,
        residue_expansions=certified_expansions,
    )


def _analytic_tree_cycle_pole_certificate(
    specification: C1FLabeledSpecification,
    analysis: Any,
    precision_bits: int,
) -> C1FAsymptoticCertificate:
    """Certify a simple cycle pole above an analytic tree subsystem."""

    critical_component = tuple(analysis.critical_component)
    if analysis.cycle_pole_order != 1 or not critical_component:
        raise C1FAsymptoticCertificationError(
            "The analytic-tree interval path currently requires one simple cycle pole"
        )
    centers = (
        log(analysis.radius),
        *(log(value) for value in analysis.tree_values),
        *(
            log(analysis.perron_right[index])
            for index in critical_component
        ),
        *(
            log(analysis.perron_left[index])
            for index in critical_component
        ),
    )
    root_box = _krawczyk_box(
        lambda values: _analytic_tree_cycle_pole_system(
            specification,
            critical_component,
            values,
        ),
        centers,
    )
    tree_dimension = specification.tree_dimension
    critical_dimension = len(critical_component)
    radius = root_box[0].exp()
    trees = tuple(
        root_box[1 + index].exp()
        for index in range(tree_dimension)
    )
    right_offset = 1 + tree_dimension
    certified_right = tuple(
        root_box[right_offset + index].exp()
        for index in range(critical_dimension)
    )
    left_offset = right_offset + critical_dimension
    certified_left = tuple(
        root_box[left_offset + index].exp()
        for index in range(critical_dimension)
    )
    for index, value in enumerate(certified_right):
        _require_positive(f"cycle right Perron coordinate {index}", value)
    for index, value in enumerate(certified_left):
        _require_positive(f"cycle left Perron coordinate {index}", value)

    tree_variables = tuple(
        _Jet.variable(value, index, tree_dimension)
        for index, value in enumerate(trees)
    )
    tree_phi = _tree_phi(
        specification,
        _Jet.constant(radius, tree_dimension),
        tree_variables,
    )
    tree_jacobian = arb_mat(
        tree_dimension,
        tree_dimension,
        tuple(
            tree_phi[row].gradient[column]
            for row in range(tree_dimension)
            for column in range(tree_dimension)
        ),
    )
    tree_operator = _arb_identity(tree_dimension) - tree_jacobian
    ones = arb_mat(tree_dimension, 1, tuple(arb(1) for _ in trees))
    try:
        weights = tree_operator.mid().solve(ones)
    except ZeroDivisionError as exc:
        raise C1FAsymptoticCertificationError(
            "The analytic tree Jacobian could not be separated from one"
        ) from exc
    for index in range(tree_dimension):
        _require_positive(f"tree subcritical weight {index}", weights[index, 0])
    slack = weights - tree_jacobian * weights
    for index in range(tree_dimension):
        _require_positive(f"tree subcritical Collatz slack {index}", slack[index, 0])

    try:
        tree_direction = tree_operator.solve(
            arb_mat(tree_dimension, 1, tuple(trees))
        )
    except ZeroDivisionError as exc:
        raise C1FAsymptoticCertificationError(
            "The analytic tree derivative could not be enclosed"
        ) from exc
    directional_x = _Jet(radius, (-radius,))
    directional_trees = tuple(
        _Jet(trees[index], (-tree_direction[index, 0],))
        for index in range(tree_dimension)
    )
    transfer = _cycle_matrix(
        specification,
        _long_profiles(specification),
        directional_x,
        directional_trees,
    )
    eta = _determinant_jet(_identity_minus(transfer)).gradient[0]
    _require_positive("analytic-tree cycle-pole coefficient", eta)
    correction = _finite_cycle_correction(specification, radius, trees)
    leading_constant = correction.exp() / eta
    _require_positive("analytic-tree cycle-pole leading constant", leading_constant)

    full_right = [arb(0) for _ in range(len(_long_profiles(specification)))]
    full_left = [arb(0) for _ in range(len(_long_profiles(specification)))]
    for local, global_index in enumerate(critical_component):
        full_right[global_index] = certified_right[local]
        full_left[global_index] = certified_left[local]
    convert = lambda value: _decimal_interval(value, precision_bits)
    return C1FAsymptoticCertificate(
        precision_bits=precision_bits,
        method="arb-krawczyk-analytic-tree-cycle-pole",
        regime=analysis.regime,
        radius=convert(radius),
        exponential_growth=convert(1 / radius),
        leading_constant=convert(leading_constant),
        finite_cycle_correction=convert(correction),
        denominator_coefficient=convert(eta),
        tree_values=tuple(convert(value) for value in trees),
        tree_sqrt_coefficients=tuple(convert(arb(0)) for _ in trees),
        perron_right=tuple(convert(value) for value in full_right),
        perron_left=tuple(convert(value) for value in full_left),
        verified_properties=(
            "unique joint analytic-tree and cycle-Perron point in the Krawczyk box",
            "positive normalized cycle Perron vectors",
            "strictly subcritical tree Jacobian by a Collatz supersolution",
            "simple positive cycle-pole coefficient",
            "positive local leading constant",
        ),
        dominance_certified=False,
    )


def _cycle_pole_certificate(
    specification: C1FLabeledSpecification,
    analysis: Any,
    precision_bits: int,
) -> C1FAsymptoticCertificate:
    _verify_tree_independent_cycle_data(specification)
    from .asymptotics import (
        _dot,
        _kernel_vector,
        _polynomial_derivative,
        _transpose,
        _tree_independent_cycle_polynomials,
    )

    denominator, correction_polynomial = _tree_independent_cycle_polynomials(
        specification
    )
    roots = fmpz_poly(list(denominator)).complex_roots()
    if not roots:
        raise C1FAsymptoticCertificationError(
            "The exact cycle denominator has no roots"
        )
    positive_roots = tuple(
        (root, multiplicity)
        for root, multiplicity in roots
        if root.imag.contains(0) and root.real.lower() > 0
    )
    if not positive_roots:
        raise C1FAsymptoticCertificationError(
            "The exact cycle denominator has no isolated positive root"
        )
    positive_root, positive_order = min(
        positive_roots,
        key=lambda item: abs(float(item[0].real.mid()) - analysis.radius),
    )
    radius = positive_root.real
    dominant_roots = []
    for root, multiplicity in roots:
        modulus = abs(root)
        if modulus.overlaps(radius):
            dominant_roots.append((root, multiplicity))
        elif not modulus.lower() > radius.upper():
            raise C1FAsymptoticCertificationError(
                "The isolated denominator roots do not yet separate the dominant circle"
            )
    if positive_order != analysis.cycle_pole_order:
        raise C1FAsymptoticCertificationError(
            "The exact positive pole multiplicity disagrees with the numerical analysis"
        )

    def polynomial_value(coefficients: Sequence[Any], value: Any) -> Any:
        result = acb(0)
        for coefficient in reversed(coefficients):
            if hasattr(coefficient, "numerator"):
                converted = arb(coefficient.numerator) / coefficient.denominator
            else:
                converted = arb(coefficient)
            result = result * value + converted
        return result

    def local_pole_coefficient(root: Any, order: int) -> Any:
        derivative = polynomial_value(
            _polynomial_derivative(denominator, order),
            root,
        )
        eta_at_root = derivative * ((-root) ** order) / factorial(order)
        if eta_at_root.contains(0):
            raise C1FAsymptoticCertificationError(
                "The leading denominator coefficient at a dominant pole contains zero"
            )
        return polynomial_value(correction_polynomial, root).exp() / eta_at_root

    def local_pole_polynomial(root: Any, order: int) -> tuple[Any, ...]:
        required = order - 1

        def compose(coefficients: Sequence[Any], limit: int) -> list[Any]:
            result = [acb(0) for _ in range(limit + 1)]
            for degree, coefficient in enumerate(coefficients):
                converted = (
                    arb(coefficient.numerator) / coefficient.denominator
                    if hasattr(coefficient, "numerator")
                    else arb(coefficient)
                )
                binomial = 1
                for local_degree in range(min(degree, limit) + 1):
                    if local_degree:
                        binomial = (
                            binomial
                            * (degree - local_degree + 1)
                            // local_degree
                        )
                    result[local_degree] += (
                        converted
                        * root**degree
                        * ((-1) ** local_degree)
                        * binomial
                    )
            return result

        denominator_u = compose(denominator, 2 * order - 1)
        correction_u = compose(correction_polynomial, required)
        numerator_u = [acb(0) for _ in range(required + 1)]
        numerator_u[0] = correction_u[0].exp()
        for degree in range(1, required + 1):
            numerator_u[degree] = sum(
                (
                    local_degree
                    * correction_u[local_degree]
                    * numerator_u[degree - local_degree]
                    for local_degree in range(1, degree + 1)
                ),
                acb(0),
            ) / degree
        leading_denominator = denominator_u[order]
        if leading_denominator.contains(0):
            raise C1FAsymptoticCertificationError(
                "A local denominator leading coefficient contains zero"
            )
        quotient = [acb(0) for _ in range(required + 1)]
        for degree in range(required + 1):
            quotient[degree] = (
                numerator_u[degree]
                - sum(
                    (
                        denominator_u[order + offset]
                        * quotient[degree - offset]
                        for offset in range(1, degree + 1)
                    ),
                    acb(0),
                )
            ) / leading_denominator
        polynomial = [acb(0) for _ in range(order)]
        for pole_order in range(1, order + 1):
            binomial = [acb(1)]
            for offset in range(1, pole_order):
                updated = [acb(0) for _ in range(len(binomial) + 1)]
                for degree, value in enumerate(binomial):
                    updated[degree] += value
                    updated[degree + 1] += value / offset
                binomial = updated
            laurent = quotient[order - pole_order]
            for degree, value in enumerate(binomial):
                polynomial[degree] += laurent * value
        return tuple(polynomial)

    certified_local_polynomials = tuple(
        (root, local_pole_polynomial(root, multiplicity))
        for root, multiplicity in dominant_roots
    )
    residue_polynomials: list[tuple[Any, ...]] = []
    for residue, numerical_polynomial in enumerate(
        analysis.residue_polynomial_coefficients
    ):
        if not numerical_polynomial:
            residue_polynomials.append(tuple())
            continue
        combined = [acb(0) for _ in range(len(numerical_polynomial))]
        for root, polynomial in certified_local_polynomials:
            phase = (acb(radius) / root) ** residue
            for degree in range(min(len(combined), len(polynomial))):
                combined[degree] += phase * polynomial[degree]
        real_coefficients = []
        for coefficient in combined:
            if not coefficient.imag.contains(0):
                raise C1FAsymptoticCertificationError(
                    "A residue polynomial coefficient could not be proved real"
                )
            real_coefficients.append(coefficient.real)
        _require_positive(
            f"residue-{residue} polynomial leading coefficient",
            real_coefficients[-1],
        )
        residue_polynomials.append(tuple(real_coefficients))

    residue_constants: list[Any | None] = []
    for residue, power in enumerate(analysis.residue_powers):
        if power is None:
            residue_constants.append(None)
            continue
        order = int(power) + 1
        coefficient = acb(0)
        for root, multiplicity in dominant_roots:
            if multiplicity != order:
                continue
            coefficient += (
                local_pole_coefficient(root, order)
                * (acb(radius) / root) ** residue
                / factorial(order - 1)
            )
        if not coefficient.imag.contains(0):
            raise C1FAsymptoticCertificationError(
                "A residue-class pole sum could not be proved real"
            )
        _require_positive(
            f"residue-{residue} leading constant",
            coefficient.real,
        )
        residue_constants.append(coefficient.real)

    positive_derivative = polynomial_value(
        _polynomial_derivative(denominator, positive_order),
        positive_root,
    )
    eta = (
        positive_derivative
        * ((-positive_root) ** positive_order)
        / factorial(positive_order)
    ).real
    _require_positive("cycle-pole denominator coefficient", eta)
    correction = polynomial_value(correction_polynomial, positive_root).real
    class_zero = residue_constants[0]
    if class_zero is None:
        raise C1FAsymptoticCertificationError(
            "The zero residue class unexpectedly has no asymptotic contribution"
        )

    right: tuple[Any, ...] = tuple()
    left: tuple[Any, ...] = tuple()
    critical_components = analysis.critical_components
    if len(critical_components) == 1:
        long_profiles = _long_profiles(specification)
        component_profiles = tuple(
            long_profiles[index]
            for index in critical_components[0]
        )
        zero_trees = tuple(0.0 for _ in range(specification.tree_dimension))
        transfer = specification.cycle_matrix(
            component_profiles,
            analysis.radius,
            zero_trees,
        )
        numerical_right = _kernel_vector(transfer)
        raw_left = _kernel_vector(_transpose(transfer))
        normalization = _dot(raw_left, numerical_right)
        numerical_left = tuple(value / normalization for value in raw_left)
        centers = (
            log(analysis.radius),
            *(log(value) for value in numerical_right),
            *(log(value) for value in numerical_left),
        )
        root_box = _krawczyk_box(
            lambda values: _cycle_pole_system(
                specification,
                values,
                component_profiles,
            ),
            centers,
        )
        component_dimension = len(component_profiles)
        certified_radius = root_box[0].exp()
        if not certified_radius.overlaps(radius):
            raise C1FAsymptoticCertificationError(
                "Perron and exact-polynomial pole enclosures are disjoint"
            )
        right = tuple(
            root_box[1 + index].exp()
            for index in range(component_dimension)
        )
        left_offset = 1 + component_dimension
        left = tuple(
            root_box[left_offset + index].exp()
            for index in range(component_dimension)
        )

    convert = lambda value: _decimal_interval(value, precision_bits)
    return C1FAsymptoticCertificate(
        precision_bits=precision_bits,
        method="arb-krawczyk",
        regime=analysis.regime,
        radius=convert(radius),
        exponential_growth=convert(1 / radius),
        leading_constant=convert(class_zero),
        finite_cycle_correction=convert(correction),
        denominator_coefficient=convert(eta),
        tree_values=tuple(),
        tree_sqrt_coefficients=tuple(),
        perron_right=tuple(convert(value) for value in right),
        perron_left=tuple(convert(value) for value in left),
        verified_properties=(
            "all denominator roots isolated from the exact integer polynomial",
            "dominant root circle separated from every subdominant root",
            "exact graph periods and exact positive-pole multiplicity",
            "positive pole and nonzero residue-class leading coefficients",
        ),
        dominance_certified=True,
        residue_powers=analysis.residue_powers,
        residue_leading_constants=tuple(
            None if value is None else convert(value)
            for value in residue_constants
        ),
        residue_polynomial_coefficients=tuple(
            tuple(convert(value) for value in polynomial)
            for polynomial in residue_polynomials
        ),
        residue_expansions=tuple(
            tuple(
                C1FCertifiedAsymptoticTerm(float(degree), convert(value))
                for degree, value in reversed(tuple(enumerate(polynomial)))
            )
            for polynomial in residue_polynomials
        ),
    )


def certify_c1f_labeled_asymptotics(
    specification: C1FLabeledSpecification,
    analysis: Any,
    *,
    precision_bits: int = 192,
    max_precision_bits: int | None = None,
) -> C1FAsymptoticCertificate:
    """Certify one supported floating-point asymptotic result with Arb."""

    _require_arb()
    if precision_bits < 64:
        raise ValueError("Interval certification requires at least 64 bits")
    if max_precision_bits is None:
        max_precision_bits = precision_bits
    if max_precision_bits < precision_bits:
        raise ValueError("max_precision_bits must be at least precision_bits")

    attempted: list[int] = []
    current_precision = precision_bits
    last_error: C1FAsymptoticCertificationError | None = None
    while current_precision <= max_precision_bits:
        attempted.append(current_precision)
        try:
            with ctx.workprec(current_precision):
                if analysis.regime == "analytic-tree-cycle-pole":
                    return _analytic_tree_cycle_pole_certificate(
                        specification,
                        analysis,
                        current_precision,
                    )
                if analysis.regime == "cycle-pole":
                    return _cycle_pole_certificate(
                        specification,
                        analysis,
                        current_precision,
                    )
                if len(analysis.critical_components) > 1 and analysis.tree_values:
                    return _multiple_regular_certificate(
                        specification,
                        analysis,
                        current_precision,
                    )
                if analysis.tree_values:
                    return _regular_certificate(
                        specification,
                        analysis,
                        current_precision,
                    )
                raise C1FAsymptoticCertificationError(
                    "Interval certification does not support regime "
                    f"{analysis.regime!r}"
                )
        except C1FAsymptoticCertificationError as exc:
            last_error = exc
        if current_precision == max_precision_bits:
            break
        current_precision = min(2 * current_precision, max_precision_bits)

    assert last_error is not None
    raise C1FAsymptoticCertificationError(
        f"Certification failed at precisions {attempted}: {last_error}"
    ) from last_error


def certify_complete_compatibility_unlabeled(
    multiplicity: int,
    analysis: Any,
    *,
    precision_bits: int = 192,
    max_precision_bits: int | None = None,
    tail_terms: int = 128,
    forbidden_cycle_depth: int = 0,
) -> C1FAsymptoticCertificate:
    """Certify the scalar subcritical Pólya schema and short-cycle exclusions."""

    _require_arb()
    if precision_bits < 64:
        raise ValueError("Interval certification requires at least 64 bits")
    if max_precision_bits is None:
        max_precision_bits = precision_bits
    if max_precision_bits < precision_bits:
        raise ValueError("max_precision_bits must be at least precision_bits")

    def nonnegative_ball(upper: Any) -> Any:
        bound = arb(upper.upper() if hasattr(upper, "upper") else upper)
        return arb(bound / 2, bound / 2)

    def evaluate_tails(radius: Any) -> tuple[list[Any], list[Any]]:
        values = [arb(0) for _ in range(tail_terms + 1)]
        derivatives = [arb(0) for _ in range(tail_terms + 1)]
        for index in range(tail_terms, 1, -1):
            repeats = tail_terms // index
            tail = sum(
                (
                    values[index * repeat] / repeat
                    for repeat in range(2, repeats + 1)
                ),
                arb(0),
            )
            first = radius ** (index * (repeats + 1))
            tail_bound = (
                multiplicity
                * first
                / (
                    (repeats + 1)
                    * (1 - radius**index)
                    * (1 - 4 * multiplicity * first)
                )
            )
            tail += nonnegative_ball(tail_bound)
            atom = multiplicity * radius**index * tail.exp()
            if not (1 - arb(1).exp() * atom).lower() > 0:
                raise C1FAsymptoticCertificationError(
                    "A Polya tail Lambert-W argument is not separated from its branch point"
                )
            values[index] = -(-atom).lambertw()

            derivative_tail = sum(
                (
                    derivatives[index * repeat] / repeat
                    for repeat in range(2, repeats + 1)
                ),
                arb(0),
            )
            derivative_bound = (
                4
                * multiplicity
                * index
                * radius ** (index * (repeats + 1) - 1)
                / (
                    (1 - radius**index)
                    * (1 - 4 * multiplicity * first) ** 2
                )
            )
            derivative_tail += nonnegative_ball(derivative_bound)
            derivatives[index] = values[index] * (
                index / radius + derivative_tail
            ) / (1 - values[index])
        return values, derivatives

    attempted = []
    current_precision = precision_bits
    last_error: Exception | None = None
    while current_precision <= max_precision_bits:
        attempted.append(current_precision)
        try:
            with ctx.workprec(current_precision):
                lower = arb(str(analysis.radius * 0.99))
                upper = arb(str(analysis.radius * 1.01))

                def characteristic(radius: Any) -> Any:
                    values, _ = evaluate_tails(radius)
                    return (
                        (multiplicity * radius).log()
                        + 1
                        + sum(
                            (
                                values[index] / index
                                for index in range(2, tail_terms + 1)
                            ),
                            arb(0),
                        )
                    )

                lower_value = characteristic(lower)
                upper_value = characteristic(upper)
                if not lower_value.upper() < 0 or not upper_value.lower() > 0:
                    raise C1FAsymptoticCertificationError(
                        "The initial Polya characteristic bracket has no certified sign change"
                    )
                target_width = arb(2) ** (-(current_precision - 16))
                for _ in range(current_precision + 64):
                    if upper - lower < target_width:
                        break
                    midpoint = (lower + upper) / 2
                    midpoint_value = characteristic(midpoint)
                    if midpoint_value.upper() < 0:
                        lower = midpoint
                    elif midpoint_value.lower() > 0:
                        upper = midpoint
                    else:
                        gap = upper - lower
                        lower = midpoint - gap / 4
                        upper = midpoint + gap / 4
                else:
                    raise C1FAsymptoticCertificationError(
                        "The Polya characteristic root could not be isolated"
                    )
                radius = arb((lower + upper) / 2, (upper - lower) / 2)
                values, derivatives = evaluate_tails(radius)
                values[1] = arb(1)
                characteristic_value = (
                    (multiplicity * radius).log()
                    + 1
                    + sum(
                        (
                            values[index] / index
                            for index in range(2, tail_terms + 1)
                        ),
                        arb(0),
                    )
                )
                if not characteristic_value.contains(0):
                    raise C1FAsymptoticCertificationError(
                        "The final Polya radius box does not contain the characteristic root"
                    )
                characteristic_derivative = 1 / radius + sum(
                    (
                        derivatives[index] / index
                        for index in range(2, tail_terms + 1)
                    ),
                    arb(0),
                )
                _require_positive(
                    "Polya characteristic derivative",
                    characteristic_derivative,
                )
                square_root = (2 * radius * characteristic_derivative).sqrt()
                log_factor = sum(
                    (
                        -(1 - values[index]).log()
                        for index in range(2, tail_terms + 1)
                    ),
                    arb(0),
                )
                cycle_correction = arb(0)
                correction_tail_bound = arb(0)
                for cycle_length in range(1, forbidden_cycle_depth + 1):
                    for adams in range(1, cycle_length + 1):
                        if cycle_length % adams:
                            continue
                        exponent = cycle_length // adams
                        for outer in range(1, tail_terms // adams + 1):
                            cycle_correction -= (
                                _integer_totient(adams)
                                * values[outer * adams] ** exponent
                                / (outer * cycle_length)
                            )
                        first_outer = tail_terms // adams + 1
                        first_power = radius ** (first_outer * adams)
                        tree_bound = multiplicity / (
                            1 - 4 * multiplicity * first_power
                        )
                        correction_tail_bound += (
                            _integer_totient(adams)
                            * tree_bound**exponent
                            * radius ** (first_outer * adams * exponent)
                            / (
                                cycle_length
                                * first_outer
                                * (1 - radius ** (adams * exponent))
                            )
                        )
                cycle_correction -= nonnegative_ball(correction_tail_bound)
                log_factor += cycle_correction
                first = radius ** (tail_terms + 1)
                factor_tail = (
                    multiplicity
                    * first
                    / (
                        (1 - radius)
                        * (1 - 5 * multiplicity * first)
                    )
                )
                log_factor += nonnegative_ball(factor_tail)
                leading_constant = log_factor.exp() / (
                    square_root * arb.pi().sqrt()
                )
                _require_positive("unlabeled leading constant", leading_constant)
                convert = lambda value: _decimal_interval(value, current_precision)

                certified_expansions = tuple()
                if analysis.residue_expansions:
                    scalar_specification = C1FLabeledSpecification(
                        depth=0,
                        successor=((True,),),
                        multiplicities=(multiplicity,),
                        visible_cycle_lengths=(None,),
                        transient_profiles=(0,),
                    )
                    half_order = max(
                        int(round(2.0 * (analysis.power - term.power)))
                        for residue in analysis.residue_expansions
                        for term in residue
                    )
                    formal_order = max(2, 1 + half_order) + 1
                    omitted_tolls, omitted_log = _polya_puiseux_tail_balls(
                        scalar_specification,
                        radius,
                        tail_terms,
                        formal_order,
                    )
                    level_values = [None] + [
                        (values[index],)
                        for index in range(1, tail_terms + 1)
                    ]
                    certified_expansions = _certify_unlabeled_puiseux_expansions(
                        scalar_specification,
                        analysis,
                        radius,
                        level_values,
                        (-square_root,),
                        ((arb(1),),),
                        ((arb(1),),),
                        current_precision,
                        omitted_tolls=omitted_tolls,
                        omitted_log=omitted_log,
                    )
                    certified_expansions = _replace_expansion_leading_intervals(
                        certified_expansions,
                        (convert(leading_constant),),
                    )

                return C1FAsymptoticCertificate(
                    precision_bits=current_precision,
                    method="arb-polya-tail-bisection",
                    regime=analysis.regime,
                    radius=convert(radius),
                    exponential_growth=convert(1 / radius),
                    leading_constant=convert(leading_constant),
                    finite_cycle_correction=convert(log_factor),
                    denominator_coefficient=convert(square_root),
                    tree_values=(convert(arb(1)),),
                    tree_sqrt_coefficients=(convert(square_root),),
                    perron_right=(convert(arb(1)),),
                    perron_left=(convert(arb(1)),),
                    verified_properties=(
                        "certified subcritical radius strictly below one",
                        "explicit plane-tree bounds for every omitted Polya tail",
                        "isolated positive characteristic root",
                        "positive square-root and analytic-factor constants",
                        "zero-free substituted determinants from T(x^k) < 1 for k >= 2",
                        "finite forbidden-cycle correction with an explicit tail bound",
                    ) + (
                        (
                            "recursive interval Pólya-Puiseux expansion",
                            "Cauchy bounds for every omitted expansion coefficient",
                        )
                        if certified_expansions
                        else tuple()
                    ),
                    dominance_certified=True,
                    residue_powers=(
                        analysis.residue_powers if certified_expansions else tuple()
                    ),
                    residue_leading_constants=(
                        tuple(
                            residue[0].constant if residue else None
                            for residue in certified_expansions
                        )
                        if certified_expansions
                        else tuple()
                    ),
                    residue_expansions=certified_expansions,
                )
        except (C1FAsymptoticCertificationError, ValueError, ZeroDivisionError) as exc:
            last_error = exc
        if current_precision == max_precision_bits:
            break
        current_precision = min(2 * current_precision, max_precision_bits)

    raise C1FAsymptoticCertificationError(
        f"Unlabeled certification failed at precisions {attempted}: {last_error}"
    ) from last_error


def certify_scalar_guarded_unlabeled(
    specification: C1FLabeledSpecification,
    analysis: Any,
    *,
    precision_bits: int = 192,
    max_precision_bits: int | None = None,
    tail_terms: int = 128,
) -> C1FAsymptoticCertificate:
    """Certify a one-coordinate guarded subcritical Pólya system."""

    _require_arb()
    if specification.tree_dimension != 1 or specification.cycle_dimension != 1:
        raise C1FAsymptoticCertificationError(
            "Scalar Pólya certification requires one tree and one long-cycle type"
        )
    if specification._long_cycle_profiles() != specification.transient_profiles:
        raise C1FAsymptoticCertificationError(
            "Scalar Pólya certification requires the long-cycle type to be productive"
        )
    if not specification.local_caps:
        raise C1FAsymptoticCertificationError(
            "Unguarded scalar systems use the complete-compatibility certificate"
        )
    if precision_bits < 64:
        raise ValueError("Interval certification requires at least 64 bits")
    if max_precision_bits is None:
        max_precision_bits = precision_bits
    if max_precision_bits < precision_bits:
        raise ValueError("max_precision_bits must be at least precision_bits")

    from .asymptotics import _exact_periodic_orbit, _solve_unlabeled_tree_levels

    aperiodicity_witness = _exact_unlabeled_tree_aperiodicity_witness(
        specification,
        (0,),
    )
    periodic_orbit = _exact_periodic_orbit(
        specification,
        (0,),
        unlabeled=True,
    )
    periodic_projection = bool(
        periodic_orbit is not None
        and analysis.periodicity == periodic_orbit.period
    )

    truncation = min(tail_terms, 32)
    truncation = max(truncation, 16, max(specification.local_caps))
    numerical_levels = _solve_unlabeled_tree_levels(
        specification,
        analysis.radius * (1.0 - 1e-8),
        truncation,
        tolerance=1e-12,
        max_iterations=120,
    )
    root_profile = specification.transient_profiles[0]
    contribution = _local_contribution(specification, root_profile)
    multiplicity = specification.multiplicities[root_profile]

    def nonnegative_ball(upper: Any) -> Any:
        bound = arb(upper.upper() if hasattr(upper, "upper") else upper)
        return arb(bound / 2, bound / 2)

    def distribution(
        x: _Jet,
        values: Sequence[_Jet],
        index: int,
    ) -> dict[tuple[int, ...], _Jet]:
        dimension = len(x.gradient)
        grouped: dict[tuple[int, ...], _Jet] = {}
        repeat_limit = truncation // index
        for repeat in range(1, repeat_limit + 1):
            state = tuple(
                min(cap, repeat * increment)
                for cap, increment in zip(
                    specification.local_caps,
                    contribution,
                )
            )
            grouped[state] = grouped.get(
                state,
                _Jet.constant(0, dimension),
            ) + values[index * repeat] / repeat

        first_repeat = repeat_limit + 1
        first_power = x.value ** (index * first_repeat)
        if not (1 - 4 * multiplicity * first_power).lower() > 0:
            raise C1FAsymptoticCertificationError(
                "The omitted scalar Pólya tail is not below its plane-tree bound"
            )
        omitted = (
            multiplicity
            * first_power
            / (
                first_repeat
                * (1 - x.value**index)
                * (1 - 4 * multiplicity * first_power)
            )
        )
        saturated = tuple(
            cap if increment else 0
            for cap, increment in zip(
                specification.local_caps,
                contribution,
            )
        )
        grouped[saturated] = grouped.get(
            saturated,
            _Jet.constant(0, dimension),
        ) + _Jet.constant(nonnegative_ball(omitted), dimension)

        zero = tuple(0 for _ in specification.local_caps)
        result = {zero: _Jet.constant(1, dimension)}
        for state, value in grouped.items():
            result = _distribution_multiply(
                result,
                _capped_exponential_factor(
                    value,
                    state,
                    specification.local_caps,
                ),
                specification.local_caps,
            )
        return result

    def system(raw_values: Sequence[Any]) -> tuple[_Jet, ...]:
        dimension = truncation + 1
        x = _Jet.variable(raw_values[0], 0, dimension)
        values = [_Jet.constant(0, dimension)]
        values.extend(
            _Jet.variable(raw_values[index], index, dimension)
            for index in range(1, truncation + 1)
        )
        equations = []
        first_distribution = None
        for index in range(1, truncation + 1):
            children = distribution(x, values, index)
            phi = (
                x**index
                * multiplicity
                * _projected_value(
                    specification,
                    root_profile,
                    children,
                    dimension,
                )
            )
            equations.append(values[index] - phi)
            if index == 1:
                first_distribution = children
        assert first_distribution is not None
        jacobian = (
            x
            * multiplicity
            * _projected_value(
                specification,
                root_profile,
                first_distribution,
                dimension,
                contribution,
            )
        )
        return tuple(equations) + (_Jet.constant(1, dimension) - jacobian,)

    centers = (
        analysis.radius,
        *(
            (numerical_levels[index] or (0.0,))[0]
            for index in range(1, truncation + 1)
        ),
    )
    attempted = []
    current_precision = precision_bits
    last_error: Exception | None = None
    while current_precision <= max_precision_bits:
        attempted.append(current_precision)
        try:
            with ctx.workprec(current_precision):
                root_box = _krawczyk_box(system, centers)
                radius = root_box[0]
                if not radius.lower() > 0 or not radius.upper() < 1:
                    raise C1FAsymptoticCertificationError(
                        "The scalar Pólya radius is not certified inside (0,1)"
                    )
                equations = system(root_box)
                tail_derivatives: list[Any | None] = [None] * (truncation + 1)
                for index in range(truncation, 1, -1):
                    equation = equations[index - 1]
                    forcing = equation.gradient[0]
                    for higher in range(index + 1, truncation + 1):
                        derivative = tail_derivatives[higher]
                        if derivative is not None:
                            forcing += equation.gradient[higher] * derivative
                    denominator = equation.gradient[index]
                    if denominator.contains(0):
                        raise C1FAsymptoticCertificationError(
                            "A substituted scalar tree Jacobian contains zero"
                        )
                    tail_derivatives[index] = -forcing / denominator

                first_equation = equations[0]
                effective_x_derivative = first_equation.gradient[0]
                for index in range(2, truncation + 1):
                    derivative = tail_derivatives[index]
                    if derivative is not None:
                        effective_x_derivative += (
                            first_equation.gradient[index] * derivative
                        )
                effective_x_derivative = -effective_x_derivative
                _require_positive(
                    "scalar Pólya characteristic x derivative",
                    effective_x_derivative,
                )

                dimension = truncation + 1
                x = _Jet.variable(radius, 0, dimension)
                values = [_Jet.constant(0, dimension)] + [
                    _Jet.variable(root_box[index], index, dimension)
                    for index in range(1, truncation + 1)
                ]
                distributions = [None] + [
                    distribution(x, values, index)
                    for index in range(1, truncation + 1)
                ]
                doubled = _capped_add(
                    contribution,
                    contribution,
                    specification.local_caps,
                )
                hessian = (
                    radius
                    * multiplicity
                    * _projected_value(
                        specification,
                        root_profile,
                        distributions[1],
                        dimension,
                        doubled,
                    ).value
                )
                _require_positive("scalar Pólya quadratic derivative", hessian)
                square_root = (
                    2 * radius * effective_x_derivative / hessian
                ).sqrt()
                determinant_coefficient = hessian * square_root
                _require_positive(
                    "scalar Pólya determinant coefficient",
                    determinant_coefficient,
                )

                transfers = [arb(0)] * (truncation + 1)
                for index in range(1, truncation + 1):
                    transfers[index] = (
                        radius**index
                        * multiplicity
                        * _projected_value(
                            specification,
                            root_profile,
                            distributions[index],
                            dimension,
                            contribution,
                        ).value
                    )
                if not transfers[1].contains(1):
                    raise C1FAsymptoticCertificationError(
                        "The k=1 cycle transfer does not contain one"
                    )
                log_factor = arb(0)
                for outer in range(1, truncation + 1):
                    for adams in range(1, truncation // outer + 1):
                        index = outer * adams
                        if outer == 1 and adams == 1:
                            continue
                        if not (1 - transfers[index]).lower() > 0:
                            raise C1FAsymptoticCertificationError(
                                "A substituted scalar cycle determinant is not zero-free"
                            )
                        log_factor -= (
                            _integer_totient(adams)
                            * (1 - transfers[index]).log()
                            / (outer * adams)
                        )
                for outer in range(1, truncation + 1):
                    for cycle_length in range(1, specification.depth + 1):
                        for adams in range(1, cycle_length + 1):
                            if (
                                cycle_length % adams
                                or outer * adams > truncation
                            ):
                                continue
                            log_factor -= (
                                _integer_totient(adams)
                                * transfers[outer * adams]
                                ** (cycle_length // adams)
                                / (outer * cycle_length)
                            )

                first = radius ** (truncation + 1)
                analytic_tail = (
                    multiplicity
                    * first
                    / (
                        (1 - radius)
                        * (1 - 5 * multiplicity * first)
                    )
                )
                log_factor += nonnegative_ball(analytic_tail)
                local_leading_constant = log_factor.exp() / (
                    determinant_coefficient * arb.pi().sqrt()
                )
                _require_positive(
                    "scalar guarded Pólya leading constant",
                    local_leading_constant,
                )
                convert = lambda value: _decimal_interval(value, current_precision)
                residue_constants: tuple[
                    C1FRealInterval | None, ...
                ] = tuple()
                certified_leading_constant = convert(local_leading_constant)
                if periodic_projection and periodic_orbit is not None:
                    periodic_levels = (tuple(),) + tuple(
                        (root_box[level],)
                        for level in range(1, truncation + 1)
                    )
                    residue_constants, certified_leading_constant = (
                        _certify_periodic_unlabeled_residues(
                            specification,
                            periodic_orbit,
                            analysis,
                            radius,
                            periodic_levels,
                            (square_root,),
                            multiplicity,
                            current_precision,
                        )
                    )
                certified_expansions = tuple()
                if analysis.residue_expansions:
                    half_order = max(
                        int(round(2.0 * (analysis.power - term.power)))
                        for residue in analysis.residue_expansions
                        for term in residue
                    )
                    formal_order = max(2, 1 + half_order) + 1
                    omitted_tolls, omitted_log = _polya_puiseux_tail_balls(
                        specification,
                        radius,
                        truncation,
                        formal_order,
                    )
                    level_values = [None] + [
                        (root_box[level],)
                        for level in range(1, truncation + 1)
                    ]
                    certified_expansions = _certify_unlabeled_puiseux_expansions(
                        specification,
                        analysis,
                        radius,
                        level_values,
                        (-square_root,),
                        ((arb(1),),),
                        ((arb(1),),),
                        current_precision,
                        omitted_tolls=omitted_tolls,
                        omitted_log=omitted_log,
                    )
                    certified_expansions = _replace_expansion_leading_intervals(
                        certified_expansions,
                        residue_constants
                        if residue_constants
                        else (certified_leading_constant,),
                    )
                    residue_constants = tuple(
                        residue[0].constant if residue else None
                        for residue in certified_expansions
                    )
                    certified_leading_constant = next(
                        residue[0].constant
                        for residue in certified_expansions
                        if residue
                    )
                return C1FAsymptoticCertificate(
                    precision_bits=current_precision,
                    method="arb-polya-finite-hierarchy-krawczyk",
                    regime=analysis.regime,
                    radius=convert(radius),
                    exponential_growth=convert(1 / radius),
                    leading_constant=certified_leading_constant,
                    finite_cycle_correction=convert(log_factor),
                    denominator_coefficient=convert(determinant_coefficient),
                    tree_values=(convert(root_box[1]),),
                    tree_sqrt_coefficients=(convert(square_root),),
                    perron_right=(convert(arb(1)),),
                    perron_left=(convert(arb(1)),),
                    verified_properties=(
                        "certified finite Pólya hierarchy with explicit omitted-tail bounds",
                        "isolated scalar characteristic solution below radius one",
                        "positive quadratic square-root scale",
                        "zero-free substituted cycle determinants",
                        "positive projected leading constant",
                    ) + (
                        (
                            "exact span-one critical-tree coefficient witness",
                            "monotone and phase-rigidity global dominance",
                        )
                        if aperiodicity_witness is not None
                        else (
                            "exact periodic critical-tree support congruences",
                            "complete root-of-unity dominant orbit",
                            "exact residue-class support projection",
                            "monotone and phase-rigidity global dominance",
                        )
                        if periodic_projection
                        else tuple()
                    ) + (
                        (
                            "recursive interval Pólya-Puiseux expansion",
                            "Cauchy bounds for every omitted expansion coefficient",
                            "interval Fourier projection through the requested order",
                        )
                        if certified_expansions
                        else tuple()
                    ),
                    dominance_certified=(
                        aperiodicity_witness is not None or periodic_projection
                    ),
                    residue_powers=(
                        analysis.residue_powers
                        if periodic_projection or certified_expansions
                        else tuple()
                    ),
                    residue_leading_constants=(
                        residue_constants
                        if periodic_projection or certified_expansions
                        else tuple()
                    ),
                    residue_expansions=certified_expansions,
                )
        except (C1FAsymptoticCertificationError, ValueError, ZeroDivisionError) as exc:
            last_error = exc
        if current_precision == max_precision_bits:
            break
        current_precision = min(2 * current_precision, max_precision_bits)

    raise C1FAsymptoticCertificationError(
        f"Scalar guarded Pólya certification failed at precisions {attempted}: "
        f"{last_error}"
    ) from last_error


def _certify_repeated_multitype_unlabeled(
    specification: C1FLabeledSpecification,
    analysis: Any,
    *,
    precision_bits: int,
    max_precision_bits: int | None,
    tail_terms: int,
) -> C1FAsymptoticCertificate:
    """Certify uncoupled isomorphic Pólya-critical blocks and their orbit."""

    from .asymptotics import _analyze_subcritical_unlabeled_specification

    long_profiles = _long_profiles(specification)
    transient_positions = {
        profile: index for index, profile in enumerate(specification.transient_profiles)
    }
    components = tuple(
        tuple(transient_positions[long_profiles[index]] for index in component)
        for component in analysis.critical_components
        if all(long_profiles[index] in transient_positions for index in component)
    )
    if len(components) != len(analysis.critical_components):
        raise C1FAsymptoticCertificationError(
            "Repeated Pólya certification requires tree-critical components"
        )
    block = _repeated_critical_block_specification(specification, components)
    expansion_order = max(
        (
            int(round(2.0 * (analysis.power - term.power)))
            for residue in analysis.residue_expansions
            for term in residue
        ),
        default=0,
    )
    block_analysis = _analyze_subcritical_unlabeled_specification(
        block,
        tolerance=1e-11,
        max_iterations=120,
        tail_terms=tail_terms,
        expansion_order=expansion_order,
    )
    block_certificate, context = certify_multitype_unlabeled(
        block,
        block_analysis,
        precision_bits=precision_bits,
        max_precision_bits=max_precision_bits,
        tail_terms=tail_terms,
        _return_context=True,
    )
    root_box = context["root_box"]
    x, block_values = context["unpack"](root_box)
    radius = x.value
    truncation = context["truncation"]
    tree_dimension = specification.tree_dimension
    level_values: list[tuple[Any, ...] | None] = [None]
    for level in range(1, truncation + 1):
        expanded = [arb(0) for _ in range(tree_dimension)]
        for component in components:
            for local, global_index in enumerate(component):
                expanded[global_index] = block_values[level][local].value
        level_values.append(tuple(expanded))

    block_sqrt = tuple(
        arb(value.enclosure) for value in block_certificate.tree_sqrt_coefficients
    )
    block_right = tuple(
        arb(value.enclosure) for value in block_certificate.perron_right
    )
    block_left = tuple(
        arb(value.enclosure) for value in block_certificate.perron_left
    )
    first = [arb(0) for _ in range(tree_dimension)]
    right_directions = []
    left_directions = []
    for component in components:
        right = [arb(0) for _ in range(tree_dimension)]
        left = [arb(0) for _ in range(tree_dimension)]
        for local, global_index in enumerate(component):
            first[global_index] = -block_sqrt[local]
            right[global_index] = block_right[local]
            left[global_index] = block_left[local]
        right_directions.append(tuple(right))
        left_directions.append(tuple(left))

    formal_order = max(2, len(components) + expansion_order) + 1
    omitted_tolls, omitted_log = _polya_puiseux_tail_balls(
        specification,
        radius,
        truncation,
        formal_order,
    )
    certified_expansions = _certify_unlabeled_puiseux_expansions(
        specification,
        analysis,
        radius,
        level_values,
        tuple(first),
        tuple(right_directions),
        tuple(left_directions),
        block_certificate.precision_bits,
        omitted_tolls=omitted_tolls,
        omitted_log=omitted_log,
    )
    leading = next(
        residue[0].constant for residue in certified_expansions if residue
    )
    copies = len(components)
    correction = arb(block_certificate.finite_cycle_correction.enclosure) * copies
    denominator = arb(block_certificate.denominator_coefficient.enclosure) ** copies
    convert = lambda value: _decimal_interval(value, block_certificate.precision_bits)
    trees = level_values[1] or tuple()
    sqrt_values = tuple(-value for value in first)
    aggregate_right = tuple(
        sum((direction[index] for direction in right_directions), arb(0))
        for index in range(tree_dimension)
    )
    aggregate_left = tuple(
        sum((direction[index] for direction in left_directions), arb(0))
        for index in range(tree_dimension)
    )
    return C1FAsymptoticCertificate(
        precision_bits=block_certificate.precision_bits,
        method="arb-polya-repeated-block-puiseux",
        regime=analysis.regime,
        radius=convert(radius),
        exponential_growth=convert(1 / radius),
        leading_constant=leading,
        finite_cycle_correction=convert(correction),
        denominator_coefficient=convert(denominator),
        tree_values=tuple(convert(value) for value in trees),
        tree_sqrt_coefficients=tuple(convert(value) for value in sqrt_values),
        perron_right=tuple(convert(value) for value in aggregate_right),
        perron_left=tuple(convert(value) for value in aggregate_left),
        verified_properties=(
            "exact decomposition into uncoupled isomorphic Pólya-critical blocks",
            "certified representative finite Pólya hierarchy",
            "positive square-root scale in every critical block",
            "complete interacting root-of-unity orbit",
            "recursive interval Pólya-Puiseux expansion",
            "Cauchy bounds for every omitted expansion coefficient",
            "interval Fourier projection through the requested order",
        ),
        dominance_certified=block_certificate.dominance_certified,
        residue_powers=analysis.residue_powers,
        residue_leading_constants=tuple(
            residue[0].constant if residue else None
            for residue in certified_expansions
        ),
        residue_expansions=certified_expansions,
    )


def certify_multitype_unlabeled(
    specification: C1FLabeledSpecification,
    analysis: Any,
    *,
    precision_bits: int = 192,
    max_precision_bits: int | None = None,
    tail_terms: int = 128,
    _return_context: bool = False,
) -> Any:
    """Certify one regular multitype tree-critical Pólya system."""

    _require_arb()
    if len(analysis.critical_components) != 1:
        return _certify_repeated_multitype_unlabeled(
            specification,
            analysis,
            precision_bits=precision_bits,
            max_precision_bits=max_precision_bits,
            tail_terms=tail_terms,
        )
    long_profiles = _long_profiles(specification)
    transient_positions = {
        profile: index
        for index, profile in enumerate(specification.transient_profiles)
    }
    critical_profiles = tuple(
        long_profiles[index] for index in analysis.critical_components[0]
    )
    tree_critical = all(
        profile in transient_positions for profile in critical_profiles
    ) and analysis.power < -0.25
    critical_component = (
        tuple(transient_positions[profile] for profile in critical_profiles)
        if tree_critical
        else tuple(analysis.critical_components[0])
    )
    if precision_bits < 64:
        raise ValueError("Interval certification requires at least 64 bits")
    if max_precision_bits is None:
        max_precision_bits = precision_bits
    if max_precision_bits < precision_bits:
        raise ValueError("max_precision_bits must be at least precision_bits")

    from .asymptotics import (
        _exact_periodic_orbit,
        _is_strongly_connected,
        _solve_unlabeled_tree_levels,
        _strongly_connected_components,
    )

    dependency = specification.dependency_matrix()
    critical_dependency = tuple(
        tuple(dependency[row][column] for column in critical_component)
        for row in critical_component
    ) if tree_critical else tuple()
    structurally_irreducible = bool(
        tree_critical and _is_strongly_connected(critical_dependency)
    )
    aperiodicity_witness = (
        _exact_unlabeled_tree_aperiodicity_witness(
            specification,
            critical_component,
        )
        if structurally_irreducible
        else None
    )
    periodic_orbit = (
        _exact_periodic_orbit(
            specification,
            critical_component,
            unlabeled=True,
        )
        if structurally_irreducible
        else None
    )
    periodic_projection = bool(
        periodic_orbit is not None
        and analysis.periodicity == periodic_orbit.period
    )

    tree_dimension = specification.tree_dimension
    truncation = min(tail_terms, 16)
    truncation = max(truncation, max(specification.local_caps, default=0), 12)
    numerical_levels = _solve_unlabeled_tree_levels(
        specification,
        analysis.radius * (1.0 - 1e-8) if tree_critical else analysis.radius,
        truncation,
        tolerance=1e-12,
        max_iterations=120,
    )
    total_multiplicity = sum(specification.multiplicities)

    def nonnegative_ball(upper: Any) -> Any:
        bound = arb(upper.upper() if hasattr(upper, "upper") else upper)
        return arb(bound / 2, bound / 2)

    def variable_index(level: int, coordinate: int) -> int:
        return 1 + (level - 1) * tree_dimension + coordinate

    def distributions(
        x: _Jet,
        values: Sequence[Sequence[_Jet]],
        level: int,
        root_profiles: Sequence[int],
    ) -> dict[int, dict[tuple[int, ...], _Jet]]:
        dimension = len(x.gradient)
        repeat_limit = truncation // level
        result = {}
        for root_profile in root_profiles:
            grouped: dict[tuple[int, ...], _Jet] = {}
            omitted_states: set[tuple[int, ...]] = set()
            for child_coordinate, child_profile in enumerate(
                specification.transient_profiles
            ):
                if not specification.successor[child_profile][root_profile]:
                    continue
                child_contribution = _local_contribution(
                    specification,
                    child_profile,
                )
                for repeat in range(1, repeat_limit + 1):
                    state = tuple(
                        min(cap, repeat * increment)
                        for cap, increment in zip(
                            specification.local_caps,
                            child_contribution,
                        )
                    )
                    grouped[state] = grouped.get(
                        state,
                        _Jet.constant(0, dimension),
                    ) + values[level * repeat][child_coordinate] / repeat
                omitted_states.add(
                    tuple(
                        cap if increment else 0
                        for cap, increment in zip(
                            specification.local_caps,
                            child_contribution,
                        )
                    )
                )

            if omitted_states:
                first_repeat = repeat_limit + 1
                first_power = x.value ** (level * first_repeat)
                if not (1 - 4 * total_multiplicity * first_power).lower() > 0:
                    raise C1FAsymptoticCertificationError(
                        "A multitype Pólya tail is not below its plane-tree bound"
                    )
                omitted = (
                    total_multiplicity
                    * first_power
                    / (
                        first_repeat
                        * (1 - x.value**level)
                        * (1 - 4 * total_multiplicity * first_power)
                    )
                )
                for state in omitted_states:
                    grouped[state] = grouped.get(
                        state,
                        _Jet.constant(0, dimension),
                    ) + _Jet.constant(nonnegative_ball(omitted), dimension)

            zero = tuple(0 for _ in specification.local_caps)
            child_distribution = {zero: _Jet.constant(1, dimension)}
            for state, value in grouped.items():
                child_distribution = _distribution_multiply(
                    child_distribution,
                    _capped_exponential_factor(
                        value,
                        state,
                        specification.local_caps,
                    ),
                    specification.local_caps,
                )
            result[root_profile] = child_distribution
        return result

    def tree_phi_and_jacobian(
        x: _Jet,
        values: Sequence[Sequence[_Jet]],
        level: int,
    ) -> tuple[tuple[_Jet, ...], tuple[tuple[_Jet, ...], ...]]:
        dimension = len(x.gradient)
        root_profiles = specification.transient_profiles
        child_distributions = distributions(x, values, level, root_profiles)
        atom = x**level
        phi = tuple(
            atom
            * specification.multiplicities[root_profile]
            * _projected_value(
                specification,
                root_profile,
                child_distributions[root_profile],
                dimension,
            )
            for root_profile in root_profiles
        )
        jacobian = tuple(
            tuple(
                _Jet.constant(0, dimension)
                if not specification.successor[child_profile][root_profile]
                else atom
                * specification.multiplicities[root_profile]
                * _projected_value(
                    specification,
                    root_profile,
                    child_distributions[root_profile],
                    dimension,
                    _local_contribution(specification, child_profile),
                )
                for child_profile in specification.transient_profiles
            )
            for root_profile in root_profiles
        )
        return phi, jacobian

    def cycle_matrix(
        x: _Jet,
        values: Sequence[Sequence[_Jet]],
        level: int,
        profiles: Sequence[int],
    ) -> tuple[tuple[_Jet, ...], ...]:
        dimension = len(x.gradient)
        child_distributions = distributions(x, values, level, profiles)
        atom = x**level
        return tuple(
            tuple(
                _Jet.constant(0, dimension)
                if not specification.successor[previous][current]
                else atom
                * specification.multiplicities[current]
                * _projected_value(
                    specification,
                    current,
                    child_distributions[current],
                    dimension,
                    _local_contribution(specification, previous),
                )
                for current in profiles
            )
            for previous in profiles
        )

    system_dimension = 1 + truncation * tree_dimension

    def unpack(raw_values: Sequence[Any]) -> tuple[_Jet, list[tuple[_Jet, ...]]]:
        x = _Jet.variable(raw_values[0], 0, system_dimension)
        values = [tuple()]
        for level in range(1, truncation + 1):
            values.append(
                tuple(
                    _Jet.variable(
                        raw_values[variable_index(level, coordinate)],
                        variable_index(level, coordinate),
                        system_dimension,
                    )
                    for coordinate in range(tree_dimension)
                )
            )
        return x, values

    def system(raw_values: Sequence[Any]) -> tuple[_Jet, ...]:
        x, values = unpack(raw_values)
        equations = []
        first_jacobian = None
        first_cycle_transfer = None
        for level in range(1, truncation + 1):
            phi, jacobian = tree_phi_and_jacobian(x, values, level)
            equations.extend(
                tree - value for tree, value in zip(values[level], phi)
            )
            if level == 1:
                first_jacobian = jacobian
                first_cycle_transfer = cycle_matrix(
                    x,
                    values,
                    1,
                    long_profiles,
                )
        assert first_jacobian is not None
        equations.append(
            _determinant_jet(
                _identity_minus(
                    first_jacobian
                    if tree_critical
                    else first_cycle_transfer
                )
            )
        )
        return tuple(equations)

    centers = [analysis.radius]
    for level in range(1, truncation + 1):
        assert numerical_levels[level] is not None
        centers.extend(numerical_levels[level])

    attempted = []
    current_precision = precision_bits
    last_error: Exception | None = None
    while current_precision <= max_precision_bits:
        attempted.append(current_precision)
        try:
            with ctx.workprec(current_precision):
                root_box = _krawczyk_box(system, centers)
                radius = root_box[0]
                if not radius.lower() > 0 or not radius.upper() < 1:
                    raise C1FAsymptoticCertificationError(
                        "The multitype Pólya radius is not certified inside (0,1)"
                    )
                x, values = unpack(root_box)
                equations = system(root_box)

                tail_derivatives: list[tuple[Any, ...] | None] = [
                    None for _ in range(truncation + 1)
                ]
                last_implicit_level = 2 if tree_critical else 1
                for level in range(truncation, last_implicit_level - 1, -1):
                    row_offset = (level - 1) * tree_dimension
                    operator = arb_mat(
                        tree_dimension,
                        tree_dimension,
                        tuple(
                            equations[row_offset + row].gradient[
                                variable_index(level, column)
                            ]
                            for row in range(tree_dimension)
                            for column in range(tree_dimension)
                        ),
                    )
                    forcing_values = []
                    for row in range(tree_dimension):
                        equation = equations[row_offset + row]
                        forcing = equation.gradient[0]
                        for higher in range(level + 1, truncation + 1):
                            derivative = tail_derivatives[higher]
                            if derivative is None:
                                continue
                            forcing += sum(
                                equation.gradient[
                                    variable_index(higher, column)
                                ]
                                * derivative[column]
                                for column in range(tree_dimension)
                            )
                        forcing_values.append(-forcing)
                    try:
                        derivative_column = operator.solve(
                            arb_mat(tree_dimension, 1, tuple(forcing_values))
                        )
                    except ZeroDivisionError as exc:
                        raise C1FAsymptoticCertificationError(
                            "A substituted multitype tree Jacobian is singular"
                        ) from exc
                    tail_derivatives[level] = tuple(
                        derivative_column[index, 0]
                        for index in range(tree_dimension)
                    )

                _, first_jacobian_jets = tree_phi_and_jacobian(x, values, 1)
                first_jacobian = arb_mat(
                    tree_dimension,
                    tree_dimension,
                    tuple(
                        first_jacobian_jets[row][column].value
                        for row in range(tree_dimension)
                        for column in range(tree_dimension)
                    ),
                )
                dominance_checks_succeeded = bool(
                    tree_critical
                    and (
                        aperiodicity_witness is not None
                        or periodic_projection
                    )
                )
                if dominance_checks_succeeded:
                    try:
                        tree_components = _strongly_connected_components(
                            dependency
                        )
                        if tuple(critical_component) not in tree_components:
                            raise C1FAsymptoticCertificationError(
                                "The selected critical tree component is not an exact SCC"
                            )
                        for component in tree_components:
                            if component == tuple(critical_component):
                                continue
                            block = arb_mat(
                                len(component),
                                len(component),
                                tuple(
                                    first_jacobian[row, column]
                                    for row in component
                                    for column in component
                                ),
                            )
                            _certify_nonnegative_matrix_subcritical(
                                block,
                                "noncritical k=1 tree block",
                            )
                        first_cycle_jets = cycle_matrix(
                            x,
                            values,
                            1,
                            long_profiles,
                        )
                        cycle_support = []
                        for row in first_cycle_jets:
                            support_row = []
                            for entry in row:
                                if _strictly_positive(entry.value):
                                    support_row.append(True)
                                elif entry.value.is_zero():
                                    support_row.append(False)
                                else:
                                    raise C1FAsymptoticCertificationError(
                                        "A k=1 cycle-transfer support entry is inconclusive"
                                    )
                            cycle_support.append(tuple(support_row))
                        cycle_components = _strongly_connected_components(
                            tuple(cycle_support)
                        )
                        selected_cycle_component = tuple(
                            analysis.critical_components[0]
                        )
                        if selected_cycle_component not in cycle_components:
                            raise C1FAsymptoticCertificationError(
                                "The selected critical cycle component is not an exact SCC"
                            )
                        first_cycle = _arb_matrix(first_cycle_jets)
                        for component in cycle_components:
                            if component == selected_cycle_component:
                                continue
                            block = arb_mat(
                                len(component),
                                len(component),
                                tuple(
                                    first_cycle[row, column]
                                    for row in component
                                    for column in component
                                ),
                            )
                            _certify_nonnegative_matrix_subcritical(
                                block,
                                "noncritical k=1 cycle block",
                            )
                    except C1FAsymptoticCertificationError:
                        dominance_checks_succeeded = False
                if tree_critical:
                    effective_x_derivative = []
                    for row in range(tree_dimension):
                        equation = equations[row]
                        forcing = equation.gradient[0]
                        for level in range(2, truncation + 1):
                            derivative = tail_derivatives[level]
                            if derivative is None:
                                continue
                            forcing += sum(
                                equation.gradient[
                                    variable_index(level, column)
                                ]
                                * derivative[column]
                                for column in range(tree_dimension)
                            )
                        effective_x_derivative.append(-forcing)
                    right, left = _normalized_full_kernel_vectors(
                        first_jacobian,
                        critical_component,
                    )
                    for coordinate in critical_component:
                        _require_positive(
                            f"multitype Pólya right Perron coordinate {coordinate}",
                            right[coordinate],
                        )
                        _require_positive(
                            f"multitype Pólya left Perron coordinate {coordinate}",
                            left[coordinate],
                        )

                    first_distributions = distributions(
                        x,
                        values,
                        1,
                        specification.transient_profiles,
                    )
                    hessian = []
                    for root_profile in specification.transient_profiles:
                        pointed: dict[tuple[int, ...], Any] = {}
                        for child_coordinate, child_profile in enumerate(
                            specification.transient_profiles
                        ):
                            if not specification.successor[child_profile][root_profile]:
                                continue
                            state = _local_contribution(
                                specification,
                                child_profile,
                            )
                            pointed[state] = pointed.get(state, arb(0)) + right[
                                child_coordinate
                            ]
                        double_pointed: dict[tuple[int, ...], Any] = {}
                        for left_state, left_value in pointed.items():
                            for right_state, right_value in pointed.items():
                                state = _capped_add(
                                    left_state,
                                    right_state,
                                    specification.local_caps,
                                )
                                double_pointed[state] = (
                                    double_pointed.get(state, arb(0))
                                    + left_value * right_value
                                )
                        value = arb(0)
                        for state, coefficient in double_pointed.items():
                            value += coefficient * _projected_value(
                                specification,
                                root_profile,
                                first_distributions[root_profile],
                                system_dimension,
                                state,
                            ).value
                        hessian.append(
                            radius
                            * specification.multiplicities[root_profile]
                            * value
                        )
                    numerator = 2 * radius * sum(
                        (
                            left[index] * effective_x_derivative[index]
                            for index in range(tree_dimension)
                        ),
                        arb(0),
                    )
                    denominator = sum(
                        (
                            left[index] * hessian[index]
                            for index in range(tree_dimension)
                        ),
                        arb(0),
                    )
                    _require_positive(
                        "multitype Pólya square-root numerator",
                        numerator,
                    )
                    _require_positive(
                        "multitype Pólya square-root denominator",
                        denominator,
                    )
                    kappa = (numerator / denominator).sqrt()
                    tree_sqrt = tuple(kappa * value for value in right)
                    directional_x = _Jet.constant(radius, 1)
                    directional_values = [tuple()]
                    directional_values.append(
                        tuple(
                            _Jet(
                                values[1][coordinate].value,
                                (-tree_sqrt[coordinate],),
                            )
                            for coordinate in range(tree_dimension)
                        )
                    )
                    for level in range(2, truncation + 1):
                        directional_values.append(
                            tuple(
                                _Jet.constant(value.value, 1)
                                for value in values[level]
                            )
                        )
                else:
                    tree_operator = _arb_identity(tree_dimension) - first_jacobian
                    ones = arb_mat(
                        tree_dimension,
                        1,
                        tuple(arb(1) for _ in range(tree_dimension)),
                    )
                    try:
                        weights = tree_operator.mid().solve(ones)
                    except ZeroDivisionError as exc:
                        raise C1FAsymptoticCertificationError(
                            "The k=1 Pólya tree Jacobian is not separated from one"
                        ) from exc
                    slack = weights - first_jacobian * weights
                    for coordinate in range(tree_dimension):
                        _require_positive(
                            f"Pólya tree subcritical weight {coordinate}",
                            weights[coordinate, 0],
                        )
                        _require_positive(
                            f"Pólya tree subcritical slack {coordinate}",
                            slack[coordinate, 0],
                        )
                    first_cycle = _arb_matrix(
                        cycle_matrix(x, values, 1, long_profiles)
                    )
                    right, left = _normalized_full_kernel_vectors(
                        first_cycle,
                        critical_component,
                    )
                    for coordinate in critical_component:
                        _require_positive(
                            f"Pólya cycle right Perron coordinate {coordinate}",
                            right[coordinate],
                        )
                        _require_positive(
                            f"Pólya cycle left Perron coordinate {coordinate}",
                            left[coordinate],
                        )
                    tree_sqrt = tuple(arb(0) for _ in range(tree_dimension))
                    directional_x = _Jet(radius, (-radius,))
                    directional_values = [tuple()]
                    for level in range(1, truncation + 1):
                        derivative = tail_derivatives[level]
                        assert derivative is not None
                        directional_values.append(
                            tuple(
                                _Jet(
                                    values[level][coordinate].value,
                                    (-radius * derivative[coordinate],),
                                )
                                for coordinate in range(tree_dimension)
                            )
                        )
                directional_transfer = cycle_matrix(
                    directional_x,
                    directional_values,
                    1,
                    long_profiles,
                )
                determinant_coefficient = _determinant_jet(
                    _identity_minus(directional_transfer)
                ).gradient[0]
                _require_positive(
                    "multitype Pólya determinant coefficient",
                    determinant_coefficient,
                )

                matrix_cache: dict[
                    tuple[tuple[int, ...], int], tuple[tuple[_Jet, ...], ...]
                ] = {}

                def transfer_matrix(
                    profiles: tuple[int, ...],
                    level: int,
                ) -> tuple[tuple[_Jet, ...], ...]:
                    key = profiles, level
                    cached = matrix_cache.get(key)
                    if cached is None:
                        cached = cycle_matrix(x, values, level, profiles)
                        matrix_cache[key] = cached
                    return cached

                log_factor = arb(0)
                for outer in range(1, truncation + 1):
                    for adams in range(1, truncation // outer + 1):
                        if outer == 1 and adams == 1:
                            continue
                        matrix_value = _arb_matrix(
                            transfer_matrix(long_profiles, outer * adams)
                        )
                        determinant = (_arb_identity(matrix_value.nrows()) - matrix_value).det()
                        _require_positive(
                            "substituted multitype cycle determinant",
                            determinant,
                        )
                        if dominance_checks_succeeded:
                            try:
                                _certify_nonnegative_matrix_subcritical(
                                    matrix_value,
                                    f"Adams cycle matrix k={outer * adams}",
                                )
                            except C1FAsymptoticCertificationError:
                                dominance_checks_succeeded = False
                        log_factor -= (
                            _integer_totient(adams)
                            * determinant.log()
                            / (outer * adams)
                        )

                for outer in range(1, truncation + 1):
                    for cycle_length in range(1, specification.depth + 1):
                        visible_profiles = tuple(
                            profile
                            for profile, visible_length in enumerate(
                                specification.visible_cycle_lengths
                            )
                            if visible_length == cycle_length
                        )
                        for adams in range(1, cycle_length + 1):
                            if (
                                cycle_length % adams
                                or outer * adams > truncation
                            ):
                                continue
                            exponent = cycle_length // adams
                            visible = _arb_matrix(
                                transfer_matrix(
                                    visible_profiles,
                                    outer * adams,
                                )
                            )
                            long = _arb_matrix(
                                transfer_matrix(long_profiles, outer * adams)
                            )
                            log_factor += (
                                _integer_totient(adams)
                                * (
                                    (visible**exponent).trace()
                                    - (long**exponent).trace()
                                )
                                / (outer * cycle_length)
                            )

                first = radius ** (truncation + 1)
                tail_separation = 1 - 5 * total_multiplicity * first
                _require_positive(
                    "automatic Pólya tail spectral separation",
                    tail_separation,
                )
                analytic_tail = (
                    total_multiplicity
                    * first
                    / (
                        (1 - radius)
                        * tail_separation
                    )
                )
                log_factor += arb(0, analytic_tail)
                transfer_gamma = arb.pi().sqrt() if tree_critical else arb(1)
                local_leading_constant = log_factor.exp() / (
                    determinant_coefficient * transfer_gamma
                )
                _require_positive(
                    "multitype Pólya leading constant",
                    local_leading_constant,
                )
                convert = lambda value: _decimal_interval(value, current_precision)
                residue_constants: tuple[
                    C1FRealInterval | None, ...
                ] = tuple()
                certified_leading_constant = convert(local_leading_constant)
                if periodic_projection and periodic_orbit is not None:
                    periodic_levels = (tuple(),) + tuple(
                        tuple(value.value for value in values[level])
                        for level in range(1, truncation + 1)
                    )
                    residue_constants, certified_leading_constant = (
                        _certify_periodic_unlabeled_residues(
                            specification,
                            periodic_orbit,
                            analysis,
                            radius,
                            periodic_levels,
                            tree_sqrt,
                            total_multiplicity,
                            current_precision,
                        )
                    )
                certified_expansions = tuple()
                if tree_critical and analysis.residue_expansions:
                    half_order = max(
                        int(round(2.0 * (analysis.power - term.power)))
                        for residue in analysis.residue_expansions
                        for term in residue
                    )
                    formal_order = max(2, 1 + half_order) + 1
                    omitted_tolls, omitted_log = _polya_puiseux_tail_balls(
                        specification,
                        radius,
                        truncation,
                        formal_order,
                    )
                    level_values = [None] + [
                        tuple(value.value for value in values[level])
                        for level in range(1, truncation + 1)
                    ]
                    certified_expansions = _certify_unlabeled_puiseux_expansions(
                        specification,
                        analysis,
                        radius,
                        level_values,
                        tuple(-value for value in tree_sqrt),
                        (right,),
                        (left,),
                        current_precision,
                        omitted_tolls=omitted_tolls,
                        omitted_log=omitted_log,
                    )
                    certified_expansions = _replace_expansion_leading_intervals(
                        certified_expansions,
                        residue_constants
                        if residue_constants
                        else (certified_leading_constant,),
                    )
                    residue_constants = tuple(
                        residue[0].constant if residue else None
                        for residue in certified_expansions
                    )
                    certified_leading_constant = next(
                        residue[0].constant
                        for residue in certified_expansions
                        if residue
                    )
                certificate = C1FAsymptoticCertificate(
                    precision_bits=current_precision,
                    method=(
                        "arb-polya-multitype-krawczyk"
                        if tree_critical
                        else "arb-polya-analytic-tree-cycle-pole"
                    ),
                    regime=analysis.regime,
                    radius=convert(radius),
                    exponential_growth=convert(1 / radius),
                    leading_constant=certified_leading_constant,
                    finite_cycle_correction=convert(log_factor),
                    denominator_coefficient=convert(determinant_coefficient),
                    tree_values=tuple(
                        convert(value.value) for value in values[1]
                    ),
                    tree_sqrt_coefficients=tuple(
                        convert(value) for value in tree_sqrt
                    ),
                    perron_right=tuple(convert(value) for value in right),
                    perron_left=tuple(convert(value) for value in left),
                    verified_properties=(
                        "certified multitype Pólya hierarchy with omitted-tail bounds",
                        "isolated regular characteristic solution below radius one",
                        (
                            "positive full Perron directions and quadratic scale"
                            if tree_critical
                            else "strictly subcritical tree hierarchy and positive cycle Perron directions"
                        ),
                        "zero-free retained substituted cycle determinants",
                        "nonzero projected multitype leading constant",
                    ) + (
                        (
                            "exact span-one critical-tree coefficient witness",
                            "monotone majorant excludes interior singularities",
                            "phase rigidity excludes nonpositive boundary singularities",
                            "all noncritical tree/cycle and Adams blocks have spectral radius below one",
                        )
                        if dominance_checks_succeeded
                        and aperiodicity_witness is not None
                        else (
                            "exact periodic critical-tree support congruences",
                            "complete root-of-unity dominant orbit",
                            "exact residue-class support projection",
                            "monotone majorant excludes interior singularities",
                            "phase rigidity excludes unlisted boundary singularities",
                            "all noncritical tree/cycle and Adams blocks have spectral radius below one",
                        )
                        if dominance_checks_succeeded and periodic_projection
                        else tuple()
                    ) + (
                        (
                            "recursive interval Pólya-Puiseux expansion",
                            "Cauchy bounds for every omitted expansion coefficient",
                            "interval Fourier projection through the requested order",
                        )
                        if certified_expansions
                        else tuple()
                    ),
                    dominance_certified=dominance_checks_succeeded,
                    residue_powers=(
                        analysis.residue_powers
                        if periodic_projection or certified_expansions
                        else tuple()
                    ),
                    residue_leading_constants=(
                        residue_constants
                        if periodic_projection or certified_expansions
                        else tuple()
                    ),
                    residue_expansions=certified_expansions,
                )
                if not _return_context:
                    return certificate
                return certificate, {
                    "root_box": root_box,
                    "system": system,
                    "unpack": unpack,
                    "distributions": distributions,
                    "tree_phi_and_jacobian": tree_phi_and_jacobian,
                    "truncation": truncation,
                    "tree_dimension": tree_dimension,
                    "system_dimension": system_dimension,
                    "total_multiplicity": total_multiplicity,
                }
        except (C1FAsymptoticCertificationError, ValueError, ZeroDivisionError) as exc:
            last_error = exc
        if current_precision == max_precision_bits:
            break
        current_precision = min(2 * current_precision, max_precision_bits)

    raise C1FAsymptoticCertificationError(
        f"Multitype Pólya certification failed at precisions {attempted}: "
        f"{last_error}"
    ) from last_error


def certify_cycle_pole_unlabeled(
    specification: C1FLabeledSpecification,
    analysis: Any,
    *,
    precision_bits: int = 192,
    max_precision_bits: int | None = None,
    tail_terms: int = 128,
) -> C1FAsymptoticCertificate:
    """Certify a pure subcritical unlabeled cycle-pole system."""

    _require_arb()
    if specification.tree_dimension != 0:
        raise C1FAsymptoticCertificationError(
            "The pure-cycle Pólya certificate requires analytic zero tree coordinates"
        )
    long_profiles = _long_profiles(specification)
    if analysis.periodicity != 1 and (
        len(analysis.critical_components) != 1
        or analysis.critical_components[0] != tuple(range(len(long_profiles)))
        or any(
            visible_length is not None
            for visible_length in specification.visible_cycle_lengths
        )
    ):
        raise C1FAsymptoticCertificationError(
            "This periodic unlabeled cycle pole requires complex residue projection"
        )
    from .asymptotics import analyze_c1f_labeled_asymptotics

    labeled = analyze_c1f_labeled_asymptotics(
        specification,
        certify=True,
        precision_bits=precision_bits,
        max_precision_bits=max_precision_bits,
    )
    labeled_certificate = labeled.certificate
    assert labeled_certificate is not None
    if labeled.regime != "cycle-pole" or labeled.cycle_pole_order is None:
        raise C1FAsymptoticCertificationError(
            "The zero-tree specification is not an exact finite cycle pole"
        )
    if not labeled_certificate.dominance_certified:
        raise C1FAsymptoticCertificationError(
            "The underlying cycle pole lacks an exact dominance certificate"
        )
    if not Decimal(labeled_certificate.radius.upper) < Decimal(1):
        raise C1FAsymptoticCertificationError(
            "The unlabeled cycle pole is not strictly below radius one"
        )

    def integer_matrix(profiles: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        numerical = specification.cycle_matrix(profiles, 1.0, tuple())
        rounded = tuple(
            tuple(int(round(value)) for value in row)
            for row in numerical
        )
        if any(
            abs(numerical[row][column] - rounded[row][column]) > 1e-12
            for row in range(len(rounded))
            for column in range(len(rounded))
        ):
            raise C1FAsymptoticCertificationError(
                "A pure-cycle transfer matrix is not integral"
            )
        return rounded

    long_matrix = integer_matrix(long_profiles)
    visible_matrices = tuple(
        integer_matrix(
            tuple(
                profile
                for profile, visible_length in enumerate(
                    specification.visible_cycle_lengths
                )
                if visible_length == cycle_length
            )
        )
        for cycle_length in range(1, specification.depth + 1)
    )
    max_row_sum = max(
        (sum(row) for row in long_matrix),
        default=0,
    )
    matrix_dimension = len(long_matrix)
    current_precision = labeled_certificate.precision_bits
    with ctx.workprec(current_precision):
        radius = arb(labeled_certificate.radius.enclosure)
        eta = arb(labeled_certificate.denominator_coefficient.enclosure)
        pole_order = labeled.cycle_pole_order

        def scaled_matrix(matrix: Sequence[Sequence[int]], exponent: int) -> Any:
            scale = radius**exponent
            return arb_mat(
                len(matrix),
                len(matrix),
                tuple(scale * value for row in matrix for value in row),
            )

        log_factor = arb(0)
        for outer in range(1, tail_terms + 1):
            for adams in range(1, tail_terms // outer + 1):
                if outer == 1 and adams == 1:
                    continue
                transfer = scaled_matrix(long_matrix, outer * adams)
                determinant = (_arb_identity(transfer.nrows()) - transfer).det()
                _require_positive(
                    "pure-cycle substituted determinant",
                    determinant,
                )
                log_factor -= (
                    _integer_totient(adams)
                    * determinant.log()
                    / (outer * adams)
                )

        for outer in range(1, tail_terms + 1):
            for cycle_length, visible_matrix in enumerate(visible_matrices, 1):
                for adams in range(1, cycle_length + 1):
                    if (
                        cycle_length % adams
                        or outer * adams > tail_terms
                    ):
                        continue
                    exponent = cycle_length // adams
                    visible = scaled_matrix(visible_matrix, outer * adams)
                    long = scaled_matrix(long_matrix, outer * adams)
                    log_factor += (
                        _integer_totient(adams)
                        * (
                            (visible**exponent).trace()
                            - (long**exponent).trace()
                        )
                        / (outer * cycle_length)
                    )

        first = radius ** (tail_terms + 1)
        if max_row_sum and not (1 - max_row_sum * first).lower() > 0:
            raise C1FAsymptoticCertificationError(
                "The pure-cycle Pólya tail is not geometrically separated"
            )
        determinant_tail = (
            matrix_dimension
            * max_row_sum
            * first
            / (
                (1 - radius)
                * (1 - max_row_sum * first)
            )
            if max_row_sum
            else arb(0)
        )
        correction_tail = determinant_tail * max(1, specification.depth)
        log_factor += arb(0, determinant_tail + correction_tail)
        leading_constant = log_factor.exp() / (
            eta * arb(pole_order).gamma()
        )
        if analysis.periodicity != 1:
            leading_constant *= analysis.periodicity
        _require_positive(
            "pure-cycle unlabeled leading constant",
            leading_constant,
        )
        convert = lambda value: _decimal_interval(value, current_precision)
        return C1FAsymptoticCertificate(
            precision_bits=current_precision,
            method="arb-polya-exact-cycle-pole",
            regime=analysis.regime,
            radius=convert(radius),
            exponential_growth=convert(1 / radius),
            leading_constant=convert(leading_constant),
            finite_cycle_correction=convert(log_factor),
            denominator_coefficient=convert(eta),
            tree_values=tuple(),
            tree_sqrt_coefficients=tuple(),
            perron_right=labeled_certificate.perron_right,
            perron_left=labeled_certificate.perron_left,
            verified_properties=labeled_certificate.verified_properties
            + (
                "certified radius strictly below one",
                "zero-free retained Adams-substituted cycle determinants",
                "explicit geometric bound for the omitted Pólya tail",
                "positive unlabeled cycle-pole leading constant",
                *(
                    (
                        "exact periodic support and residue-class projection",
                    )
                    if analysis.periodicity != 1
                    else tuple()
                ),
            ),
            dominance_certified=True,
            residue_powers=(
                analysis.residue_powers
                if analysis.periodicity != 1
                else tuple()
            ),
            residue_leading_constants=(
                (convert(leading_constant),)
                + (None,) * (analysis.periodicity - 1)
                if analysis.periodicity != 1
                else tuple()
            ),
        )


def certify_unlabeled_global_marker_face(
    monadic: Any,
    active_types: Sequence[int],
    base_types: Sequence[int],
    remaining_markers: Sequence[int],
    accepted_state: Sequence[int],
    base_analysis: Any,
    *,
    precision_bits: int = 192,
    max_precision_bits: int | None = None,
    tail_terms: int = 128,
) -> C1FAsymptoticCertificate:
    """Certify exact global marker lifts of one tree-critical Pólya face."""

    _require_arb()
    from .asymptotics import (
        _analyze_subcritical_unlabeled_specification,
        _build_labeled_specification,
    )

    saturated_markers = tuple(
        index
        for index, (value, cap) in enumerate(
            zip(accepted_state, monadic.cardinality_plan.caps)
        )
        if value == cap
    )
    base_specification = _build_labeled_specification(
        monadic,
        base_types,
        eventual_start=max(
            base_analysis.global_marker_caps,
            default=0,
        ),
    )
    core_certificate, context = certify_multitype_unlabeled(
        base_specification,
        base_analysis,
        precision_bits=precision_bits,
        max_precision_bits=max_precision_bits,
        tail_terms=tail_terms,
        _return_context=True,
    )
    if (
        base_analysis.power >= -0.25
        and any(accepted_state[index] for index in remaining_markers)
    ):
        raise C1FAsymptoticCertificationError(
            "Exact global markers over an unlabeled cycle-pole core require "
            "cycle-coordinate fugacity differentiation"
        )
    for marker_index in saturated_markers:
        restricted_types = tuple(
            type_id
            for type_id in base_types
            if monadic.cardinality_plan.profile_states[type_id][marker_index] == 0
        )
        if not restricted_types:
            continue
        try:
            restricted_specification = _build_labeled_specification(
                monadic,
                restricted_types,
                eventual_start=base_specification.eventual_start,
            )
            restricted_analysis = _analyze_subcritical_unlabeled_specification(
                restricted_specification,
                tolerance=1e-10,
                max_iterations=120,
                tail_terms=tail_terms,
            )
            if restricted_specification.tree_dimension == 0:
                restricted_certificate = certify_cycle_pole_unlabeled(
                    restricted_specification,
                    restricted_analysis,
                    precision_bits=precision_bits,
                    max_precision_bits=max_precision_bits,
                    tail_terms=tail_terms,
                )
            elif (
                restricted_specification.tree_dimension
                == restricted_specification.cycle_dimension
                == 1
                and restricted_specification.local_caps
            ):
                restricted_certificate = certify_scalar_guarded_unlabeled(
                    restricted_specification,
                    restricted_analysis,
                    precision_bits=precision_bits,
                    max_precision_bits=max_precision_bits,
                    tail_terms=tail_terms,
                )
            else:
                restricted_certificate = certify_multitype_unlabeled(
                    restricted_specification,
                    restricted_analysis,
                    precision_bits=precision_bits,
                    max_precision_bits=max_precision_bits,
                    tail_terms=tail_terms,
                )
        except ValueError:
            continue
        if not (
            Decimal(core_certificate.radius.upper)
            < Decimal(restricted_certificate.radius.lower)
        ):
            raise C1FAsymptoticCertificationError(
                "A saturated global Pólya face is not separated from its "
                "finite-count restriction"
            )
    root_box = context["root_box"]
    system = context["system"]
    unpack = context["unpack"]
    truncation = context["truncation"]
    core_dimension = context["tree_dimension"]
    system_dimension = context["system_dimension"]
    x, core_values = unpack(root_box)
    core_equations = system(root_box)
    base_old_by_profile = tuple(base_types)
    core_coordinate_by_old = {
        base_old_by_profile[profile]: coordinate
        for coordinate, profile in enumerate(
            base_specification.transient_profiles
        )
    }

    def variable_index(level: int, coordinate: int) -> int:
        return 1 + (level - 1) * core_dimension + coordinate

    derivative_intervals: list[C1FRealInterval | None] = [
        None for _ in monadic.cardinality_plan.caps
    ]
    derivative_balls: list[Any | None] = [
        None for _ in monadic.cardinality_plan.caps
    ]
    for marker_index in remaining_markers:
        count = accepted_state[marker_index]
        if count == 0:
            continue
        perturbation_types = tuple(
            type_id
            for type_id in active_types
            if all(
                monadic.cardinality_plan.profile_states[type_id][other] == 0
                for other in remaining_markers
                if other != marker_index
            )
        )
        full_specification = _build_labeled_specification(
            monadic,
            perturbation_types,
            eventual_start=base_specification.eventual_start,
        )
        full_old_by_profile = tuple(perturbation_types)
        full_coordinate_by_old = {
            full_old_by_profile[profile]: coordinate
            for coordinate, profile in enumerate(
                full_specification.transient_profiles
            )
        }
        marked_old_types = tuple(
            old_type
            for old_type in perturbation_types
            if monadic.cardinality_plan.profile_states[old_type][marker_index]
            and old_type in full_coordinate_by_old
        )
        if not marked_old_types:
            raise C1FAsymptoticCertificationError(
                "The global marker has no productive marked tree type"
            )

        def full_distribution(root_profile: int) -> dict[tuple[int, ...], _Jet]:
            grouped: dict[tuple[int, ...], _Jet] = {}
            omitted_states: set[tuple[int, ...]] = set()
            for child_profile in full_specification.transient_profiles:
                if not full_specification.successor[child_profile][root_profile]:
                    continue
                child_old = full_old_by_profile[child_profile]
                contribution = _local_contribution(
                    full_specification,
                    child_profile,
                )
                core_coordinate = core_coordinate_by_old.get(child_old)
                if core_coordinate is not None:
                    for repeat in range(1, truncation + 1):
                        state = tuple(
                            min(cap, repeat * increment)
                            for cap, increment in zip(
                                full_specification.local_caps,
                                contribution,
                            )
                        )
                        grouped[state] = grouped.get(
                            state,
                            _Jet.constant(0, system_dimension),
                        ) + core_values[repeat][core_coordinate] / repeat
                omitted_states.add(
                    tuple(
                        cap if increment else 0
                        for cap, increment in zip(
                            full_specification.local_caps,
                            contribution,
                        )
                    )
                )
            first = x.value ** (truncation + 1)
            total_multiplicity = sum(base_specification.multiplicities)
            if not (1 - 4 * total_multiplicity * first).lower() > 0:
                raise C1FAsymptoticCertificationError(
                    "The global-marker Pólya tail is not geometrically bounded"
                )
            omitted = (
                total_multiplicity
                * first
                / (
                    (truncation + 1)
                    * (1 - x.value)
                    * (1 - 4 * total_multiplicity * first)
                )
            )
            omitted_ball = arb(omitted.upper() / 2, omitted.upper() / 2)
            for state in omitted_states:
                grouped[state] = grouped.get(
                    state,
                    _Jet.constant(0, system_dimension),
                ) + _Jet.constant(omitted_ball, system_dimension)
            zero = tuple(0 for _ in full_specification.local_caps)
            result = {zero: _Jet.constant(1, system_dimension)}
            for state, value in grouped.items():
                result = _distribution_multiply(
                    result,
                    _capped_exponential_factor(
                        value,
                        state,
                        full_specification.local_caps,
                    ),
                    full_specification.local_caps,
                )
            return result

        distributions_by_profile = {
            profile: full_distribution(profile)
            for profile in full_specification.transient_profiles
        }
        marked_tree_derivatives: dict[int, _Jet] = {}
        for marked_old in marked_old_types:
            profile = full_specification.transient_profiles[
                full_coordinate_by_old[marked_old]
            ]
            marked_tree_derivatives[marked_old] = (
                x
                * full_specification.multiplicities[profile]
                * _projected_value(
                    full_specification,
                    profile,
                    distributions_by_profile[profile],
                    system_dimension,
                )
            )

        marker_forcing = []
        for root_coordinate, root_profile in enumerate(
            base_specification.transient_profiles
        ):
            root_old = base_old_by_profile[root_profile]
            full_root_profile = full_specification.transient_profiles[
                full_coordinate_by_old[root_old]
            ]
            forcing = _Jet.constant(0, system_dimension)
            for marked_old, marked_tree in marked_tree_derivatives.items():
                marked_profile = full_specification.transient_profiles[
                    full_coordinate_by_old[marked_old]
                ]
                if not full_specification.successor[marked_profile][full_root_profile]:
                    continue
                derivative = (
                    x
                    * full_specification.multiplicities[full_root_profile]
                    * _projected_value(
                        full_specification,
                        full_root_profile,
                        distributions_by_profile[full_root_profile],
                        system_dimension,
                        _local_contribution(full_specification, marked_profile),
                    )
                )
                forcing = forcing + derivative * marked_tree
            marker_forcing.append(forcing)

        _, core_jacobian_jets = context["tree_phi_and_jacobian"](
            x,
            core_values,
            1,
        )
        marker_jacobian = tuple(
            tuple(
                marker_forcing[row].gradient[variable_index(1, column)]
                for column in range(core_dimension)
            )
            for row in range(core_dimension)
        )
        determinant_direction = _determinant_jet(
            _identity_minus(
                tuple(
                    tuple(
                        _Jet(
                            core_jacobian_jets[row][column].value,
                            (marker_jacobian[row][column],),
                        )
                        for column in range(core_dimension)
                    )
                    for row in range(core_dimension)
                )
            )
        ).gradient[0]
        forcing_values = [arb(0) for _ in range(system_dimension)]
        for row, forcing in enumerate(marker_forcing):
            forcing_values[row] = -forcing.value
        forcing_values[-1] = determinant_direction
        operator = arb_mat(
            system_dimension,
            system_dimension,
            tuple(
                equation.gradient[column]
                for equation in core_equations
                for column in range(system_dimension)
            ),
        )
        try:
            response = operator.solve(
                arb_mat(
                    system_dimension,
                    1,
                    tuple(-value for value in forcing_values),
                )
            )
        except ZeroDivisionError as exc:
            raise C1FAsymptoticCertificationError(
                "The global Pólya marker derivative system is singular"
            ) from exc
        marker_derivative = -response[0, 0] / x.value
        _require_positive(
            "global Pólya marker growth derivative",
            marker_derivative,
        )
        derivative_balls[marker_index] = marker_derivative
        derivative_intervals[marker_index] = _decimal_interval(
            marker_derivative,
            core_certificate.precision_bits,
        )

    with ctx.workprec(core_certificate.precision_bits):
        factor = arb(1)
        for marker_index in remaining_markers:
            count = accepted_state[marker_index]
            if count:
                derivative = derivative_balls[marker_index]
                assert derivative is not None
                factor *= derivative**count / factorial(count)
        leading_constant = (
            arb(core_certificate.leading_constant.enclosure) * factor
        )
        _require_positive(
            "global Pólya face leading constant",
            leading_constant,
        )
        converted_leading = _decimal_interval(
            leading_constant,
            core_certificate.precision_bits,
        )
    return replace(
        core_certificate,
        method=(
            "arb-polya-global-marker-derivative"
            if any(accepted_state[index] for index in remaining_markers)
            else "arb-polya-global-face"
        ),
        regime=f"capped-global-face:{core_certificate.regime}",
        leading_constant=converted_leading,
        marker_growth_derivatives=tuple(derivative_intervals),
        verified_properties=(
            core_certificate.verified_properties
            + (
                ("certified saturated Pólya face radius separation",)
                if saturated_markers
                else tuple()
            )
            + (
                (
                    "vanishing k>=2 first marker derivatives at zero fugacity",
                    "positive implicit global-marker growth derivative",
                )
                if any(accepted_state[index] for index in remaining_markers)
                else tuple()
            )
            + ("nonzero capped-state Pólya face leading constant",)
        ),
    )


def certify_single_global_marker(
    monadic: Any,
    active_types: Sequence[int],
    zero_types: Sequence[int],
    core_analysis: Any,
    marker_count: int,
    *,
    marker_index: int = 0,
    precision_bits: int = 192,
    max_precision_bits: int | None = None,
) -> C1FAsymptoticCertificate:
    """Certify the first fugacity derivative for one unsaturated counter."""

    _require_arb()
    from .asymptotics import (
        _build_labeled_specification,
        analyze_c1f_labeled_asymptotics,
    )

    if core_analysis.certificate is None:
        raise C1FAsymptoticCertificationError(
            "The zero-state core must be interval-certified first"
        )
    if max_precision_bits is None:
        max_precision_bits = precision_bits
    if max_precision_bits < precision_bits:
        raise ValueError("max_precision_bits must be at least precision_bits")

    attempted = []
    current_precision = precision_bits
    last_error: Exception | None = None
    while current_precision <= max_precision_bits:
        attempted.append(current_precision)
        try:
            with ctx.workprec(current_precision):
                core_specification = _build_labeled_specification(
                    monadic,
                    zero_types,
                    eventual_start=core_analysis.eventual_start,
                )
                refined_core = analyze_c1f_labeled_asymptotics(
                    core_specification,
                    certify=True,
                    precision_bits=current_precision,
                )
                core_certificate = refined_core.certificate
                assert core_certificate is not None
                radius = arb(core_certificate.radius.enclosure)
                full_specification = _build_labeled_specification(
                    monadic,
                    active_types,
                    eventual_start=core_analysis.eventual_start,
                )
                if refined_core.regime == "cycle-pole":
                    if refined_core.cycle_pole_order != 1:
                        raise C1FAsymptoticCertificationError(
                            "Marker differentiation over coincident cycle poles "
                            "requires separate branch derivatives"
                        )
                    long_profiles = _long_profiles(full_specification)
                    numerical_transfer = full_specification.cycle_matrix(
                        long_profiles,
                        1.0,
                        tuple(0.0 for _ in range(full_specification.tree_dimension)),
                    )
                    transfer = tuple(
                        tuple(int(round(value)) for value in row)
                        for row in numerical_transfer
                    )
                    if any(
                        abs(numerical_transfer[row][column] - transfer[row][column])
                        > 1e-12
                        for row in range(len(transfer))
                        for column in range(len(transfer))
                    ):
                        raise C1FAsymptoticCertificationError(
                            "The unweighted cycle transfer is not integral"
                        )
                    marked_columns = tuple(
                        monadic.cardinality_plan.profile_states[
                            active_types[profile]
                        ][marker_index]
                        == 1
                        for profile in long_profiles
                    )
                    if not any(marked_columns):
                        raise C1FAsymptoticCertificationError(
                            "The marker does not occur on any long-cycle type"
                        )
                    eigenvalue = 1 / radius
                    size = len(transfer)
                    zero_positions = tuple(
                        index
                        for index, marked in enumerate(marked_columns)
                        if not marked
                    )
                    marked_positions = tuple(
                        index
                        for index, marked in enumerate(marked_columns)
                        if marked
                    )
                    base_transfer = tuple(
                        tuple(
                            transfer[row][column]
                            for column in zero_positions
                        )
                        for row in zero_positions
                    )
                    zero_to_marker = tuple(
                        tuple(
                            transfer[row][column]
                            for column in marked_positions
                        )
                        for row in zero_positions
                    )
                    marker_to_zero = tuple(
                        tuple(
                            transfer[row][column]
                            for column in zero_positions
                        )
                        for row in marked_positions
                    )
                    core_size = len(zero_positions)
                    effective_marker_transfer = tuple(
                        tuple(
                            sum(
                                zero_to_marker[row][middle]
                                * marker_to_zero[middle][column]
                                for middle in range(len(marked_positions))
                            )
                            / eigenvalue
                            for column in range(core_size)
                        )
                        for row in range(core_size)
                    )
                    operator = tuple(
                        tuple(
                            (eigenvalue if row == column else arb(0))
                            - base_transfer[row][column]
                            for column in range(core_size)
                        )
                        for row in range(core_size)
                    )

                    def cofactor(row: int, column: int) -> Any:
                        if core_size == 1:
                            return arb(1)
                        minor = arb_mat(
                            core_size - 1,
                            core_size - 1,
                            tuple(
                                operator[minor_row][minor_column]
                                for minor_row in range(core_size)
                                if minor_row != row
                                for minor_column in range(core_size)
                                if minor_column != column
                            ),
                        ).det()
                        return -minor if (row + column) % 2 else minor

                    characteristic_x_derivative = sum(
                        (cofactor(index, index) for index in range(core_size)),
                        arb(0),
                    )
                    characteristic_marker_derivative = -sum(
                        (
                            cofactor(row, column)
                            * effective_marker_transfer[row][column]
                            for row in range(core_size)
                            for column in range(core_size)
                            if effective_marker_transfer[row][column]
                        ),
                        arb(0),
                    )
                    if characteristic_x_derivative.contains(0):
                        raise C1FAsymptoticCertificationError(
                            "The dominant cycle eigenvalue is not certified simple"
                        )
                    marker_derivative = -characteristic_marker_derivative / (
                        characteristic_x_derivative * eigenvalue
                    )
                    _require_positive("marker growth derivative", marker_derivative)
                    core_leading = arb(core_certificate.leading_constant.enclosure)
                    leading_constant = (
                        core_leading
                        * marker_derivative**marker_count
                        / factorial(marker_count)
                    )
                    _require_positive("marked leading constant", leading_constant)
                    convert = lambda value: _decimal_interval(
                        value,
                        current_precision,
                    )
                    return replace(
                        core_certificate,
                        precision_bits=current_precision,
                        method="arb-cycle-pole-marker-derivative",
                        regime=f"capped-global-unsaturated:{core_certificate.regime}",
                        leading_constant=convert(leading_constant),
                        marker_growth_derivative=convert(marker_derivative),
                        marker_growth_derivatives=(convert(marker_derivative),),
                        verified_properties=core_certificate.verified_properties
                        + (
                            "simple dominant cycle eigenvalue under marker perturbation",
                            "positive exact-matrix growth derivative",
                            "nonzero unsaturated-state leading coefficient",
                        ),
                    )
                if (
                    _long_profiles(full_specification)
                    != full_specification.transient_profiles
                ):
                    raise C1FAsymptoticCertificationError(
                        "Marker differentiation requires every long-cycle type "
                        "to have a transient-tree coordinate"
                    )

                core_value_by_old_type = {
                    zero_types[core_profile]: arb(
                        core_certificate.tree_values[value_index].enclosure
                    )
                    for value_index, core_profile in enumerate(
                        core_specification.transient_profiles
                    )
                }
                tree_values = tuple(
                    core_value_by_old_type.get(active_types[profile], arb(0))
                    for profile in full_specification.transient_profiles
                )
                core_positions = tuple(
                    index
                    for index, profile in enumerate(
                        full_specification.transient_profiles
                    )
                    if active_types[profile] in core_value_by_old_type
                )
                tree_dimension = full_specification.tree_dimension
                jet_dimension = tree_dimension + 2
                x = _Jet.variable(radius, 0, jet_dimension)
                trees = tuple(
                    _Jet.variable(value, 1 + index, jet_dimension)
                    for index, value in enumerate(tree_values)
                )
                fugacity = _Jet.variable(arb(0), jet_dimension - 1, jet_dimension)
                marked = tuple(
                    monadic.cardinality_plan.profile_states[old_type][marker_index] == 1
                    for old_type in active_types
                )
                phi = []
                jacobian = []
                distributions = tuple(
                    _child_distribution(
                        full_specification,
                        root_profile,
                        trees,
                        jet_dimension,
                    )
                    for root_profile in full_specification.transient_profiles
                )
                for row, root_profile in enumerate(
                    full_specification.transient_profiles
                ):
                    root_weight = fugacity if marked[root_profile] else 1
                    base = (
                        root_weight
                        * x
                        * full_specification.multiplicities[root_profile]
                    )
                    phi.append(
                        base
                        * _projected_value(
                            full_specification,
                            root_profile,
                            distributions[row],
                            jet_dimension,
                        )
                    )
                    jacobian.append(
                        tuple(
                            _Jet.constant(0, jet_dimension)
                            if not full_specification.successor[
                                child_profile
                            ][root_profile]
                            else base
                            * _projected_value(
                                full_specification,
                                root_profile,
                                distributions[row],
                                jet_dimension,
                                _local_contribution(
                                    full_specification,
                                    child_profile,
                                ),
                            )
                            for child_profile in full_specification.transient_profiles
                        )
                    )
                equations = tuple(
                    tree - value for tree, value in zip(trees, phi)
                ) + (
                    _determinant_jet(
                        _identity_minus(
                            tuple(
                                tuple(jacobian[row][column] for column in core_positions)
                                for row in core_positions
                            )
                        )
                    ),
                )
                variable_dimension = tree_dimension + 1
                operator = arb_mat(
                    variable_dimension,
                    variable_dimension,
                    tuple(
                        equation.gradient[column]
                        for equation in equations
                        for column in range(variable_dimension)
                    ),
                )
                forcing = arb_mat(
                    variable_dimension,
                    1,
                    tuple(
                        -equation.gradient[-1]
                        for equation in equations
                    ),
                )
                try:
                    derivative = operator.solve(forcing)
                except ZeroDivisionError as exc:
                    raise C1FAsymptoticCertificationError(
                        "The marker-perturbed characteristic derivative is singular"
                    ) from exc
                marker_derivative = -derivative[0, 0] / radius
                _require_positive(
                    "marker growth derivative",
                    marker_derivative,
                )
                core_leading = arb(core_certificate.leading_constant.enclosure)
                leading_constant = (
                    core_leading
                    * marker_derivative**marker_count
                    / factorial(marker_count)
                )
                _require_positive("marked leading constant", leading_constant)
                convert = lambda value: _decimal_interval(value, current_precision)
                return replace(
                    core_certificate,
                    precision_bits=current_precision,
                    method="arb-krawczyk-marker-derivative",
                    regime=f"capped-global-unsaturated:{core_certificate.regime}",
                    leading_constant=convert(leading_constant),
                    marker_growth_derivative=convert(marker_derivative),
                    marker_growth_derivatives=(convert(marker_derivative),),
                    verified_properties=core_certificate.verified_properties
                    + (
                        "unique zero-state extension at marker fugacity zero",
                        "positive implicit derivative of the exponential growth",
                        "nonzero unsaturated-state leading coefficient",
                    ),
                )
        except (C1FAsymptoticCertificationError, ValueError, ZeroDivisionError) as exc:
            last_error = exc
        if current_precision == max_precision_bits:
            break
        current_precision = min(2 * current_precision, max_precision_bits)

    raise C1FAsymptoticCertificationError(
        f"Global-marker certification failed at precisions {attempted}: {last_error}"
    ) from last_error


def certify_global_marker_face(
    monadic: Any,
    active_types: Sequence[int],
    base_types: Sequence[int],
    remaining_markers: Sequence[int],
    accepted_state: Sequence[int],
    base_analysis: Any,
    *,
    precision_bits: int = 192,
    max_precision_bits: int | None = None,
) -> C1FAsymptoticCertificate:
    """Certify all first marker derivatives on one capped-state face."""

    if base_analysis.certificate is None:
        raise C1FAsymptoticCertificationError(
            "The capped-state face core must be certified first"
        )
    from .asymptotics import (
        _build_labeled_specification,
        analyze_c1f_labeled_asymptotics,
    )

    saturated_markers = tuple(
        index
        for index, (value, cap) in enumerate(
            zip(accepted_state, monadic.cardinality_plan.caps)
        )
        if value == cap
    )
    for marker_index in saturated_markers:
        restricted_types = tuple(
            type_id
            for type_id in base_types
            if monadic.cardinality_plan.profile_states[type_id][marker_index] == 0
        )
        if not restricted_types:
            continue
        try:
            restricted = analyze_c1f_labeled_asymptotics(
                _build_labeled_specification(
                    monadic,
                    restricted_types,
                    eventual_start=base_analysis.eventual_start,
                ),
                certify=True,
                precision_bits=precision_bits,
                max_precision_bits=max_precision_bits,
            )
        except ValueError:
            continue
        assert restricted.certificate is not None
        if not (
            Decimal(base_analysis.certificate.radius.upper)
            < Decimal(restricted.certificate.radius.lower)
        ):
            raise C1FAsymptoticCertificationError(
                "A saturated counter is not separated from its finite-count face"
            )
    derivative_intervals: list[C1FRealInterval | None] = [
        None for _ in monadic.cardinality_plan.caps
    ]
    latest = base_analysis.certificate
    for marker_index in remaining_markers:
        count = accepted_state[marker_index]
        if count == 0:
            continue
        perturbation_types = tuple(
            type_id
            for type_id in active_types
            if all(
                monadic.cardinality_plan.profile_states[type_id][other] == 0
                for other in remaining_markers
                if other != marker_index
            )
        )
        latest = certify_single_global_marker(
            monadic,
            perturbation_types,
            base_types,
            base_analysis,
            count,
            marker_index=marker_index,
            precision_bits=precision_bits,
            max_precision_bits=max_precision_bits,
        )
        marker_interval = latest.marker_growth_derivative
        assert marker_interval is not None
        derivative_intervals[marker_index] = marker_interval

    base_certificate = base_analysis.certificate
    with ctx.workprec(latest.precision_bits):
        leading_constant = arb(base_certificate.leading_constant.enclosure)
        marker_factor = arb(1)
        fixed_degree = 0
        for marker_index in remaining_markers:
            count = accepted_state[marker_index]
            if count:
                derivative_interval = derivative_intervals[marker_index]
                assert derivative_interval is not None
                derivative = arb(derivative_interval.enclosure)
                marker_factor *= derivative**count / factorial(count)
                fixed_degree += count
        leading_constant *= marker_factor
        _require_positive("capped-state face leading constant", leading_constant)
        converted_leading = _decimal_interval(
            leading_constant,
            latest.precision_bits,
        )
        residue_constants = tuple(
            None
            if constant is None
            else _decimal_interval(
                arb(constant.enclosure) * marker_factor,
                latest.precision_bits,
            )
            for constant in base_certificate.residue_leading_constants
        )
    return replace(
        base_certificate,
        precision_bits=latest.precision_bits,
        method="arb-krawczyk-marker-face",
        regime=f"capped-global-face:{base_certificate.regime}",
        leading_constant=converted_leading,
        marker_growth_derivative=None,
        marker_growth_derivatives=tuple(derivative_intervals),
        residue_powers=tuple(
            None if power is None else power + fixed_degree
            for power in base_certificate.residue_powers
        ),
        residue_leading_constants=residue_constants,
        residue_polynomial_coefficients=tuple(),
        verified_properties=latest.verified_properties
        + (
            "certified saturated-face decomposition",
            "positive implicit growth derivative for every fixed counter",
            "nonzero capped-state face leading coefficient",
        ),
    )
