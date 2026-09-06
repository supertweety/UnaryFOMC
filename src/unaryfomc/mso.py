"""Exact labeled/unlabeled counting for full monadic MSO with one function.

Compilation recognizes sentences with finite four-sorted composition
algebras. Boolean products and powerset projection give a complete
sentence-specific alternative to constructing every bounded-rank theory.
Coefficient extraction is the finite specification of Part II, Section 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
import re

from ._mso_algebra import (
    Budget, MSOCompilationLimitError, MSOCompilerLimits, combine, complement,
    constant, edge_condition, project, vertex_condition,
)
from .mso_specification import MSOSpecification, evaluate_mso_specification
from .mso_syntax import (
    MSOAtom, MSOBinary, MSOBooleanConstant, MSOEquality, MSOFormula, MSONot,
    MSOQuantifier, MSOTerm, parse_mso_sentence, validate_mso_formula,
)


@dataclass(frozen=True)
class MSOCompilationStatistics:
    compiler: str
    original_nodes: int
    relational_nodes: int
    algebras_built: int
    transitions_evaluated: int
    peak_intermediate_states: int
    branch_state_counts: tuple[tuple[int, int, int, int], ...]


@dataclass
class CompiledMSO:
    formula: MSOFormula
    branches: tuple[MSOSpecification, ...]
    predicates: tuple[str, ...]
    nullary_predicates: tuple[str, ...]
    statistics: MSOCompilationStatistics
    _sequences: dict[bool, tuple[int, ...]] = field(default_factory=dict, repr=False)

    @property
    def specification(self):
        if len(self.branches) != 1:
            raise ValueError("Nullary case splitting produced several specifications; use .branches")
        return self.branches[0]

    def sequence(self, bound: int, *, labeled=True):
        if type(bound) is not int or bound < 0:
            raise ValueError("The domain bound must be a nonnegative integer")
        if type(labeled) is not bool:
            raise ValueError("labeled must be a bool")
        cached = self._sequences.get(labeled, ())
        if len(cached) <= bound:
            counts = [0] * (bound + 1)
            for branch in self.branches:
                for n, value in enumerate(evaluate_mso_specification(branch, bound, labeled=labeled)):
                    counts[n] += value
            cached = self._sequences[labeled] = tuple(counts)
        return cached[:bound + 1]

    def fomc(self, domain_size):
        return self.sequence(domain_size, labeled=True)[-1]

    def ufomc(self, domain_size):
        return self.sequence(domain_size, labeled=False)[-1]

    def has_model(self, *, nonempty=True):
        return any(branch.has_model(nonempty=nonempty) for branch in self.branches)


def _walk(node):
    yield node
    if isinstance(node, MSOBinary):
        yield from _walk(node.left)
        yield from _walk(node.right)
    elif isinstance(node, (MSONot, MSOQuantifier)):
        yield from _walk(node.body)


def _terms(node):
    if isinstance(node, MSOAtom):
        return node.arguments
    if isinstance(node, MSOEquality):
        return node.left, node.right
    return ()


class _Normalizer:
    def __init__(self, formula, edge_predicate, budget):
        self.edge = edge_predicate
        self.budget = budget
        self.occupied = {edge_predicate}
        for node in _walk(formula):
            if isinstance(node, MSOAtom):
                self.occupied.add(node.predicate)
            if isinstance(node, MSOQuantifier):
                self.occupied.add(node.variable)
            self.occupied.update(t.variable for t in _terms(node))
        self.index = 0

    def fresh(self):
        while True:
            name = f"__mso_{self.index}"
            self.index += 1
            if name not in self.occupied:
                self.occupied.add(name)
                return name

    def alpha(self, node, nodes=None, sets=None):
        nodes, sets = nodes or {}, sets or {}
        if isinstance(node, MSOQuantifier):
            name = self.fresh()
            scope = dict(nodes if node.sort == "node" else sets)
            scope[node.variable] = name
            body = self.alpha(node.body, scope if node.sort == "node" else nodes,
                              scope if node.sort == "set" else sets)
            return replace(node, variable=name, body=body)
        if isinstance(node, MSOAtom):
            return MSOAtom(sets.get(node.predicate, node.predicate),
                           tuple(MSOTerm(nodes[t.variable], t.exponent) for t in node.arguments))
        if isinstance(node, MSOEquality):
            return MSOEquality(*(MSOTerm(nodes[t.variable], t.exponent) for t in (node.left, node.right)))
        if isinstance(node, MSOBinary):
            return replace(node, left=self.alpha(node.left, nodes, sets), right=self.alpha(node.right, nodes, sets))
        if isinstance(node, MSONot):
            return MSONot(self.alpha(node.body, nodes, sets))
        return node

    def path(self, start, length, end):
        intermediate = [self.fresh() for _ in range(length - 1)]
        vertices = [start, *intermediate, end]
        atoms = [MSOAtom(self.edge, (MSOTerm(a), MSOTerm(b))) for a, b in zip(vertices, vertices[1:])]
        formula = atoms[0]
        for atom in atoms[1:]:
            formula = MSOBinary("&", formula, atom)
        for variable in reversed(intermediate):
            formula = MSOQuantifier("exists", variable, formula)
        return formula

    def relationalize(self, node):
        if isinstance(node, MSOEquality):
            a, b = node.left, node.right
            if a == b:
                return MSOEquality(MSOTerm(a.variable), MSOTerm(a.variable))
            if a.exponent == 0 and b.exponent:
                return self.path(b.variable, b.exponent, a.variable)
            if b.exponent == 0 and a.exponent:
                return self.path(a.variable, a.exponent, b.variable)
        if isinstance(node, (MSOAtom, MSOEquality)):
            endpoints, clauses, variables = [], [], []
            for t in _terms(node):
                if t.exponent:
                    variable = self.fresh()
                    variables.append(variable)
                    clauses.append(self.path(t.variable, t.exponent, variable))
                    endpoints.append(MSOTerm(variable))
                else:
                    endpoints.append(t)
            formula = MSOAtom(node.predicate, tuple(endpoints)) if isinstance(node, MSOAtom) else MSOEquality(*endpoints)
            for clause in clauses:
                formula = MSOBinary("&", clause, formula)
            for variable in reversed(variables):
                formula = MSOQuantifier("exists", variable, formula)
            return formula
        if isinstance(node, MSOBinary):
            return replace(node, left=self.relationalize(node.left), right=self.relationalize(node.right))
        if isinstance(node, (MSONot, MSOQuantifier)):
            return replace(node, body=self.relationalize(node.body))
        return node


def _substitute_nullaries(node, values):
    if isinstance(node, MSOAtom) and not node.arguments:
        return MSOBooleanConstant(values[node.predicate])
    if isinstance(node, MSOBinary):
        return replace(node, left=_substitute_nullaries(node.left, values), right=_substitute_nullaries(node.right, values))
    if isinstance(node, (MSONot, MSOQuantifier)):
        return replace(node, body=_substitute_nullaries(node.body, values))
    return node


class _FormulaCompiler:
    def __init__(self, vocabulary, edge_predicate, budget):
        self.vocabulary = set(vocabulary)
        self.edge = edge_predicate
        self.budget = budget
        self.cache = {}

    def singleton(self, name):
        vocabulary = tuple(sorted(self.vocabulary | {name}))
        index = vocabulary.index(name)
        return vertex_condition(vocabulary, lambda color: color >> index & 1, self.budget, singleton=True)

    def compile(self, node):
        if node in self.cache:
            return self.cache[node]
        if isinstance(node, MSOBooleanConstant):
            result = constant(tuple(sorted(self.vocabulary)), node.value, self.budget)
        elif isinstance(node, MSOEquality):
            names = {node.left.variable, node.right.variable}
            vocabulary = tuple(sorted(self.vocabulary | names))
            mask = sum(1 << vocabulary.index(name) for name in names)
            result = vertex_condition(vocabulary, lambda color: color & mask == mask, self.budget)
        elif isinstance(node, MSOAtom):
            names = {t.variable for t in node.arguments}
            if node.predicate != self.edge:
                names.add(node.predicate)
            vocabulary = tuple(sorted(self.vocabulary | names))
            if node.predicate == self.edge:
                result = edge_condition(vocabulary, *(vocabulary.index(t.variable) for t in node.arguments), self.budget)
            else:
                mask = sum(1 << vocabulary.index(name) for name in names)
                result = vertex_condition(vocabulary, lambda color: color & mask == mask, self.budget)
        elif isinstance(node, MSONot):
            result = complement(self.compile(node.body))
        elif isinstance(node, MSOBinary):
            result = combine(self.compile(node.left), self.compile(node.right), node.op, self.budget)
        elif isinstance(node, MSOQuantifier):
            body = self.compile(node.body)
            if node.kind == "forall":
                body = complement(body)
            if node.sort == "node":
                body = combine(self.singleton(node.variable), body, "&", self.budget)
            result = project(body, node.variable, self.budget)
            if node.kind == "forall":
                result = complement(result)
        else:
            raise TypeError("Unsupported MSO node")
        self.cache[node] = result
        return result


def compile_mso(formula: str | MSOFormula, *, function_symbol="f", edge_predicate="E",
                predicates=(), limits: MSOCompilerLimits | None = None) -> CompiledMSO:
    """Compile a sentence once, independently of the domain sizes to be counted.

    ``predicates`` may retain additional, unused unary symbols in the counted
    vocabulary. Quantified sets are witnesses, never counted as free symbols.
    Compilation is nonelementary in general; limits fail explicitly.
    """
    if limits is not None and not isinstance(limits, MSOCompilerLimits):
        raise TypeError("limits must be MSOCompilerLimits or None")
    if isinstance(formula, str):
        formula = parse_mso_sentence(formula, function_symbol=function_symbol, edge_predicate=edge_predicate)
    arities = validate_mso_formula(formula, function_symbol=function_symbol, edge_predicate=edge_predicate)
    if isinstance(predicates, str):
        raise TypeError("predicates must be a collection of unary predicate names, not a string")
    vocabulary = set(predicates)
    if any(not isinstance(p, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", p)
           or p in (function_symbol, edge_predicate) or arities.get(p) == 0 for p in vocabulary):
        raise ValueError("Invalid or conflicting additional unary predicate")
    vocabulary |= {p for p, arity in arities.items() if arity == 1}
    nullaries = tuple(sorted(p for p, arity in arities.items() if arity == 0))
    budget = Budget(limits or MSOCompilerLimits())
    original_nodes = sum(1 + sum(t.exponent for t in _terms(node)) for node in _walk(formula))
    budget.check("max_formula_nodes", original_nodes)
    normalizer = _Normalizer(formula, edge_predicate, budget)
    # Fresh element/set markers must also avoid explicitly declared symbols.
    normalizer.occupied.update(vocabulary)
    normalized = normalizer.relationalize(normalizer.alpha(formula))
    relational_nodes = sum(1 for _ in _walk(normalized))
    budget.check("max_formula_nodes", relational_nodes)
    budget.check("max_colors", 1 << len(vocabulary))
    budget.check("max_states", 1 << len(nullaries))
    branches = []
    compiler = _FormulaCompiler(vocabulary, edge_predicate, budget)
    for values in product((False, True), repeat=len(nullaries)):
        substituted = _substitute_nullaries(normalized, dict(zip(nullaries, values)))
        branches.append(compiler.compile(substituted))
    stats = MSOCompilationStatistics(
        "formula-congruence", original_nodes, relational_nodes, budget.algebras,
        budget.transitions, budget.peak_states,
        tuple(tuple(s.state_counts.values()) for s in branches))
    return CompiledMSO(formula, tuple(branches), tuple(sorted(vocabulary)), nullaries, stats)


def compile_mso_fomc(formula, **options):
    return compile_mso(formula, **options)


def compile_mso_ufomc(formula, **options):
    return compile_mso(formula, **options)


def evaluate_mso_fomc(compiled: CompiledMSO, domain_size: int):
    return compiled.fomc(domain_size)


def evaluate_mso_ufomc(compiled: CompiledMSO, domain_size: int):
    return compiled.ufomc(domain_size)


def mso_fomc(formula, domain_size, **options):
    return compile_mso(formula, **options).fomc(domain_size)


def mso_ufomc(formula, domain_size, **options):
    return compile_mso(formula, **options).ufomc(domain_size)
