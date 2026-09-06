"""Syntax for monadic second-order logic over one total unary function.

Uppercase quantified names denote sets; lowercase names denote elements.
Explicit ``exists set X`` and ``exists node x`` binders are also accepted.
The distinguished binary predicate E denotes the graph of f, not a free
relation. All other free predicates must have arity zero or one.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Union


@dataclass(frozen=True)
class MSOTerm:
    variable: str
    exponent: int = 0


@dataclass(frozen=True)
class MSOBooleanConstant:
    value: bool


@dataclass(frozen=True)
class MSOAtom:
    predicate: str
    arguments: tuple[MSOTerm, ...]


@dataclass(frozen=True)
class MSOEquality:
    left: MSOTerm
    right: MSOTerm


@dataclass(frozen=True)
class MSONot:
    body: MSOFormula


@dataclass(frozen=True)
class MSOBinary:
    op: str
    left: MSOFormula
    right: MSOFormula


@dataclass(frozen=True)
class MSOQuantifier:
    kind: str
    variable: str
    body: MSOFormula
    sort: str = "node"


MSOFormula = Union[MSOBooleanConstant, MSOAtom, MSOEquality, MSONot, MSOBinary, MSOQuantifier]
_TOKEN = re.compile(r"\s*(<->|->|!=|==|[()~&|=:,.^]|[A-Za-z_][A-Za-z_0-9]*|[0-9]+)")
_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")


def _identifier(value):
    return isinstance(value, str) and _NAME.fullmatch(value) is not None


class _Parser:
    def __init__(self, text, function_symbol):
        self.tokens = []
        self.position = 0
        self.function_symbol = function_symbol
        offset = 0
        while offset < len(text):
            if not text[offset:].strip():
                break
            match = _TOKEN.match(text, offset)
            if match is None:
                raise ValueError(f"Invalid MSO syntax at character {offset}: {text[offset:offset + 20]!r}")
            self.tokens.append(match.group(1))
            offset = match.end()

    def peek(self):
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def pop(self, expected=None):
        value = self.peek()
        if value is None or (expected is not None and value != expected):
            raise ValueError(f"Expected {expected or 'a token'}, found {value!r}")
        self.position += 1
        return value

    def name(self):
        value = self.pop()
        if not _NAME.fullmatch(value):
            raise ValueError(f"Expected a variable or predicate name, found {value!r}")
        return value

    def term(self):
        name = self.name()
        if name == self.function_symbol and self.peek() in ("(", "^"):
            power = 1
            if self.peek() == "^":
                self.pop()
                token = self.pop()
                if not token.isdigit():
                    raise ValueError("A function exponent must be a nonnegative integer")
                power = int(token)
            self.pop("(")
            argument = self.term()
            self.pop(")")
            return MSOTerm(argument.variable, argument.exponent + power)
        if self.peek() == "(":
            raise ValueError(f"Only the unary function {self.function_symbol!r} is allowed in terms")
        return MSOTerm(name)

    def unary(self):
        token = self.peek()
        if token in ("~", "not"):
            self.pop()
            return MSONot(self.unary())
        if token in ("exists", "forall"):
            kind = self.pop()
            explicit = self.pop() if self.peek() in ("set", "node") else None
            variables = [self.name()]
            while self.peek() == ",":
                self.pop()
                variables.append(self.name())
            if self.peek() not in (":", "."):
                raise ValueError("MSO quantifiers need ':' or '.'; native counting quantifiers belong to the C1 backend")
            self.pop()
            body = self.expression()
            for variable in reversed(variables):
                sort = explicit or ("set" if variable[0].isupper() else "node")
                body = MSOQuantifier(kind, variable, body, sort)
            return body
        if token == "(":
            self.pop()
            body = self.expression()
            self.pop(")")
            return body
        if token in ("TRUE", "FALSE", "true", "false"):
            return MSOBooleanConstant(self.pop().upper() == "TRUE")
        # Predicate applications and function terms are distinguished by f.
        if (token != self.function_symbol and self.position + 1 < len(self.tokens)
                and self.tokens[self.position + 1] == "("):
            name = self.name()
            self.pop("(")
            args = []
            if self.peek() != ")":
                args.append(self.term())
                while self.peek() == ",":
                    self.pop()
                    args.append(self.term())
            self.pop(")")
            return MSOAtom(name, tuple(args))
        term = self.term()
        if self.peek() in ("=", "==", "!="):
            op = self.pop()
            equality = MSOEquality(term, self.term())
            return MSONot(equality) if op == "!=" else equality
        if self.peek() == "in":
            self.pop()
            return MSOAtom(self.name(), (term,))
        if term.exponent:
            raise ValueError("A function term must occur in an atom or equality")
        return MSOAtom(term.variable, ())

    def expression(self, minimum=0):
        left = self.unary()
        priorities = {"<->": 1, "->": 2, "|": 3, "or": 3, "&": 4, "and": 4}
        while self.peek() in priorities and priorities[self.peek()] >= minimum:
            op = self.pop()
            right = self.expression(priorities[op] + (op != "->"))
            left = MSOBinary({"and": "&", "or": "|"}.get(op, op), left, right)
        return left


def parse_mso_formula(text: str, *, function_symbol="f", edge_predicate="E") -> MSOFormula:
    if not isinstance(text, str):
        raise TypeError("An MSO source formula must be a string")
    if not _identifier(function_symbol) or not _identifier(edge_predicate):
        raise ValueError("Function and edge symbols must be identifiers")
    if function_symbol == edge_predicate:
        raise ValueError("Function and edge symbols must differ")
    parser = _Parser(text, function_symbol)
    formula = parser.expression()
    if parser.peek() is not None:
        raise ValueError(f"Unexpected token after the MSO formula: {parser.peek()!r}")
    validate_mso_formula(formula, require_closed=False, function_symbol=function_symbol,
                         edge_predicate=edge_predicate)
    return formula


def parse_mso_sentence(text: str, *, function_symbol="f", edge_predicate="E") -> MSOFormula:
    formula = parse_mso_formula(text, function_symbol=function_symbol, edge_predicate=edge_predicate)
    validate_mso_formula(formula, function_symbol=function_symbol, edge_predicate=edge_predicate)
    return formula


def validate_mso_formula(formula, *, require_closed=True, function_symbol="f", edge_predicate="E"):
    if (not _identifier(function_symbol) or not _identifier(edge_predicate)
            or function_symbol == edge_predicate):
        raise ValueError("Function and edge symbols must be distinct identifiers")
    arities = {}

    def term(value, nodes):
        if (not isinstance(value, MSOTerm) or not _identifier(value.variable)
                or type(value.exponent) is not int or value.exponent < 0):
            raise ValueError("Invalid MSO term")
        if require_closed and value.variable not in nodes:
            raise ValueError(f"Free element variable {value.variable!r}")

    def visit(node, nodes, sets):
        if isinstance(node, MSOBooleanConstant):
            if type(node.value) is not bool:
                raise ValueError("A Boolean constant needs a bool")
        elif isinstance(node, MSOAtom):
            if not _identifier(node.predicate) or node.predicate == function_symbol:
                raise ValueError("Invalid or conflicting predicate name")
            for arg in node.arguments:
                term(arg, nodes)
            arity = len(node.arguments)
            if node.predicate in sets:
                if arity != 1:
                    raise ValueError("A quantified set must have exactly one argument")
            elif node.predicate == edge_predicate:
                if arity != 2:
                    raise ValueError("The distinguished graph predicate needs two arguments")
            else:
                if arity > 1:
                    raise ValueError("MSO counting only permits unary/nullary free relations besides the graph of f")
                if node.predicate in arities and arities[node.predicate] != arity:
                    raise ValueError("Inconsistent predicate arity")
                arities[node.predicate] = arity
        elif isinstance(node, MSOEquality):
            term(node.left, nodes)
            term(node.right, nodes)
        elif isinstance(node, MSONot):
            visit(node.body, nodes, sets)
        elif isinstance(node, MSOBinary):
            if node.op not in ("&", "|", "->", "<->"):
                raise ValueError("Invalid MSO connective")
            visit(node.left, nodes, sets)
            visit(node.right, nodes, sets)
        elif isinstance(node, MSOQuantifier):
            if (node.kind not in ("exists", "forall") or node.sort not in ("set", "node")
                    or not _identifier(node.variable)
                    or node.variable in (function_symbol, edge_predicate)):
                raise ValueError("Invalid or conflicting MSO binder")
            visit(node.body, nodes | {node.variable} if node.sort == "node" else nodes,
                  sets | {node.variable} if node.sort == "set" else sets)
        else:
            raise TypeError(f"Not an MSO formula: {type(node).__name__}")

    visit(formula, set(), set())
    return arities


def mso_predicate_arities(formula, *, function_symbol="f", edge_predicate="E"):
    return validate_mso_formula(formula, require_closed=False, function_symbol=function_symbol,
                                edge_predicate=edge_predicate)


def render_mso_term(term, *, function_symbol="f"):
    return (function_symbol + "(") * term.exponent + term.variable + ")" * term.exponent


def render_mso_formula(formula, *, function_symbol="f"):
    def render(node):
        if isinstance(node, MSOBooleanConstant):
            return "TRUE" if node.value else "FALSE"
        if isinstance(node, MSOAtom):
            return node.predicate + "(" + ",".join(render_mso_term(t, function_symbol=function_symbol)
                                                  for t in node.arguments) + ")"
        if isinstance(node, MSOEquality):
            return f"({render_mso_term(node.left, function_symbol=function_symbol)}={render_mso_term(node.right, function_symbol=function_symbol)})"
        if isinstance(node, MSONot):
            return f"~({render(node.body)})"
        if isinstance(node, MSOBinary):
            return f"({render(node.left)} {node.op} {render(node.right)})"
        if isinstance(node, MSOQuantifier):
            return f"({node.kind} {node.sort} {node.variable}: {render(node.body)})"
        raise TypeError("Not an MSO formula")

    return render(formula)


def evaluate_mso_formula(formula, function, predicates=None, *, edge_predicate="E"):
    """Direct finite-model semantics, intended for independent small tests.

    Unary predicates are sets of vertices; nullary predicates are bools.
    This is model checking, not the polynomial-data-complexity counter.
    """
    predicates = dict(predicates or {})
    function = tuple(function)
    n = len(function)
    if any(type(v) is not int or not 0 <= v < n for v in function):
        raise ValueError("Expected a total function on range(n)")

    def term(t, env):
        v = env[t.variable]
        for _ in range(t.exponent):
            v = function[v]
        return v

    def visit(node, env, sets):
        if isinstance(node, MSOBooleanConstant):
            return node.value
        if isinstance(node, MSOEquality):
            return term(node.left, env) == term(node.right, env)
        if isinstance(node, MSOAtom):
            args = tuple(term(t, env) for t in node.arguments)
            if node.predicate in sets:
                return args[0] in sets[node.predicate]
            if node.predicate == edge_predicate:
                return function[args[0]] == args[1]
            return bool(predicates[node.predicate]) if not args else args[0] in predicates[node.predicate]
        if isinstance(node, MSONot):
            return not visit(node.body, env, sets)
        if isinstance(node, MSOBinary):
            a, b = visit(node.left, env, sets), visit(node.right, env, sets)
            return {"&": a and b, "|": a or b, "->": not a or b, "<->": a == b}[node.op]
        if isinstance(node, MSOQuantifier):
            if node.sort == "node":
                values = (visit(node.body, dict(env, **{node.variable: v}), sets) for v in range(n))
            else:
                values = (visit(node.body, env, dict(sets, **{node.variable: {i for i in range(n) if mask >> i & 1}}))
                          for mask in range(1 << n))
            return all(values) if node.kind == "forall" else any(values)
        raise TypeError("Not an MSO formula")

    return visit(formula, {}, {})
