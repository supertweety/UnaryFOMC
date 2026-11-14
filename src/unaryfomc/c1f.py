from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from math import factorial
import re
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .arithmetic import mpq, mpz

if TYPE_CHECKING:
    from .asymptotics import C1FLabeledAsymptotics
    from .specification import C1FLabeledSpecification


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class C1FTerm:
    variable: str
    exponent: int = 0

    def shifted(self, amount: int) -> "C1FTerm":
        return C1FTerm(self.variable, self.exponent + amount)


@dataclass(frozen=True)
class C1FFormula:
    pass


@dataclass(frozen=True)
class C1FBooleanConstant(C1FFormula):
    value: bool


@dataclass(frozen=True)
class C1FAtom(C1FFormula):
    predicate: str
    arguments: Tuple[C1FTerm, ...]


@dataclass(frozen=True)
class C1FEquality(C1FFormula):
    left: C1FTerm
    right: C1FTerm


@dataclass(frozen=True)
class C1FNot(C1FFormula):
    body: C1FFormula


@dataclass(frozen=True)
class C1FBinary(C1FFormula):
    op: str
    left: C1FFormula
    right: C1FFormula


@dataclass(frozen=True)
class C1FQuantifier(C1FFormula):
    kind: str
    variable: str
    body: C1FFormula
    comparator: Optional[str] = None
    count: Optional[int] = None


@dataclass(frozen=True)
class C1FPredecessorCount(C1FFormula):
    anchor: str
    witness: str
    body: C1FFormula
    comparator: str
    count: int


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    position: int


_TOKEN_RE = re.compile(
    r"""
    (?P<SPACE>\s+)
    | (?P<IFF><->)
    | (?P<IMPLIES>->)
    | (?P<NEQ>!=)
    | (?P<LEQ><=)
    | (?P<GEQ>>=)
    | (?P<LPAREN>\()
    | (?P<RPAREN>\))
    | (?P<COMMA>,)
    | (?P<COLON>:)
    | (?P<DOT>\.)
    | (?P<NOT>~)
    | (?P<AND>&)
    | (?P<OR>\|)
    | (?P<EQ>=)
    | (?P<LT><)
    | (?P<GT>>)
    | (?P<INT>\d+)
    | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)

_KEYWORDS = {
    "forall": "FORALL",
    "exists": "EXISTS",
    "true": "TRUE",
    "false": "FALSE",
}
_RESERVED_IDENTIFIERS = set(_KEYWORDS)
_COUNT_COMPARATORS = {"=", "==", "<", "<=", ">", ">="}


def _validate_identifier(name: str, label: str) -> None:
    if _IDENTIFIER_RE.fullmatch(name) is None:
        raise ValueError(f"{label} {name!r} is not a valid identifier")
    if name.lower() in _RESERVED_IDENTIFIERS:
        raise ValueError(f"{label} {name!r} is reserved")


def tokenize_c1f(formula: str) -> Tuple[_Token, ...]:
    tokens: List[_Token] = []
    pos = 0
    while pos < len(formula):
        match = _TOKEN_RE.match(formula, pos)
        if match is None:
            raise ValueError(f"Unexpected character at position {pos}: {formula[pos]!r}")
        kind = match.lastgroup
        assert kind is not None
        text = match.group()
        pos = match.end()
        if kind == "SPACE":
            continue
        if kind == "IDENT":
            kind = _KEYWORDS.get(text.lower(), "IDENT")
        tokens.append(_Token(kind, text, match.start()))
    return tuple(tokens)


class _C1FParser:
    def __init__(self, tokens: Sequence[_Token], function_symbol: str):
        self._tokens = list(tokens)
        self._index = 0
        self._function_symbol = function_symbol

    def parse(self) -> C1FFormula:
        node = self._parse_iff()
        if self._index != len(self._tokens):
            token = self._peek()
            raise ValueError(f"Unexpected token {token.text!r} at position {token.position}")
        return node

    def _peek(self) -> _Token:
        if self._index >= len(self._tokens):
            return _Token("EOF", "", -1)
        return self._tokens[self._index]

    def _peek_ahead(self, offset: int) -> _Token:
        index = self._index + offset
        if index >= len(self._tokens):
            return _Token("EOF", "", -1)
        return self._tokens[index]

    def _consume(self, expected_kind: str | None = None) -> _Token:
        token = self._peek()
        if token.kind == "EOF":
            raise ValueError("Unexpected end of input")
        if expected_kind is not None and token.kind != expected_kind:
            raise ValueError(
                f"Expected {expected_kind}, found {token.text!r} at position {token.position}"
            )
        self._index += 1
        return token

    def _match(self, *kinds: str) -> Optional[_Token]:
        token = self._peek()
        if token.kind in kinds:
            self._index += 1
            return token
        return None

    def _parse_iff(self) -> C1FFormula:
        left = self._parse_implies()
        while self._match("IFF") is not None:
            right = self._parse_implies()
            left = C1FBinary("<->", left, right)
        return left

    def _parse_implies(self) -> C1FFormula:
        # A chain of implications groups to the right, as in the MSO parser:
        # A -> B -> C is A -> (B -> C).
        left = self._parse_or()
        if self._match("IMPLIES") is not None:
            return C1FBinary("->", left, self._parse_implies())
        return left

    def _parse_or(self) -> C1FFormula:
        left = self._parse_and()
        while self._match("OR") is not None:
            right = self._parse_and()
            left = C1FBinary("|", left, right)
        return left

    def _parse_and(self) -> C1FFormula:
        left = self._parse_unary_formula()
        while self._match("AND") is not None:
            right = self._parse_unary_formula()
            left = C1FBinary("&", left, right)
        return left

    def _parse_unary_formula(self) -> C1FFormula:
        if self._match("NOT") is not None:
            return C1FNot(self._parse_unary_formula())
        token = self._peek()
        if token.kind in {"FORALL", "EXISTS"}:
            return self._parse_quantified()
        if self._match("LPAREN") is not None:
            node = self._parse_iff()
            self._consume("RPAREN")
            return node
        if self._match("TRUE") is not None:
            return C1FBooleanConstant(True)
        if self._match("FALSE") is not None:
            return C1FBooleanConstant(False)
        return self._parse_atom_or_equality()

    def _parse_quantified(self) -> C1FFormula:
        binders: List[Tuple[str, str, Optional[str], Optional[int]]] = []
        while True:
            token = self._peek()
            if token.kind == "FORALL":
                self._consume("FORALL")
                variable = self._consume("IDENT").text
                binders.append(("forall", variable, None, None))
            elif token.kind == "EXISTS":
                self._consume("EXISTS")
                comparator, count = self._parse_optional_count_bound()
                variable = self._consume("IDENT").text
                binders.append(("exists", variable, comparator, count))
            else:
                break
            if self._peek().kind not in {"FORALL", "EXISTS"}:
                break
        self._match("COLON", "DOT")
        body = self._parse_iff()
        for kind, variable, comparator, count in reversed(binders):
            body = C1FQuantifier(kind, variable, body, comparator, count)
        return body

    def _parse_optional_count_bound(self) -> Tuple[Optional[str], Optional[int]]:
        token = self._peek()
        comparator: Optional[str]
        if token.kind == "EQ":
            self._consume("EQ")
            comparator = "=="
        elif token.kind == "LEQ":
            self._consume("LEQ")
            comparator = "<="
        elif token.kind == "GEQ":
            self._consume("GEQ")
            comparator = ">="
        elif token.kind == "LT":
            self._consume("LT")
            comparator = "<"
        elif token.kind == "GT":
            self._consume("GT")
            comparator = ">"
        else:
            return None, None
        return comparator, int(self._consume("INT").text)

    def _parse_atom_or_equality(self) -> C1FFormula:
        if self._starts_term():
            left = self._parse_term()
            op_token = self._match("EQ", "NEQ")
            if op_token is None:
                raise ValueError("A standalone term is not a formula; expected '=' or '!='")
            right = self._parse_term()
            eq = C1FEquality(left, right)
            return C1FNot(eq) if op_token.kind == "NEQ" else eq

        predicate = self._consume("IDENT").text
        if predicate == self._function_symbol:
            raise ValueError(
                f"Predicate name {predicate!r} conflicts with the unary function symbol"
            )
        if self._match("LPAREN") is None:
            return C1FAtom(predicate, tuple())

        args: List[C1FTerm] = []
        if self._peek().kind != "RPAREN":
            args.append(self._parse_term())
            while self._match("COMMA") is not None:
                args.append(self._parse_term())
        self._consume("RPAREN")
        return C1FAtom(predicate, tuple(args))

    def _starts_term(self) -> bool:
        token = self._peek()
        if token.kind != "IDENT":
            return False
        if token.text == self._function_symbol:
            return True
        return self._peek_ahead(1).kind in {"EQ", "NEQ"}

    def _parse_term(self) -> C1FTerm:
        token = self._consume("IDENT")
        if token.text == self._function_symbol:
            self._consume("LPAREN")
            inner = self._parse_term()
            self._consume("RPAREN")
            return inner.shifted(1)
        return C1FTerm(token.text, 0)


def _flatten_conjuncts(node: C1FFormula) -> Tuple[C1FFormula, ...]:
    if isinstance(node, C1FBinary) and node.op == "&":
        return _flatten_conjuncts(node.left) + _flatten_conjuncts(node.right)
    return (node,)


def _conjoin_formulas(nodes: Sequence[C1FFormula]) -> C1FFormula:
    if not nodes:
        return C1FBooleanConstant(True)
    result = nodes[0]
    for node in nodes[1:]:
        result = C1FBinary("&", result, node)
    return result


def _guard_anchor(
    node: C1FFormula,
    witness: str,
) -> Optional[str]:
    if not isinstance(node, C1FEquality):
        return None
    for guarded, anchor in ((node.left, node.right), (node.right, node.left)):
        if (
            guarded.variable == witness
            and guarded.exponent == 1
            and anchor.variable != witness
            and anchor.exponent == 0
        ):
            return anchor.variable
    return None


def _normalize_guarded_predecessor_counts(node: C1FFormula) -> C1FFormula:
    if isinstance(node, (C1FBooleanConstant, C1FAtom, C1FEquality)):
        return node
    if isinstance(node, C1FNot):
        return C1FNot(_normalize_guarded_predecessor_counts(node.body))
    if isinstance(node, C1FBinary):
        return C1FBinary(
            node.op,
            _normalize_guarded_predecessor_counts(node.left),
            _normalize_guarded_predecessor_counts(node.right),
        )
    if isinstance(node, C1FPredecessorCount):
        return C1FPredecessorCount(
            node.anchor,
            node.witness,
            _normalize_guarded_predecessor_counts(node.body),
            node.comparator,
            node.count,
        )
    if not isinstance(node, C1FQuantifier):
        raise TypeError(f"Unsupported C1F node: {type(node)!r}")

    body = _normalize_guarded_predecessor_counts(node.body)
    normalized = C1FQuantifier(
        node.kind,
        node.variable,
        body,
        node.comparator,
        node.count,
    )
    if node.kind != "exists":
        return normalized

    conjuncts = _flatten_conjuncts(body)
    candidates: List[Tuple[str, C1FFormula]] = []
    for index, conjunct in enumerate(conjuncts):
        anchor = _guard_anchor(conjunct, node.variable)
        if anchor is None:
            continue
        filler = _conjoin_formulas(conjuncts[:index] + conjuncts[index + 1 :])
        if c1f_free_variables(filler) <= {node.variable}:
            candidates.append((anchor, filler))
    if not candidates:
        return normalized
    if len(candidates) > 1:
        raise ValueError(
            "Ambiguous guarded predecessor count: expected one guard f(witness) = anchor"
        )
    anchor, filler = candidates[0]
    comparator = node.comparator if node.comparator is not None else ">="
    count = node.count if node.count is not None else 1
    return C1FPredecessorCount(
        anchor=anchor,
        witness=node.variable,
        body=filler,
        comparator=comparator,
        count=count,
    )


def parse_c1f_formula(formula: str, *, function_symbol: str = "f") -> C1FFormula:
    _validate_identifier(function_symbol, "Function symbol")
    parsed = _C1FParser(tokenize_c1f(formula), function_symbol).parse()
    return _normalize_guarded_predecessor_counts(parsed)


def parse_c1f_sentence(formula: str, *, function_symbol: str = "f") -> C1FFormula:
    parsed = parse_c1f_formula(formula, function_symbol=function_symbol)
    validate_c1f_formula(parsed, require_closed=True, function_symbol=function_symbol)
    return parsed


def render_c1f_term(term: C1FTerm, *, function_symbol: str = "f") -> str:
    rendered = term.variable
    for _ in range(term.exponent):
        rendered = f"{function_symbol}({rendered})"
    return rendered


def render_c1f_formula(node: C1FFormula, *, function_symbol: str = "f") -> str:
    if isinstance(node, C1FBooleanConstant):
        return "TRUE" if node.value else "FALSE"
    if isinstance(node, C1FAtom):
        if not node.arguments:
            return node.predicate
        args = ",".join(render_c1f_term(arg, function_symbol=function_symbol) for arg in node.arguments)
        return f"{node.predicate}({args})"
    if isinstance(node, C1FEquality):
        return (
            f"{render_c1f_term(node.left, function_symbol=function_symbol)} = "
            f"{render_c1f_term(node.right, function_symbol=function_symbol)}"
        )
    if isinstance(node, C1FNot):
        if isinstance(node.body, C1FEquality):
            return (
                f"{render_c1f_term(node.body.left, function_symbol=function_symbol)} != "
                f"{render_c1f_term(node.body.right, function_symbol=function_symbol)}"
            )
        return f"~({render_c1f_formula(node.body, function_symbol=function_symbol)})"
    if isinstance(node, C1FBinary):
        left = render_c1f_formula(node.left, function_symbol=function_symbol)
        if isinstance(node.left, (C1FQuantifier, C1FPredecessorCount)):
            # A quantifier extends as far to the right as possible.  A
            # quantified left operand is therefore closed by parentheses;
            # otherwise the connective would be read as part of its body.
            left = f"({left})"
        return (
            f"({left} "
            f"{node.op} {render_c1f_formula(node.right, function_symbol=function_symbol)})"
        )
    if isinstance(node, C1FQuantifier):
        if node.kind == "forall":
            prefix = f"forall {node.variable}"
        elif node.comparator is None:
            prefix = f"exists {node.variable}"
        else:
            comparator = "=" if node.comparator == "==" else node.comparator
            prefix = f"exists{comparator}{node.count} {node.variable}"
        return f"{prefix}: {render_c1f_formula(node.body, function_symbol=function_symbol)}"
    if isinstance(node, C1FPredecessorCount):
        comparator = "=" if node.comparator == "==" else node.comparator
        guard = (
            f"{function_symbol}({node.witness}) = {node.anchor} & "
            f"{render_c1f_formula(node.body, function_symbol=function_symbol)}"
        )
        return f"exists{comparator}{node.count} {node.witness}: ({guard})"
    raise TypeError(f"Unsupported C1F node: {type(node)!r}")


def c1f_free_variables(node: C1FFormula) -> frozenset[str]:
    if isinstance(node, C1FBooleanConstant):
        return frozenset()
    if isinstance(node, C1FAtom):
        return frozenset(term.variable for term in node.arguments)
    if isinstance(node, C1FEquality):
        return frozenset((node.left.variable, node.right.variable))
    if isinstance(node, C1FNot):
        return c1f_free_variables(node.body)
    if isinstance(node, C1FBinary):
        return c1f_free_variables(node.left) | c1f_free_variables(node.right)
    if isinstance(node, C1FQuantifier):
        return c1f_free_variables(node.body) - {node.variable}
    if isinstance(node, C1FPredecessorCount):
        return (c1f_free_variables(node.body) - {node.witness}) | {node.anchor}
    raise TypeError(f"Unsupported C1F node: {type(node)!r}")


def c1f_predicate_arities(node: C1FFormula) -> Dict[str, int]:
    arities: Dict[str, int] = {}

    def add(predicate: str, arity: int) -> None:
        previous = arities.get(predicate)
        if previous is not None and previous != arity:
            raise ValueError(
                f"Predicate {predicate!r} appears with conflicting arities {previous} and {arity}"
            )
        arities[predicate] = arity

    def visit(current: C1FFormula) -> None:
        if isinstance(current, C1FAtom):
            add(current.predicate, len(current.arguments))
            return
        if isinstance(current, C1FEquality):
            return
        if isinstance(current, C1FNot):
            visit(current.body)
            return
        if isinstance(current, C1FBinary):
            visit(current.left)
            visit(current.right)
            return
        if isinstance(current, C1FQuantifier):
            visit(current.body)
            return
        if isinstance(current, C1FPredecessorCount):
            visit(current.body)
            return
        if isinstance(current, C1FBooleanConstant):
            return
        raise TypeError(f"Unsupported C1F node: {type(current)!r}")

    visit(node)
    return arities


def _c1f_variables(node: C1FFormula) -> frozenset[str]:
    variables: set[str] = set()

    def add_term(term: C1FTerm) -> None:
        variables.add(term.variable)

    def visit(current: C1FFormula) -> None:
        if isinstance(current, C1FAtom):
            for term in current.arguments:
                add_term(term)
            return
        if isinstance(current, C1FEquality):
            add_term(current.left)
            add_term(current.right)
            return
        if isinstance(current, C1FNot):
            visit(current.body)
            return
        if isinstance(current, C1FBinary):
            visit(current.left)
            visit(current.right)
            return
        if isinstance(current, C1FQuantifier):
            variables.add(current.variable)
            visit(current.body)
            return
        if isinstance(current, C1FPredecessorCount):
            variables.add(current.anchor)
            variables.add(current.witness)
            visit(current.body)
            return
        if isinstance(current, C1FBooleanConstant):
            return
        raise TypeError(f"Unsupported C1F node: {type(current)!r}")

    visit(node)
    return frozenset(variables)


def validate_c1f_formula(
    node: C1FFormula,
    *,
    require_closed: bool = False,
    function_symbol: str = "f",
) -> None:
    _validate_identifier(function_symbol, "Function symbol")
    arities = c1f_predicate_arities(node)
    if function_symbol in arities:
        raise ValueError(
            f"Predicate name {function_symbol!r} conflicts with the unary function symbol"
        )
    for predicate in arities:
        _validate_identifier(predicate, "Predicate name")
    def visit(current: C1FFormula, *, guarded_filler: bool = False) -> None:
        if not isinstance(current, C1FPredecessorCount):
            free = c1f_free_variables(current)
            if len(free) > 1:
                raise ValueError(
                    "C1F formulas may keep at most one free variable outside a "
                    f"guarded predecessor count; found {sorted(free)}"
                )
        if isinstance(current, (C1FBooleanConstant, C1FAtom, C1FEquality)):
            return
        if isinstance(current, C1FNot):
            visit(current.body, guarded_filler=guarded_filler)
            return
        if isinstance(current, C1FBinary):
            if current.op not in {"&", "|", "->", "<->"}:
                raise ValueError(f"Unsupported boolean connective {current.op!r}")
            visit(current.left, guarded_filler=guarded_filler)
            visit(current.right, guarded_filler=guarded_filler)
            return
        if isinstance(current, C1FQuantifier):
            if guarded_filler:
                raise ValueError(
                    "The body of a guarded predecessor count cannot contain "
                    "ordinary quantifiers; use a nested guarded predecessor count"
                )
            if current.kind not in {"forall", "exists"}:
                raise ValueError(f"Unsupported quantifier kind {current.kind!r}")
            _validate_identifier(current.variable, "Variable name")
            if current.kind == "forall":
                if current.comparator is not None or current.count is not None:
                    raise ValueError("Universal quantifiers cannot carry counting bounds")
            else:
                if current.comparator is not None:
                    if current.comparator not in _COUNT_COMPARATORS:
                        raise ValueError(f"Unsupported counting comparator {current.comparator!r}")
                    if current.count is None or current.count < 0:
                        raise ValueError("Counting quantifier bounds must be non-negative")
                elif current.count is not None:
                    raise ValueError("Existential quantifier count must come with a comparator")
            visit(current.body)
            return
        if isinstance(current, C1FPredecessorCount):
            _validate_identifier(current.anchor, "Anchor variable")
            _validate_identifier(current.witness, "Witness variable")
            if current.anchor == current.witness:
                raise ValueError(
                    "Guarded predecessor count requires distinct anchor and witness variables"
                )
            if current.comparator not in _COUNT_COMPARATORS:
                raise ValueError(
                    f"Unsupported counting comparator {current.comparator!r}"
                )
            if current.count < 0:
                raise ValueError("Counting quantifier bounds must be non-negative")
            filler_free = c1f_free_variables(current.body)
            if not filler_free <= {current.witness}:
                raise ValueError(
                    "The body of a guarded predecessor count may mention only its "
                    f"witness variable {current.witness!r}; found {sorted(filler_free)}"
                )
            visit(current.body, guarded_filler=True)
            return
        raise TypeError(f"Unsupported C1F node: {type(current)!r}")

    visit(node)
    if require_closed:
        free = c1f_free_variables(node)
        if free:
            raise ValueError(f"C1F sentence must be closed; found free variables {sorted(free)}")


def _max_term_depth(node: C1FFormula) -> int:
    depth = 0

    def visit_term(term: C1FTerm) -> None:
        nonlocal depth
        depth = max(depth, term.exponent)

    def visit(current: C1FFormula) -> None:
        if isinstance(current, C1FAtom):
            for term in current.arguments:
                visit_term(term)
            return
        if isinstance(current, C1FEquality):
            visit_term(current.left)
            visit_term(current.right)
            return
        if isinstance(current, C1FNot):
            visit(current.body)
            return
        if isinstance(current, C1FBinary):
            visit(current.left)
            visit(current.right)
            return
        if isinstance(current, C1FQuantifier):
            visit(current.body)
            return
        if isinstance(current, C1FPredecessorCount):
            visit(current.body)
            return
        if isinstance(current, C1FBooleanConstant):
            return
        raise TypeError(f"Unsupported C1F node: {type(current)!r}")

    visit(node)
    return depth


def _simplify(node: C1FFormula) -> C1FFormula:
    if isinstance(
        node,
        (C1FBooleanConstant, C1FAtom, C1FEquality, C1FPredecessorCount),
    ):
        return node
    if isinstance(node, C1FNot):
        body = _simplify(node.body)
        if isinstance(body, C1FBooleanConstant):
            return C1FBooleanConstant(not body.value)
        if isinstance(body, C1FNot):
            return _simplify(body.body)
        return C1FNot(body)
    if isinstance(node, C1FBinary):
        left = _simplify(node.left)
        right = _simplify(node.right)
        if node.op == "&":
            if isinstance(left, C1FBooleanConstant):
                return right if left.value else left
            if isinstance(right, C1FBooleanConstant):
                return left if right.value else right
        if node.op == "|":
            if isinstance(left, C1FBooleanConstant):
                return left if left.value else right
            if isinstance(right, C1FBooleanConstant):
                return right if right.value else left
        if node.op == "->":
            if isinstance(left, C1FBooleanConstant):
                return right if left.value else C1FBooleanConstant(True)
            if isinstance(right, C1FBooleanConstant):
                return C1FBooleanConstant(True) if right.value else _simplify(C1FNot(left))
        if node.op == "<->":
            if isinstance(left, C1FBooleanConstant):
                return right if left.value else _simplify(C1FNot(right))
            if isinstance(right, C1FBooleanConstant):
                return left if right.value else _simplify(C1FNot(left))
        return C1FBinary(node.op, left, right)
    if isinstance(node, C1FQuantifier):
        return C1FQuantifier(
            node.kind,
            node.variable,
            _simplify(node.body),
            node.comparator,
            node.count,
        )
    if isinstance(node, C1FPredecessorCount):
        return C1FPredecessorCount(
            node.anchor,
            node.witness,
            _simplify(node.body),
            node.comparator,
            node.count,
        )
    raise TypeError(f"Unsupported C1F node: {type(node)!r}")


def _replace_nullary(node: C1FFormula, assignment: Mapping[str, bool]) -> C1FFormula:
    if isinstance(node, C1FAtom):
        if not node.arguments and node.predicate in assignment:
            return C1FBooleanConstant(assignment[node.predicate])
        return node
    if isinstance(node, C1FBooleanConstant):
        return node
    if isinstance(node, C1FEquality):
        return node
    if isinstance(node, C1FNot):
        return C1FNot(_replace_nullary(node.body, assignment))
    if isinstance(node, C1FBinary):
        return C1FBinary(
            node.op,
            _replace_nullary(node.left, assignment),
            _replace_nullary(node.right, assignment),
        )
    if isinstance(node, C1FQuantifier):
        return C1FQuantifier(
            node.kind,
            node.variable,
            _replace_nullary(node.body, assignment),
            node.comparator,
            node.count,
        )
    if isinstance(node, C1FPredecessorCount):
        return C1FPredecessorCount(
            node.anchor,
            node.witness,
            _replace_nullary(node.body, assignment),
            node.comparator,
            node.count,
        )
    raise TypeError(f"Unsupported C1F node: {type(node)!r}")


def _and_all(clauses: Sequence[C1FFormula]) -> C1FFormula:
    if not clauses:
        return C1FBooleanConstant(True)
    current = clauses[0]
    for clause in clauses[1:]:
        current = C1FBinary("&", current, clause)
    return _simplify(current)


def _or_all(clauses: Sequence[C1FFormula]) -> C1FFormula:
    if not clauses:
        return C1FBooleanConstant(False)
    current = clauses[0]
    for clause in clauses[1:]:
        current = C1FBinary("|", current, clause)
    return _simplify(current)


class _PredicateNamer:
    def __init__(self, occupied: Iterable[str]):
        self._occupied = set(occupied)
        self._counter = 0

    def fresh(self, prefix: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_]", "_", prefix)
        if not sanitized or not (sanitized[0].isalpha() or sanitized[0] == "_"):
            sanitized = f"C1F_{sanitized}"
        while True:
            name = f"{sanitized}{self._counter}"
            self._counter += 1
            if name not in self._occupied and name.lower() not in _RESERVED_IDENTIFIERS:
                self._occupied.add(name)
                return name


@dataclass(frozen=True)
class _TraceConfig:
    predicate: str
    arity: int
    gammas: Tuple[Tuple[int, ...], ...]
    representative_indices: Mapping[Tuple[int, ...], int]
    unary_predicates: Mapping[Tuple[int, ...], str]


def _collect_trace_configs(
    node: C1FFormula,
    *,
    function_symbol: str,
    reserved_names: Iterable[str] = (),
) -> Dict[str, _TraceConfig]:
    arities = c1f_predicate_arities(node)
    # The fresh trace predicates must also avoid the symbols of the counted
    # vocabulary that simplification has removed from ``node``.
    occupied = set(arities) | {function_symbol} | set(reserved_names)
    gammas_by_predicate: Dict[str, set[Tuple[int, ...]]] = {
        predicate: set()
        for predicate, arity in arities.items()
        if arity >= 2
    }

    def visit(current: C1FFormula) -> None:
        if isinstance(current, C1FAtom):
            if len(current.arguments) >= 2:
                exponents = tuple(term.exponent for term in current.arguments)
                shift = min(exponents)
                beta = tuple(exponent - shift for exponent in exponents)
                gammas_by_predicate[current.predicate].add(beta)
            return
        if isinstance(current, C1FNot):
            visit(current.body)
            return
        if isinstance(current, C1FBinary):
            visit(current.left)
            visit(current.right)
            return
        if isinstance(current, C1FQuantifier):
            visit(current.body)
            return
        if isinstance(current, C1FPredecessorCount):
            visit(current.body)
            return
        if isinstance(current, (C1FBooleanConstant, C1FEquality)):
            return
        raise TypeError(f"Unsupported C1F node: {type(current)!r}")

    visit(node)
    namer = _PredicateNamer(occupied)
    configs: Dict[str, _TraceConfig] = {}
    for predicate, gammas_set in sorted(gammas_by_predicate.items()):
        gammas = tuple(sorted(gammas_set))
        representatives = {
            gamma: next(index for index, exponent in enumerate(gamma) if exponent == 0)
            for gamma in gammas
        }
        unary_predicates = {
            gamma: namer.fresh(f"C1F_{predicate}_trace_")
            for gamma in gammas
        }
        configs[predicate] = _TraceConfig(
            predicate=predicate,
            arity=arities[predicate],
            gammas=gammas,
            representative_indices=representatives,
            unary_predicates=unary_predicates,
        )
    return configs


def _trace_membership_formula(
    beta: Tuple[int, ...],
    gamma: Tuple[int, ...],
    representative_index: int,
    variable: str,
    shift: int,
) -> C1FFormula:
    clauses: List[C1FFormula] = []
    for left_exponent, gamma_exponent in zip(beta, gamma):
        clauses.append(
            C1FEquality(
                C1FTerm(variable, shift + left_exponent),
                C1FTerm(variable, shift + beta[representative_index] + gamma_exponent),
            )
        )
    return _and_all(clauses)


def _trace_canonical_formula(
    beta: Tuple[int, ...],
    gamma: Tuple[int, ...],
    config: _TraceConfig,
    variable: str,
    shift: int,
) -> C1FFormula:
    gamma_index = config.gammas.index(gamma)
    clauses = [
        _trace_membership_formula(
            beta,
            gamma,
            config.representative_indices[gamma],
            variable,
            shift,
        )
    ]
    for previous in config.gammas[:gamma_index]:
        clauses.append(
            C1FNot(
                _trace_membership_formula(
                    beta,
                    previous,
                    config.representative_indices[previous],
                    variable,
                    shift,
                )
            )
        )
    return _and_all(clauses)


def _compress_higher_arity_relations(
    node: C1FFormula,
    *,
    function_symbol: str,
    reserved_names: Iterable[str] = (),
) -> Tuple[C1FFormula, int]:
    """Replace the higher-arity atoms of ``node`` by fresh unary trace atoms.

    Returns the monadic formula and the number of trace predicates that it
    introduces.  The arities of the higher-arity relations are not returned:
    they belong to the counted vocabulary, which the caller fixes before
    simplification, and not to the atoms that are still present in ``node``.
    """
    configs = _collect_trace_configs(
        node,
        function_symbol=function_symbol,
        reserved_names=reserved_names,
    )
    if not configs:
        return node, 0
    trace_predicate_count = sum(len(config.gammas) for config in configs.values())
    def transform(current: C1FFormula) -> C1FFormula:
        if isinstance(current, C1FAtom):
            if len(current.arguments) < 2:
                return current
            config = configs[current.predicate]
            variable = current.arguments[0].variable
            if any(argument.variable != variable for argument in current.arguments):
                raise ValueError(
                    "Relation atoms in a local C1F formula must use one variable"
                )
            exponents = tuple(term.exponent for term in current.arguments)
            shift = min(exponents)
            beta = tuple(exponent - shift for exponent in exponents)
            disjuncts: List[C1FFormula] = []
            for gamma in config.gammas:
                representative = config.representative_indices[gamma]
                can = _trace_canonical_formula(
                    beta,
                    gamma,
                    config,
                    variable,
                    shift,
                )
                represented_value = C1FAtom(
                    config.unary_predicates[gamma],
                    (C1FTerm(variable, shift + beta[representative]),),
                )
                disjuncts.append(C1FBinary("&", can, represented_value))
            return _or_all(disjuncts)
        if isinstance(current, C1FBooleanConstant):
            return current
        if isinstance(current, C1FEquality):
            return current
        if isinstance(current, C1FNot):
            return C1FNot(transform(current.body))
        if isinstance(current, C1FBinary):
            return C1FBinary(current.op, transform(current.left), transform(current.right))
        if isinstance(current, C1FQuantifier):
            return C1FQuantifier(
                current.kind,
                current.variable,
                transform(current.body),
                current.comparator,
                current.count,
            )
        if isinstance(current, C1FPredecessorCount):
            return C1FPredecessorCount(
                current.anchor,
                current.witness,
                transform(current.body),
                current.comparator,
                current.count,
            )
        raise TypeError(f"Unsupported C1F node: {type(current)!r}")

    return _simplify(transform(node)), trace_predicate_count


@dataclass(frozen=True)
class _Profile:
    id: int
    unary_by_position: Tuple[Tuple[bool, ...], ...]
    equivalence: Tuple[int, ...]
    visible_cycle_length: Optional[int]


@dataclass(frozen=True)
class _GuardedType:
    id: int
    profile_id: int
    predecessor_truths: Tuple[bool, ...]


@dataclass(frozen=True)
class C1FProfileStatistics:
    raw_profiles: int
    pruned_profiles: int
    profile_classes: int
    pruning_filters: int
    twin_multiplicities: Tuple[int, ...]
    guarded_types: int


@dataclass(frozen=True)
class _PredecessorPlan:
    constraints: Tuple[C1FPredecessorCount, ...]
    types: Tuple[_GuardedType, ...]
    contributions: Tuple[Tuple[int, ...], ...]
    predecessor_index: Mapping[C1FPredecessorCount, int] = field(
        repr=False,
        compare=False,
    )

    @property
    def caps(self) -> Tuple[int, ...]:
        return tuple(constraint.count + 1 for constraint in self.constraints)

    def accepts(self, type_id: int, state: Tuple[int, ...]) -> bool:
        guarded_type = self.types[type_id]
        return all(
            _compare_count(count, constraint.comparator, constraint.count)
            == guarded_type.predecessor_truths[index]
            for index, (constraint, count) in enumerate(zip(self.constraints, state))
        )


def _forward_equality_patterns(
    depth: int,
) -> Iterable[Tuple[Tuple[int, ...], Optional[int]]]:
    if depth < 0:
        raise ValueError("Profile depth must be non-negative")

    # No repetition is visible in the orbit window.
    yield tuple(range(depth + 1)), None

    # The first repetition is f^(mu + cycle_length)(x) = f^mu(x).
    for first_repeat in range(1, depth + 1):
        for tail_length in range(first_repeat):
            cycle_length = first_repeat - tail_length
            equivalence = tuple(
                position
                if position < first_repeat
                else tail_length + ((position - tail_length) % cycle_length)
                for position in range(depth + 1)
            )
            visible_cycle_length = cycle_length if tail_length == 0 else None
            yield equivalence, visible_cycle_length


def _profile_count(predicate_count: int, depth: int) -> int:
    if predicate_count < 0 or depth < 0:
        raise ValueError("Predicate count and profile depth must be non-negative")
    return (1 << (predicate_count * (depth + 1))) + sum(
        first_repeat * (1 << (predicate_count * first_repeat))
        for first_repeat in range(1, depth + 1)
    )


def _profile_filter_from_equivalence(
    node: C1FFormula,
    equivalence: Tuple[int, ...],
) -> Optional[bool]:
    if isinstance(node, C1FBooleanConstant):
        return node.value
    if isinstance(node, C1FAtom):
        return None
    if isinstance(node, C1FEquality):
        return equivalence[node.left.exponent] == equivalence[node.right.exponent]
    if isinstance(node, C1FNot):
        value = _profile_filter_from_equivalence(node.body, equivalence)
        return None if value is None else not value
    if isinstance(node, C1FBinary):
        left = _profile_filter_from_equivalence(node.left, equivalence)
        right = _profile_filter_from_equivalence(node.right, equivalence)
        if node.op == "&":
            if left is False or right is False:
                return False
            if left is True and right is True:
                return True
            return None
        if node.op == "|":
            if left is True or right is True:
                return True
            if left is False and right is False:
                return False
            return None
        if node.op == "->":
            if left is False or right is True:
                return True
            if left is True and right is False:
                return False
            return None
        if node.op == "<->":
            if left is not None and right is not None:
                return left == right
            return None
        raise ValueError(f"Unsupported boolean connective {node.op!r}")
    raise ValueError("Profile pruning filters must be quantifier-free")


def _profiles_for(
    unary_predicates: Tuple[str, ...],
    depth: int,
    pruning_filters: Tuple[C1FFormula, ...] = tuple(),
) -> Tuple[_Profile, ...]:
    profile_rows: List[_Profile] = []
    predicate_count = len(unary_predicates)
    predicate_index = {
        predicate: index for index, predicate in enumerate(unary_predicates)
    }
    unary_types = tuple(product((False, True), repeat=predicate_count))
    for equivalence, visible_cycle_length in _forward_equality_patterns(depth):
        filter_values = tuple(
            _profile_filter_from_equivalence(pruning_filter, equivalence)
            for pruning_filter in pruning_filters
        )
        if any(value is False for value in filter_values):
            continue
        pending_filters = tuple(
            pruning_filter
            for pruning_filter, value in zip(pruning_filters, filter_values)
            if value is None
        )
        class_count = max(equivalence) + 1
        for unary_by_class in product(unary_types, repeat=class_count):
            profile = _Profile(
                id=len(profile_rows),
                unary_by_position=tuple(
                    unary_by_class[equivalence[position]]
                    for position in range(depth + 1)
                ),
                equivalence=equivalence,
                visible_cycle_length=visible_cycle_length,
            )
            if pending_filters and not all(
                _profile_satisfies_quantifier_free(
                    profile,
                    pruning_filter,
                    predicate_index,
                )
                for pruning_filter in pending_filters
            ):
                continue
            profile_rows.append(profile)
    return tuple(profile_rows)


def _successor_compatible(left: _Profile, right: _Profile, depth: int) -> bool:
    return _profile_overlap_key(left, 1, depth) == _profile_overlap_key(
        right,
        0,
        depth,
    )


def _profile_overlap_key(
    profile: _Profile,
    start: int,
    length: int,
) -> Tuple[Tuple[int, ...], Tuple[Tuple[bool, ...], ...]]:
    """Canonically encode the restricted equality relation AND all unary types.

    Equal orbit shapes alone do not make adjacent colored windows compatible.
    Keep the complete unary vector at every overlapping position in the key.
    """
    canonical_classes: Dict[int, int] = {}
    equality_key: List[int] = []
    unary_key: List[Tuple[bool, ...]] = []
    for position in range(start, start + length):
        class_id = profile.equivalence[position]
        canonical_id = canonical_classes.get(class_id)
        if canonical_id is None:
            canonical_id = len(canonical_classes)
            canonical_classes[class_id] = canonical_id
        equality_key.append(canonical_id)
        unary_key.append(profile.unary_by_position[position])
    return tuple(equality_key), tuple(unary_key)


def _with_profile_id(profile: _Profile, new_id: int) -> _Profile:
    return _Profile(
        id=new_id,
        unary_by_position=profile.unary_by_position,
        equivalence=profile.equivalence,
        visible_cycle_length=profile.visible_cycle_length,
    )


def _is_quantifier_free(node: C1FFormula) -> bool:
    if isinstance(node, (C1FBooleanConstant, C1FAtom, C1FEquality)):
        return True
    if isinstance(node, C1FNot):
        return _is_quantifier_free(node.body)
    if isinstance(node, C1FBinary):
        return _is_quantifier_free(node.left) and _is_quantifier_free(node.right)
    if isinstance(node, C1FQuantifier):
        return False
    if isinstance(node, C1FPredecessorCount):
        return False
    raise TypeError(f"Unsupported C1F node: {type(node)!r}")


def _exists_quantifier_requires_witness(node: C1FQuantifier) -> bool:
    if node.kind != "exists":
        return False
    if node.comparator is None:
        return True
    assert node.count is not None
    if node.comparator == "==":
        return node.count > 0
    if node.comparator == ">":
        return True
    if node.comparator == ">=":
        return node.count >= 1
    return False


def _profile_pruning_filters(node: C1FFormula) -> Tuple[C1FFormula, ...]:
    filters: List[C1FFormula] = []

    def collect_entailed(current: C1FFormula) -> None:
        if isinstance(current, C1FBinary) and current.op == "&":
            collect_entailed(current.left)
            collect_entailed(current.right)
            return
        if isinstance(current, C1FQuantifier):
            if current.kind == "forall":
                collect_under_forall(current.body)
                return
            if _exists_quantifier_requires_witness(current):
                collect_entailed(current.body)

    def collect_under_forall(current: C1FFormula) -> None:
        if _is_quantifier_free(current):
            filters.append(current)
            return
        if isinstance(current, C1FBinary) and current.op == "&":
            collect_under_forall(current.left)
            collect_under_forall(current.right)
            return
        if isinstance(current, C1FQuantifier) and current.kind == "forall":
            collect_under_forall(current.body)

    collect_entailed(node)
    return tuple(filters)


def _profile_satisfies_quantifier_free(
    profile: _Profile,
    node: C1FFormula,
    predicate_index: Mapping[str, int],
) -> bool:
    if isinstance(node, C1FBooleanConstant):
        return node.value
    if isinstance(node, C1FAtom):
        if len(node.arguments) != 1:
            raise ValueError("Profile pruning only supports monadic atoms")
        predicate_pos = predicate_index[node.predicate]
        return profile.unary_by_position[node.arguments[0].exponent][predicate_pos]
    if isinstance(node, C1FEquality):
        return profile.equivalence[node.left.exponent] == profile.equivalence[node.right.exponent]
    if isinstance(node, C1FNot):
        return not _profile_satisfies_quantifier_free(profile, node.body, predicate_index)
    if isinstance(node, C1FBinary):
        left = _profile_satisfies_quantifier_free(profile, node.left, predicate_index)
        right = _profile_satisfies_quantifier_free(profile, node.right, predicate_index)
        if node.op == "&":
            return left and right
        if node.op == "|":
            return left or right
        if node.op == "->":
            return (not left) or right
        if node.op == "<->":
            return left == right
        raise ValueError(f"Unsupported boolean connective {node.op!r}")
    raise ValueError("Profile pruning filters must be quantifier-free")


def _collect_profile_observations(
    node: C1FFormula,
) -> Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[int, int], ...]]:
    unary_observations: set[Tuple[str, int]] = set()
    equality_observations: set[Tuple[int, int]] = set()

    def visit(current: C1FFormula) -> None:
        if isinstance(current, C1FAtom):
            if len(current.arguments) == 1:
                unary_observations.add((current.predicate, current.arguments[0].exponent))
            return
        if isinstance(current, C1FEquality):
            left = current.left.exponent
            right = current.right.exponent
            equality_observations.add((left, right) if left <= right else (right, left))
            return
        if isinstance(current, C1FNot):
            visit(current.body)
            return
        if isinstance(current, C1FBinary):
            visit(current.left)
            visit(current.right)
            return
        if isinstance(current, C1FQuantifier):
            visit(current.body)
            return
        if isinstance(current, C1FPredecessorCount):
            visit(current.body)
            return
        if isinstance(current, C1FBooleanConstant):
            return
        raise TypeError(f"Unsupported C1F node: {type(current)!r}")

    visit(node)
    return tuple(sorted(unary_observations)), tuple(sorted(equality_observations))


def _profile_observation_vector(
    profile: _Profile,
    unary_observations: Tuple[Tuple[str, int], ...],
    equality_observations: Tuple[Tuple[int, int], ...],
    predicate_index: Mapping[str, int],
) -> Tuple[Tuple[bool, ...], Tuple[bool, ...]]:
    unary_values = tuple(
        profile.unary_by_position[position][predicate_index[predicate]]
        for predicate, position in unary_observations
    )
    equality_values = tuple(
        profile.equivalence[left] == profile.equivalence[right]
        for left, right in equality_observations
    )
    return unary_values, equality_values


def _build_successor_matrix(
    profiles: Sequence[_Profile],
    depth: int,
) -> Tuple[Tuple[bool, ...], ...]:
    profile_count = len(profiles)
    right_by_prefix: Dict[
        Tuple[Tuple[int, ...], Tuple[Tuple[bool, ...], ...]],
        List[int],
    ] = {}
    for right_index, right in enumerate(profiles):
        key = _profile_overlap_key(right, 0, depth)
        right_by_prefix.setdefault(key, []).append(right_index)

    rows: List[Tuple[bool, ...]] = []
    for left in profiles:
        row = [False] * profile_count
        key = _profile_overlap_key(left, 1, depth)
        for right_index in right_by_prefix.get(key, ()):
            row[right_index] = True
        rows.append(tuple(row))
    return tuple(rows)


def _quotient_twin_profiles(
    profiles: Tuple[_Profile, ...],
    successor: Tuple[Tuple[bool, ...], ...],
    formula: C1FFormula,
    predicate_index: Mapping[str, int],
) -> Tuple[Tuple[_Profile, ...], Tuple[Tuple[bool, ...], ...], Tuple[int, ...]]:
    unary_observations, equality_observations = _collect_profile_observations(formula)
    columns = tuple(tuple(row[col] for row in successor) for col in range(len(profiles)))
    groups: Dict[
        Tuple[
            Optional[int],
            Tuple[Tuple[bool, ...], Tuple[bool, ...]],
            Tuple[bool, ...],
            Tuple[bool, ...],
        ],
        List[int],
    ] = {}
    for index, profile in enumerate(profiles):
        key = (
            profile.visible_cycle_length,
            _profile_observation_vector(
                profile,
                unary_observations,
                equality_observations,
                predicate_index,
            ),
            successor[index],
            columns[index],
        )
        groups.setdefault(key, []).append(index)

    representatives = tuple(indices[0] for indices in groups.values())
    quotient_profiles = tuple(
        _with_profile_id(profiles[representative], new_id)
        for new_id, representative in enumerate(representatives)
    )
    multiplicities = tuple(len(indices) for indices in groups.values())
    quotient_successor = tuple(
        tuple(successor[row][col] for col in representatives)
        for row in representatives
    )
    return quotient_profiles, quotient_successor, multiplicities


def _compare_count(value: int, comparator: str, bound: int) -> bool:
    if comparator == "==":
        return value == bound
    if comparator == "<":
        return value < bound
    if comparator == "<=":
        return value <= bound
    if comparator == ">":
        return value > bound
    if comparator == ">=":
        return value >= bound
    raise ValueError(f"Unsupported comparator: {comparator}")


def _collect_predecessor_constraints(
    node: C1FFormula,
) -> Tuple[C1FPredecessorCount, ...]:
    constraints: Dict[C1FPredecessorCount, None] = {}

    def visit(current: C1FFormula) -> None:
        if isinstance(current, C1FPredecessorCount):
            constraints.setdefault(current, None)
            visit(current.body)
            return
        if isinstance(current, C1FNot):
            visit(current.body)
            return
        if isinstance(current, C1FBinary):
            visit(current.left)
            visit(current.right)
            return
        if isinstance(current, C1FQuantifier):
            visit(current.body)
            return
        if isinstance(
            current,
            (C1FBooleanConstant, C1FAtom, C1FEquality, C1FPredecessorCount),
        ):
            return
        raise TypeError(f"Unsupported C1F node: {type(current)!r}")

    visit(node)
    return tuple(constraints)


def _guarded_type_satisfies_local(
    guarded_type: _GuardedType,
    node: C1FFormula,
    profiles: Tuple[_Profile, ...],
    predicate_index: Mapping[str, int],
    predecessor_index: Mapping[C1FPredecessorCount, int],
) -> bool:
    profile = profiles[guarded_type.profile_id]
    if isinstance(node, C1FBooleanConstant):
        return node.value
    if isinstance(node, C1FAtom):
        if len(node.arguments) != 1:
            raise ValueError("Local guarded-type evaluation supports only unary atoms")
        argument = node.arguments[0]
        return profile.unary_by_position[argument.exponent][predicate_index[node.predicate]]
    if isinstance(node, C1FEquality):
        return (
            profile.equivalence[node.left.exponent]
            == profile.equivalence[node.right.exponent]
        )
    if isinstance(node, C1FNot):
        return not _guarded_type_satisfies_local(
            guarded_type,
            node.body,
            profiles,
            predicate_index,
            predecessor_index,
        )
    if isinstance(node, C1FBinary):
        left = _guarded_type_satisfies_local(
            guarded_type,
            node.left,
            profiles,
            predicate_index,
            predecessor_index,
        )
        right = _guarded_type_satisfies_local(
            guarded_type,
            node.right,
            profiles,
            predicate_index,
            predecessor_index,
        )
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
        return guarded_type.predecessor_truths[predecessor_index[node]]
    if isinstance(node, C1FQuantifier):
        raise ValueError("Expected a local formula without ordinary quantifiers")
    raise TypeError(f"Unsupported C1F node: {type(node)!r}")


def _compile_predecessor_plan(
    node: C1FFormula,
    profiles: Tuple[_Profile, ...],
    predicate_index: Mapping[str, int],
) -> _PredecessorPlan:
    constraints = _collect_predecessor_constraints(node)
    predecessor_index = {
        constraint: index
        for index, constraint in enumerate(constraints)
    }
    types: List[_GuardedType] = []
    for profile in profiles:
        for truths in product((False, True), repeat=len(constraints)):
            types.append(
                _GuardedType(
                    id=len(types),
                    profile_id=profile.id,
                    predecessor_truths=truths,
                )
            )
    guarded_types = tuple(types)
    contributions = tuple(
        tuple(
            1
            if _guarded_type_satisfies_local(
                guarded_type,
                constraint.body,
                profiles,
                predicate_index,
                predecessor_index,
            )
            else 0
            for constraint in constraints
        )
        for guarded_type in guarded_types
    )
    return _PredecessorPlan(
        constraints=constraints,
        types=guarded_types,
        contributions=contributions,
        predecessor_index=predecessor_index,
    )


@dataclass(frozen=True)
class _CardinalityMarker:
    formula: C1FFormula
    cap: int


@dataclass(frozen=True)
class _CardinalityPlan:
    markers: Tuple[_CardinalityMarker, ...]
    profile_states: Tuple[Tuple[int, ...], ...]
    counter_sources: Mapping[C1FFormula, Tuple[str, int]] = field(
        repr=False,
        compare=False,
    )

    @property
    def caps(self) -> Tuple[int, ...]:
        return tuple(marker.cap for marker in self.markers)

    def counter_value(
        self,
        formula: C1FFormula,
        marker_state: Tuple[int, ...],
        domain_size: int,
    ) -> int:
        kind, index = self.counter_sources[formula]
        if kind == "zero":
            return 0
        if kind == "domain":
            return domain_size
        return marker_state[index]

    def satisfies(
        self,
        formula: C1FFormula,
        marker_state: Tuple[int, ...],
        domain_size: int,
    ) -> bool:
        @lru_cache(maxsize=None)
        def eval_closed(node: C1FFormula) -> bool:
            if isinstance(node, C1FBooleanConstant):
                return node.value
            if isinstance(node, C1FNot):
                return not eval_closed(node.body)
            if isinstance(node, C1FBinary):
                left = eval_closed(node.left)
                right = eval_closed(node.right)
                if node.op == "&":
                    return left and right
                if node.op == "|":
                    return left or right
                if node.op == "->":
                    return (not left) or right
                if node.op == "<->":
                    return left == right
                raise ValueError(f"Unsupported boolean connective {node.op!r}")
            if isinstance(node, C1FQuantifier):
                residual = _replace_outer_quantifiers(node.body, eval_closed)
                if node.kind == "forall":
                    counter_formula = _simplify(C1FNot(residual))
                    violation_count = _quantifier_free_count(
                        counter_formula,
                        self,
                        marker_state,
                        domain_size,
                    )
                    return violation_count == 0
                witness_count = _quantifier_free_count(
                    residual,
                    self,
                    marker_state,
                    domain_size,
                )
                if node.comparator is None:
                    return witness_count >= 1
                assert node.count is not None
                return _compare_count(witness_count, node.comparator, node.count)
            raise ValueError("Expected a closed C1F formula while evaluating markers")

        return eval_closed(formula)


def _outermost_quantifiers(node: C1FFormula) -> Tuple[C1FQuantifier, ...]:
    found: List[C1FQuantifier] = []

    def visit(current: C1FFormula) -> None:
        if isinstance(current, C1FQuantifier):
            found.append(current)
            return
        if isinstance(current, C1FNot):
            visit(current.body)
            return
        if isinstance(current, C1FBinary):
            visit(current.left)
            visit(current.right)
            return
        if isinstance(
            current,
            (C1FBooleanConstant, C1FAtom, C1FEquality, C1FPredecessorCount),
        ):
            return
        raise TypeError(f"Unsupported C1F node: {type(current)!r}")

    visit(node)
    return tuple(dict.fromkeys(found))


def _replace_quantifier_values(
    node: C1FFormula,
    values: Mapping[C1FQuantifier, bool],
) -> C1FFormula:
    if isinstance(node, C1FQuantifier):
        return C1FBooleanConstant(values[node])
    if isinstance(
        node,
        (C1FBooleanConstant, C1FAtom, C1FEquality, C1FPredecessorCount),
    ):
        return node
    if isinstance(node, C1FNot):
        return C1FNot(_replace_quantifier_values(node.body, values))
    if isinstance(node, C1FBinary):
        return C1FBinary(
            node.op,
            _replace_quantifier_values(node.left, values),
            _replace_quantifier_values(node.right, values),
        )
    raise TypeError(f"Unsupported C1F node: {type(node)!r}")


def _replace_outer_quantifiers(
    node: C1FFormula,
    evaluator,
) -> C1FFormula:
    if isinstance(node, C1FQuantifier):
        return C1FBooleanConstant(evaluator(node))
    if isinstance(
        node,
        (C1FBooleanConstant, C1FAtom, C1FEquality, C1FPredecessorCount),
    ):
        return node
    if isinstance(node, C1FNot):
        return _simplify(C1FNot(_replace_outer_quantifiers(node.body, evaluator)))
    if isinstance(node, C1FBinary):
        return _simplify(
            C1FBinary(
                node.op,
                _replace_outer_quantifiers(node.left, evaluator),
                _replace_outer_quantifiers(node.right, evaluator),
            )
        )
    raise TypeError(f"Unsupported C1F node: {type(node)!r}")


def _quantifier_counter_formula(
    quantifier: C1FQuantifier,
    residual: C1FFormula,
) -> C1FFormula:
    if quantifier.kind == "forall":
        return _simplify(C1FNot(residual))
    return residual


def _quantifier_marker_cap(quantifier: C1FQuantifier) -> int:
    if quantifier.kind == "forall" or quantifier.comparator is None:
        return 1
    assert quantifier.count is not None
    return quantifier.count + 1


def _quantifier_free_count(
    formula: C1FFormula,
    plan: _CardinalityPlan,
    marker_state: Tuple[int, ...],
    domain_size: int,
) -> int:
    if isinstance(formula, C1FBooleanConstant):
        return domain_size if formula.value else 0
    return plan.counter_value(formula, marker_state, domain_size)


def _collect_cardinality_counters(node: C1FFormula) -> Dict[C1FFormula, int]:
    requested: Dict[C1FFormula, int] = {}

    def visit(current: C1FFormula) -> None:
        if isinstance(current, C1FQuantifier):
            nested = _outermost_quantifiers(current.body)
            for truth_values in product((False, True), repeat=len(nested)):
                residual = _simplify(
                    _replace_quantifier_values(
                        current.body,
                        dict(zip(nested, truth_values)),
                    )
                )
                counter_formula = _quantifier_counter_formula(current, residual)
                if not isinstance(counter_formula, C1FBooleanConstant):
                    requested[counter_formula] = max(
                        requested.get(counter_formula, 0),
                        _quantifier_marker_cap(current),
                    )
            visit(current.body)
            return
        if isinstance(current, C1FNot):
            visit(current.body)
            return
        if isinstance(current, C1FBinary):
            visit(current.left)
            visit(current.right)
            return
        if isinstance(
            current,
            (C1FBooleanConstant, C1FAtom, C1FEquality, C1FPredecessorCount),
        ):
            return
        raise TypeError(f"Unsupported C1F node: {type(current)!r}")

    visit(node)
    return requested


def _compile_cardinality_plan(
    node: C1FFormula,
    element_count: int,
    local_membership,
) -> _CardinalityPlan:
    requested = _collect_cardinality_counters(node)
    dynamic: List[_CardinalityMarker] = []
    counter_sources: Dict[C1FFormula, Tuple[str, int]] = {}
    memberships: Dict[C1FFormula, Tuple[bool, ...]] = {}
    for formula, cap in requested.items():
        membership = tuple(
            local_membership(element_id, formula)
            for element_id in range(element_count)
        )
        memberships[formula] = membership
        if not any(membership):
            counter_sources[formula] = ("zero", -1)
        elif all(membership):
            counter_sources[formula] = ("domain", -1)
        else:
            counter_sources[formula] = ("marker", len(dynamic))
            dynamic.append(_CardinalityMarker(formula, cap))

    profile_states = tuple(
        tuple(
            1 if memberships[marker.formula][element_id] else 0
            for marker in dynamic
        )
        for element_id in range(element_count)
    )
    return _CardinalityPlan(
        markers=tuple(dynamic),
        profile_states=profile_states,
        counter_sources=counter_sources,
    )


def _distribution_add_scaled(target, source, scalar) -> None:
    if scalar == 0:
        return
    for state, value in source.items():
        updated = target.get(state, mpq(0)) + scalar * value
        if updated:
            target[state] = updated
        else:
            target.pop(state, None)


def _distribution_multiply(left, right, caps: Tuple[int, ...]):
    if not left or not right:
        return {}
    out = {}
    for left_state, left_value in left.items():
        for right_state, right_value in right.items():
            state = tuple(
                min(cap, left_count + right_count)
                for cap, left_count, right_count in zip(caps, left_state, right_state)
            )
            out[state] = out.get(state, mpq(0)) + left_value * right_value
    return out


def _zero_series(degree_limit: int):
    return [{} for _ in range(degree_limit + 1)]


def _one_series(degree_limit: int, zero_state: Tuple[int, ...]):
    out = _zero_series(degree_limit)
    out[0] = {zero_state: mpq(1)}
    return out


def _series_add_in_place(target, source, scalar=1) -> None:
    for degree, coefficient in enumerate(source):
        _distribution_add_scaled(target[degree], coefficient, scalar)


def _series_multiply(left, right, caps: Tuple[int, ...], degree_limit: int):
    out = _zero_series(degree_limit)
    for left_degree, left_coefficient in enumerate(left):
        if not left_coefficient:
            continue
        for right_degree in range(degree_limit - left_degree + 1):
            right_coefficient = right[right_degree]
            if not right_coefficient:
                continue
            product_coefficient = _distribution_multiply(
                left_coefficient,
                right_coefficient,
                caps,
            )
            _distribution_add_scaled(
                out[left_degree + right_degree],
                product_coefficient,
                1,
            )
    return out


def _series_exp(series, caps: Tuple[int, ...], degree_limit: int):
    zero_state = tuple(0 for _ in caps)
    out = _one_series(degree_limit, zero_state)
    for degree in range(1, degree_limit + 1):
        coefficient = {}
        for left_degree in range(1, degree + 1):
            product_coefficient = _distribution_multiply(
                series[left_degree],
                out[degree - left_degree],
                caps,
            )
            _distribution_add_scaled(
                coefficient,
                product_coefficient,
                left_degree,
            )
        _distribution_add_scaled(out[degree], coefficient, mpq(1, degree))
    return out


def _series_adams(
    series,
    repeat: int,
    caps: Tuple[int, ...],
    degree_limit: int,
):
    if repeat < 1:
        raise ValueError("Adams-operation repeat must be positive")
    out = _zero_series(degree_limit)
    for degree in range(degree_limit // repeat + 1):
        target_degree = degree * repeat
        for state, value in series[degree].items():
            repeated_state = tuple(
                min(cap, repeat * count)
                for cap, count in zip(caps, state)
            )
            coefficient = out[target_degree]
            coefficient[repeated_state] = (
                coefficient.get(repeated_state, mpq(0)) + value
            )
    return out


def _series_multiset(series, caps: Tuple[int, ...], degree_limit: int):
    exponent = _zero_series(degree_limit)
    for repeat in range(1, degree_limit + 1):
        repeated = _series_adams(series, repeat, caps, degree_limit)
        _series_add_in_place(exponent, repeated, mpq(1, repeat))
    return _series_exp(exponent, caps, degree_limit)


@lru_cache(maxsize=None)
def _divisors(value: int) -> Tuple[int, ...]:
    return tuple(divisor for divisor in range(1, value + 1) if value % divisor == 0)


@lru_cache(maxsize=None)
def _euler_totient(value: int) -> int:
    result = value
    remaining = value
    prime = 2
    while prime * prime <= remaining:
        if remaining % prime == 0:
            while remaining % prime == 0:
                remaining //= prime
            result -= result // prime
        prime += 1
    if remaining > 1:
        result -= result // remaining
    return result


def _atom_times_series(
    atom_state: Tuple[int, ...],
    atom_weight: int,
    series,
    caps: Tuple[int, ...],
    degree_limit: int,
):
    out = _zero_series(degree_limit)
    atom = {atom_state: mpq(atom_weight)}
    for degree in range(1, degree_limit + 1):
        out[degree] = _distribution_multiply(atom, series[degree - 1], caps)
    return out


def _series_append_state(
    series,
    appended_state: Tuple[int, ...],
    degree_limit: int,
):
    out = _zero_series(degree_limit)
    for degree, coefficient in enumerate(series):
        out[degree] = {
            state + appended_state: value
            for state, value in coefficient.items()
        }
    return out


def _series_add_appended_state(
    series,
    appended_state: Tuple[int, ...],
    caps: Tuple[int, ...],
    prefix_size: int,
    degree_limit: int,
):
    out = _zero_series(degree_limit)
    for degree, coefficient in enumerate(series):
        for state, value in coefficient.items():
            prefix = state[:prefix_size]
            suffix = state[prefix_size:]
            updated_suffix = tuple(
                min(cap, count + addition)
                for cap, count, addition in zip(
                    caps[prefix_size:],
                    suffix,
                    appended_state,
                )
            )
            updated = prefix + updated_suffix
            out[degree][updated] = out[degree].get(updated, mpq(0)) + value
    return out


def _series_filter_local_state(
    series,
    prefix_size: int,
    predicate,
    degree_limit: int,
):
    out = _zero_series(degree_limit)
    for degree, coefficient in enumerate(series):
        for state, value in coefficient.items():
            if not predicate(state[prefix_size:]):
                continue
            prefix = state[:prefix_size]
            out[degree][prefix] = out[degree].get(prefix, mpq(0)) + value
    return out


def _cycle_trace_series(
    cycle_length: int,
    active_profiles: Tuple[int, ...],
    decorations,
    successor: Tuple[Tuple[bool, ...], ...],
    caps: Tuple[int, ...],
    degree_limit: int,
):
    trace = _zero_series(degree_limit)
    active_set = frozenset(active_profiles)
    outgoing = {
        profile: tuple(
            target
            for target in active_profiles
            if successor[profile][target]
        )
        for profile in active_profiles
    }
    for start in active_profiles:
        paths = {start: decorations[start]}
        for _ in range(1, cycle_length):
            next_paths = {}
            for last, path_series in paths.items():
                for target in outgoing[last]:
                    extended = _series_multiply(
                        path_series,
                        decorations[target],
                        caps,
                        degree_limit,
                    )
                    target_series = next_paths.get(target)
                    if target_series is None:
                        target_series = _zero_series(degree_limit)
                        next_paths[target] = target_series
                    _series_add_in_place(target_series, extended)
            paths = next_paths
            if not paths:
                break
        for last, path_series in paths.items():
            if last in active_set and successor[last][start]:
                _series_add_in_place(trace, path_series)
    return trace


def _cycle_transfer_trace_series(
    cycle_length: int,
    active_types: Tuple[int, ...],
    transfers,
    caps: Tuple[int, ...],
    degree_limit: int,
):
    trace = _zero_series(degree_limit)
    zero_state = tuple(0 for _ in caps)
    for start in active_types:
        paths = {start: _one_series(degree_limit, zero_state)}
        for _ in range(cycle_length):
            next_paths = {}
            for previous, path_series in paths.items():
                for target in active_types:
                    edge_series = transfers[previous][target]
                    if edge_series is None:
                        continue
                    extended = _series_multiply(
                        path_series,
                        edge_series,
                        caps,
                        degree_limit,
                    )
                    target_series = next_paths.get(target)
                    if target_series is None:
                        target_series = _zero_series(degree_limit)
                        next_paths[target] = target_series
                    _series_add_in_place(target_series, extended)
            paths = next_paths
            if not paths:
                break
        closed = paths.get(start)
        if closed is not None:
            _series_add_in_place(trace, closed)
    return trace


@dataclass(frozen=True)
class _CompiledMonadicC1F:
    formula: C1FFormula
    unary_predicates: Tuple[str, ...]
    depth: int
    profiles: Tuple[_Profile, ...]
    successor: Tuple[Tuple[bool, ...], ...]
    profile_multiplicities: Tuple[int, ...]
    raw_profile_count: int
    pruned_profile_count: int
    pruning_filter_count: int
    cardinality_plan: _CardinalityPlan = field(repr=False, compare=False)
    predecessor_plan: _PredecessorPlan = field(repr=False, compare=False)
    predicate_index: Mapping[str, int] = field(repr=False, compare=False)

    def evaluate_egf(self, domain_size: int) -> int:
        if domain_size < 0:
            raise ValueError("Domain size must be non-negative")
        if self.predecessor_plan.constraints:
            return self._evaluate_guarded_egf(domain_size)

        caps = self.cardinality_plan.caps
        profile_count = len(self.profiles)
        transient_profiles = tuple(
            profile.id
            for profile in self.profiles
            if profile.visible_cycle_length is None
        )

        # T_p = x_p exp(sum_{q -> p} T_q), truncated by iterative substitution.
        trees = {
            profile: _zero_series(domain_size)
            for profile in transient_profiles
        }
        for _ in range(domain_size):
            next_trees = {}
            for profile in transient_profiles:
                incoming = _zero_series(domain_size)
                for child in transient_profiles:
                    if self.successor[child][profile]:
                        _series_add_in_place(incoming, trees[child])
                child_sets = _series_exp(incoming, caps, domain_size)
                next_trees[profile] = _atom_times_series(
                    self.cardinality_plan.profile_states[profile],
                    self.profile_multiplicities[profile],
                    child_sets,
                    caps,
                    domain_size,
                )
            trees = next_trees

        decorations = []
        for profile in range(profile_count):
            incoming = _zero_series(domain_size)
            for child in transient_profiles:
                if self.successor[child][profile]:
                    _series_add_in_place(incoming, trees[child])
            child_sets = _series_exp(incoming, caps, domain_size)
            decorations.append(
                _atom_times_series(
                    self.cardinality_plan.profile_states[profile],
                    self.profile_multiplicities[profile],
                    child_sets,
                    caps,
                    domain_size,
                )
            )

        # A mapping is a set of directed cycles decorated by the rooted trees.
        connected = _zero_series(domain_size)
        for cycle_length in range(1, domain_size + 1):
            if cycle_length <= self.depth:
                active_profiles = tuple(
                    profile.id
                    for profile in self.profiles
                    if profile.visible_cycle_length == cycle_length
                )
            else:
                active_profiles = transient_profiles
            trace = _cycle_trace_series(
                cycle_length,
                active_profiles,
                decorations,
                self.successor,
                caps,
                domain_size,
            )
            _series_add_in_place(connected, trace, mpq(1, cycle_length))

        mappings = _series_exp(connected, caps, domain_size)
        total = mpq(0)
        for marker_state, coefficient in mappings[domain_size].items():
            if self.cardinality_plan.satisfies(
                self.formula,
                marker_state,
                domain_size,
            ):
                total += coefficient * factorial(domain_size)
        integer_total = int(total)
        if total != mpq(integer_total):
            raise ValueError("Internal error: C1F EGF result is not an integer")
        return integer_total

    def evaluate_unlabeled(self, domain_size: int) -> int:
        if domain_size < 0:
            raise ValueError("Domain size must be non-negative")
        if self.predecessor_plan.constraints:
            return self._evaluate_guarded_unlabeled(domain_size)

        caps = self.cardinality_plan.caps
        profile_count = len(self.profiles)
        transient_profiles = tuple(
            profile.id
            for profile in self.profiles
            if profile.visible_cycle_length is None
        )

        trees = {
            profile: _zero_series(domain_size)
            for profile in transient_profiles
        }
        for _ in range(domain_size):
            next_trees = {}
            for profile in transient_profiles:
                incoming = _zero_series(domain_size)
                for child in transient_profiles:
                    if self.successor[child][profile]:
                        _series_add_in_place(incoming, trees[child])
                child_multiset = _series_multiset(incoming, caps, domain_size)
                next_trees[profile] = _atom_times_series(
                    self.cardinality_plan.profile_states[profile],
                    self.profile_multiplicities[profile],
                    child_multiset,
                    caps,
                    domain_size,
                )
            trees = next_trees

        decorations = []
        for profile in range(profile_count):
            incoming = _zero_series(domain_size)
            for child in transient_profiles:
                if self.successor[child][profile]:
                    _series_add_in_place(incoming, trees[child])
            child_multiset = _series_multiset(incoming, caps, domain_size)
            decorations.append(
                _atom_times_series(
                    self.cardinality_plan.profile_states[profile],
                    self.profile_multiplicities[profile],
                    child_multiset,
                    caps,
                    domain_size,
                )
            )

        adams_decorations = {}
        connected = _zero_series(domain_size)
        for cycle_length in range(1, domain_size + 1):
            if cycle_length <= self.depth:
                active_profiles = tuple(
                    profile.id
                    for profile in self.profiles
                    if profile.visible_cycle_length == cycle_length
                )
            else:
                active_profiles = transient_profiles
            for repeat in _divisors(cycle_length):
                repeated_decorations = adams_decorations.get(repeat)
                if repeated_decorations is None:
                    repeated_decorations = tuple(
                        _series_adams(
                            decoration,
                            repeat,
                            caps,
                            domain_size,
                        )
                        for decoration in decorations
                    )
                    adams_decorations[repeat] = repeated_decorations
                trace = _cycle_trace_series(
                    cycle_length // repeat,
                    active_profiles,
                    repeated_decorations,
                    self.successor,
                    caps,
                    domain_size,
                )
                _series_add_in_place(
                    connected,
                    trace,
                    mpq(_euler_totient(repeat), cycle_length),
                )

        mappings = _series_multiset(connected, caps, domain_size)
        total = mpq(0)
        for marker_state, coefficient in mappings[domain_size].items():
            if self.cardinality_plan.satisfies(
                self.formula,
                marker_state,
                domain_size,
            ):
                total += coefficient
        integer_total = int(total)
        if total != mpq(integer_total):
            raise ValueError("Internal error: C1F unlabeled result is not an integer")
        return integer_total

    def _guarded_child_collections(self, domain_size: int, *, unlabeled: bool):
        global_caps = self.cardinality_plan.caps
        local_caps = self.predecessor_plan.caps
        combined_caps = global_caps + local_caps
        prefix_size = len(global_caps)
        types = self.predecessor_plan.types
        transient_types = tuple(
            guarded_type.id
            for guarded_type in types
            if self.profiles[guarded_type.profile_id].visible_cycle_length is None
        )
        collection = _series_multiset if unlabeled else _series_exp
        trees = {
            type_id: _zero_series(domain_size)
            for type_id in transient_types
        }
        for _ in range(domain_size):
            next_trees = {}
            for parent_id in transient_types:
                parent = types[parent_id]
                incoming = _zero_series(domain_size)
                for child_id in transient_types:
                    child = types[child_id]
                    if not self.successor[child.profile_id][parent.profile_id]:
                        continue
                    lifted = _series_append_state(
                        trees[child_id],
                        self.predecessor_plan.contributions[child_id],
                        domain_size,
                    )
                    _series_add_in_place(incoming, lifted)
                child_collection = collection(incoming, combined_caps, domain_size)
                valid_children = _series_filter_local_state(
                    child_collection,
                    prefix_size,
                    lambda state, parent_id=parent_id: self.predecessor_plan.accepts(
                        parent_id,
                        state,
                    ),
                    domain_size,
                )
                next_trees[parent_id] = _atom_times_series(
                    self.cardinality_plan.profile_states[parent_id],
                    self.profile_multiplicities[parent.profile_id],
                    valid_children,
                    global_caps,
                    domain_size,
                )
            trees = next_trees

        offcycle_collections = []
        for parent_id, parent in enumerate(types):
            incoming = _zero_series(domain_size)
            for child_id in transient_types:
                child = types[child_id]
                if not self.successor[child.profile_id][parent.profile_id]:
                    continue
                lifted = _series_append_state(
                    trees[child_id],
                    self.predecessor_plan.contributions[child_id],
                    domain_size,
                )
                _series_add_in_place(incoming, lifted)
            offcycle_collections.append(
                collection(incoming, combined_caps, domain_size)
            )
        return tuple(offcycle_collections), combined_caps

    def _guarded_cycle_transfers(self, domain_size: int, *, unlabeled: bool):
        global_caps = self.cardinality_plan.caps
        prefix_size = len(global_caps)
        offcycle_collections, combined_caps = self._guarded_child_collections(
            domain_size,
            unlabeled=unlabeled,
        )
        types = self.predecessor_plan.types
        transfers = []
        for previous_id, previous in enumerate(types):
            row = []
            for current_id, current in enumerate(types):
                if not self.successor[previous.profile_id][current.profile_id]:
                    row.append(None)
                    continue
                with_cycle_predecessor = _series_add_appended_state(
                    offcycle_collections[current_id],
                    self.predecessor_plan.contributions[previous_id],
                    combined_caps,
                    prefix_size,
                    domain_size,
                )
                valid_children = _series_filter_local_state(
                    with_cycle_predecessor,
                    prefix_size,
                    lambda state, current_id=current_id: self.predecessor_plan.accepts(
                        current_id,
                        state,
                    ),
                    domain_size,
                )
                row.append(
                    _atom_times_series(
                        self.cardinality_plan.profile_states[current_id],
                        self.profile_multiplicities[current.profile_id],
                        valid_children,
                        global_caps,
                        domain_size,
                    )
                )
            transfers.append(tuple(row))
        return tuple(transfers)

    def _active_guarded_types(self, cycle_length: int) -> Tuple[int, ...]:
        if cycle_length <= self.depth:
            return tuple(
                guarded_type.id
                for guarded_type in self.predecessor_plan.types
                if self.profiles[guarded_type.profile_id].visible_cycle_length
                == cycle_length
            )
        return tuple(
            guarded_type.id
            for guarded_type in self.predecessor_plan.types
            if self.profiles[guarded_type.profile_id].visible_cycle_length is None
        )

    def _evaluate_guarded_egf(self, domain_size: int) -> int:
        caps = self.cardinality_plan.caps
        transfers = self._guarded_cycle_transfers(
            domain_size,
            unlabeled=False,
        )
        connected = _zero_series(domain_size)
        for cycle_length in range(1, domain_size + 1):
            trace = _cycle_transfer_trace_series(
                cycle_length,
                self._active_guarded_types(cycle_length),
                transfers,
                caps,
                domain_size,
            )
            _series_add_in_place(connected, trace, mpq(1, cycle_length))
        mappings = _series_exp(connected, caps, domain_size)
        total = mpq(0)
        for marker_state, coefficient in mappings[domain_size].items():
            if self.cardinality_plan.satisfies(
                self.formula,
                marker_state,
                domain_size,
            ):
                total += coefficient * factorial(domain_size)
        integer_total = int(total)
        if total != mpq(integer_total):
            raise ValueError("Internal error: guarded C1F EGF result is not an integer")
        return integer_total

    def _evaluate_guarded_unlabeled(self, domain_size: int) -> int:
        caps = self.cardinality_plan.caps
        transfers = self._guarded_cycle_transfers(
            domain_size,
            unlabeled=True,
        )
        adams_transfers = {}
        connected = _zero_series(domain_size)
        for cycle_length in range(1, domain_size + 1):
            active_types = self._active_guarded_types(cycle_length)
            for repeat in _divisors(cycle_length):
                repeated_transfers = adams_transfers.get(repeat)
                if repeated_transfers is None:
                    repeated_transfers = tuple(
                        tuple(
                            None
                            if edge is None
                            else _series_adams(edge, repeat, caps, domain_size)
                            for edge in row
                        )
                        for row in transfers
                    )
                    adams_transfers[repeat] = repeated_transfers
                trace = _cycle_transfer_trace_series(
                    cycle_length // repeat,
                    active_types,
                    repeated_transfers,
                    caps,
                    domain_size,
                )
                _series_add_in_place(
                    connected,
                    trace,
                    mpq(_euler_totient(repeat), cycle_length),
                )
        mappings = _series_multiset(connected, caps, domain_size)
        total = mpq(0)
        for marker_state, coefficient in mappings[domain_size].items():
            if self.cardinality_plan.satisfies(
                self.formula,
                marker_state,
                domain_size,
            ):
                total += coefficient
        integer_total = int(total)
        if total != mpq(integer_total):
            raise ValueError(
                "Internal error: guarded C1F unlabeled result is not an integer"
            )
        return integer_total



def _compile_monadic_c1f(
    node: C1FFormula,
    *,
    function_symbol: str,
    profile_pruning: bool,
    twin_compression: bool,
    unary_vocabulary: Iterable[str] = (),
) -> _CompiledMonadicC1F:
    validate_c1f_formula(node, require_closed=True, function_symbol=function_symbol)
    arities = c1f_predicate_arities(node)
    non_unary = sorted(predicate for predicate, arity in arities.items() if arity != 1)
    if non_unary:
        raise ValueError(
            "Monadic C1F compilation expects only unary predicates; found "
            + ", ".join(non_unary)
    )
    # ``unary_vocabulary`` lists unary predicates that are counted even though
    # they need not occur in ``node``: a predicate that simplification removed
    # from the sentence is still interpreted freely in every structure.  The
    # profiles record all of them, and the predicates of ``node`` in any case.
    unary_predicates = tuple(sorted(set(arities) | set(unary_vocabulary)))
    depth = _max_term_depth(node)
    predicate_index = {predicate: index for index, predicate in enumerate(unary_predicates)}
    pruning_filters = _profile_pruning_filters(node) if profile_pruning else tuple()
    raw_profile_count = _profile_count(len(unary_predicates), depth)
    pruned_profiles = _profiles_for(
        unary_predicates,
        depth,
        pruning_filters,
    )

    reindexed_profiles = tuple(
        _with_profile_id(profile, index)
        for index, profile in enumerate(pruned_profiles)
    )
    successor = _build_successor_matrix(reindexed_profiles, depth)
    profile_multiplicities = tuple(1 for _ in reindexed_profiles)
    profiles = reindexed_profiles
    if twin_compression:
        profiles, successor, profile_multiplicities = _quotient_twin_profiles(
            profiles,
            successor,
            node,
            predicate_index,
        )
    predecessor_plan = _compile_predecessor_plan(
        node,
        profiles,
        predicate_index,
    )
    cardinality_plan = _compile_cardinality_plan(
        node,
        len(predecessor_plan.types),
        lambda type_id, formula: _guarded_type_satisfies_local(
            predecessor_plan.types[type_id],
            formula,
            profiles,
            predicate_index,
            predecessor_plan.predecessor_index,
        ),
    )
    return _CompiledMonadicC1F(
        formula=node,
        unary_predicates=unary_predicates,
        depth=depth,
        profiles=profiles,
        successor=successor,
        profile_multiplicities=profile_multiplicities,
        raw_profile_count=raw_profile_count,
        pruned_profile_count=len(pruned_profiles),
        pruning_filter_count=len(pruning_filters),
        cardinality_plan=cardinality_plan,
        predecessor_plan=predecessor_plan,
        predicate_index=predicate_index,
    )


@dataclass(frozen=True)
class _CompiledC1FBranch:
    monadic: _CompiledMonadicC1F
    higher_relation_arities: Tuple[int, ...]
    trace_predicate_count: int

    def evaluate(self, domain_size: int):
        monadic_count = mpq(self.monadic.evaluate_egf(domain_size))
        correction_exponent = sum(domain_size ** arity for arity in self.higher_relation_arities)
        correction_exponent -= self.trace_predicate_count * domain_size
        if correction_exponent >= 0:
            return monadic_count * (mpq(2) ** correction_exponent)
        return monadic_count / (mpq(2) ** (-correction_exponent))

    def evaluate_unlabeled(self, domain_size: int):
        return mpq(self.monadic.evaluate_unlabeled(domain_size))

    def profile_statistics(self) -> C1FProfileStatistics:
        return C1FProfileStatistics(
            raw_profiles=self.monadic.raw_profile_count,
            pruned_profiles=self.monadic.pruned_profile_count,
            profile_classes=len(self.monadic.profiles),
            pruning_filters=self.monadic.pruning_filter_count,
            twin_multiplicities=self.monadic.profile_multiplicities,
            guarded_types=len(self.monadic.predecessor_plan.types),
        )


@dataclass(frozen=True)
class CompiledC1F:
    formula: str
    function_symbol: str
    has_higher_arity_relations: bool
    branches: Tuple[_CompiledC1FBranch, ...]

    def fomc(self, domain_size: int):
        return evaluate_c1f_fomc(self, domain_size)

    def ufomc(self, domain_size: int):
        return evaluate_c1f_ufomc(self, domain_size)

    def profile_statistics(self) -> Tuple[C1FProfileStatistics, ...]:
        return tuple(branch.profile_statistics() for branch in self.branches)

    def labeled_specification(self) -> "C1FLabeledSpecification":
        """Extract the finite labeled EGF specification for this sentence."""

        from .asymptotics import compile_c1f_labeled_specification

        return compile_c1f_labeled_specification(self)

    def labeled_asymptotics(
        self,
        *,
        tolerance: float = 1e-10,
        max_iterations: int = 80,
        certify: bool = False,
        precision_bits: int = 192,
        max_precision_bits: int | None = None,
        expansion_order: int = 0,
    ) -> "C1FLabeledAsymptotics":
        """Analyze the leading labeled asymptotic of this sentence.

        Set ``certify=True`` to attach optional Arb interval enclosures and
        ``expansion_order`` to request successive half-orders.
        """

        from .asymptotics import analyze_c1f_labeled_asymptotics

        return analyze_c1f_labeled_asymptotics(
            self,
            tolerance=tolerance,
            max_iterations=max_iterations,
            certify=certify,
            precision_bits=precision_bits,
            max_precision_bits=max_precision_bits,
            expansion_order=expansion_order,
        )

    def unlabeled_asymptotics(
        self,
        *,
        tolerance: float = 1e-10,
        max_iterations: int = 80,
        certify: bool = False,
        precision_bits: int = 192,
        max_precision_bits: int | None = None,
        polya_tail_terms: int = 128,
        expansion_order: int = 0,
    ) -> "C1FUnlabeledAsymptotics":
        """Analyze an unlabeled regime and optional continued expansion."""

        from .asymptotics import c1f_unlabeled_asymptotics

        return c1f_unlabeled_asymptotics(
            self,
            tolerance=tolerance,
            max_iterations=max_iterations,
            certify=certify,
            precision_bits=precision_bits,
            max_precision_bits=max_precision_bits,
            polya_tail_terms=polya_tail_terms,
            expansion_order=expansion_order,
        )


def compile_c1f_fomc(
    formula: str | C1FFormula,
    *,
    function_symbol: str = "f",
    profile_pruning: bool = True,
    twin_compression: bool = True,
) -> CompiledC1F:
    _validate_identifier(function_symbol, "Function symbol")
    parsed = parse_c1f_sentence(formula, function_symbol=function_symbol) if isinstance(formula, str) else formula
    validate_c1f_formula(parsed, require_closed=True, function_symbol=function_symbol)
    arities = c1f_predicate_arities(parsed)
    nullaries = tuple(sorted(predicate for predicate, arity in arities.items() if arity == 0))
    # The models are counted over the vocabulary of the sentence as given.
    # Fix it here: substituting the nullary symbols and simplifying Boolean
    # constants below may remove relation symbols from a branch, but such a
    # symbol is still interpreted freely in the structures of that branch.
    unary_vocabulary = tuple(
        sorted(predicate for predicate, arity in arities.items() if arity == 1)
    )
    higher_relation_arities = tuple(
        arities[predicate]
        for predicate in sorted(arities)
        if arities[predicate] >= 2
    )
    branches: List[_CompiledC1FBranch] = []
    for values in product((False, True), repeat=len(nullaries)):
        assignment = dict(zip(nullaries, values))
        specialized = _simplify(_replace_nullary(parsed, assignment))
        monadic_formula, trace_predicate_count = _compress_higher_arity_relations(
            specialized,
            function_symbol=function_symbol,
            reserved_names=arities,
        )
        branches.append(
            _CompiledC1FBranch(
                monadic=_compile_monadic_c1f(
                    monadic_formula,
                    function_symbol=function_symbol,
                    profile_pruning=profile_pruning,
                    twin_compression=twin_compression,
                    unary_vocabulary=unary_vocabulary,
                ),
                higher_relation_arities=higher_relation_arities,
                trace_predicate_count=trace_predicate_count,
            )
        )
    return CompiledC1F(
        formula=formula if isinstance(formula, str) else render_c1f_formula(formula, function_symbol=function_symbol),
        function_symbol=function_symbol,
        has_higher_arity_relations=any(arity >= 2 for arity in arities.values()),
        branches=tuple(branches),
    )


def evaluate_c1f_fomc(compiled: CompiledC1F, domain_size: int):
    if domain_size < 0:
        raise ValueError("Domain size must be non-negative")
    total = mpq(0)
    for branch in compiled.branches:
        total += branch.evaluate(domain_size)
    integer_total = int(total)
    if total != mpq(integer_total):
        raise ValueError("Internal error: C1F FOMC result is not an integer")
    try:
        return mpz(integer_total)
    except Exception:
        return integer_total


def c1f_fomc(
    formula: str | C1FFormula,
    domain_size: int,
    *,
    function_symbol: str = "f",
    profile_pruning: bool = True,
    twin_compression: bool = True,
):
    return evaluate_c1f_fomc(
        compile_c1f_fomc(
            formula,
            function_symbol=function_symbol,
            profile_pruning=profile_pruning,
            twin_compression=twin_compression,
        ),
        domain_size,
    )


def compile_c1f_ufomc(
    formula: str | C1FFormula,
    *,
    function_symbol: str = "f",
    profile_pruning: bool = True,
    twin_compression: bool = True,
) -> CompiledC1F:
    _validate_identifier(function_symbol, "Function symbol")
    parsed = (
        parse_c1f_sentence(formula, function_symbol=function_symbol)
        if isinstance(formula, str)
        else formula
    )
    validate_c1f_formula(parsed, require_closed=True, function_symbol=function_symbol)
    arities = c1f_predicate_arities(parsed)
    non_monadic = sorted(
        (predicate, arity)
        for predicate, arity in arities.items()
        if arity >= 2
    )
    if non_monadic:
        rendered = ", ".join(
            f"{predicate}/{arity}" for predicate, arity in non_monadic
        )
        raise ValueError(
            "Exact unlabeled C1=[f] counting supports only nullary and unary "
            f"relation symbols; found {rendered}"
        )
    return compile_c1f_fomc(
        parsed,
        function_symbol=function_symbol,
        profile_pruning=profile_pruning,
        twin_compression=twin_compression,
    )


def evaluate_c1f_ufomc(compiled: CompiledC1F, domain_size: int):
    if domain_size < 0:
        raise ValueError("Domain size must be non-negative")
    if compiled.has_higher_arity_relations:
        raise ValueError(
            "Exact unlabeled C1=[f] counting supports only nullary and unary "
            "relation symbols"
        )
    total = mpq(0)
    for branch in compiled.branches:
        total += branch.evaluate_unlabeled(domain_size)
    integer_total = int(total)
    if total != mpq(integer_total):
        raise ValueError("Internal error: C1F UFOMC result is not an integer")
    try:
        return mpz(integer_total)
    except Exception:
        return integer_total


def c1f_ufomc(
    formula: str | C1FFormula,
    domain_size: int,
    *,
    function_symbol: str = "f",
    profile_pruning: bool = True,
    twin_compression: bool = True,
):
    return evaluate_c1f_ufomc(
        compile_c1f_ufomc(
            formula,
            function_symbol=function_symbol,
            profile_pruning=profile_pruning,
            twin_compression=twin_compression,
        ),
        domain_size,
    )
