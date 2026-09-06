"""Exact labeled and unlabeled sequences for fixed monadic MSO sentences."""

from __future__ import annotations

import argparse

from unaryfomc import compile_mso


EXAMPLES = {
    "even_cycles": "exists X: forall x: (X(f(x)) <-> ~X(x))",
    "rooted_trees": (
        "exists r: (f(r)=r & (forall x: forall X: "
        "((X(x) & (forall y: (X(y) -> X(f(y))))) -> X(r))))"
    ),
    "one_directed_cycle": (
        "(exists x: x=x) & (forall x: forall y: forall X: "
        "((X(x) & (forall z: (X(z) -> X(f(z))))) -> X(y)))"
    ),
    "connected_functions": (
        "(exists x: x=x) & (forall X: (((forall x: (X(x) <-> X(f(x)))) & "
        "(exists x: X(x))) -> (forall x: X(x))))"
    ),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=12)
    args = parser.parse_args(argv)
    if args.max_n < 0:
        parser.error("--max-n must be nonnegative")
    for name, sentence in EXAMPLES.items():
        compiled = compile_mso(sentence)
        print(name)
        print("  states (F,H,D,G):", compiled.statistics.branch_state_counts)
        print("  labeled:  ", compiled.sequence(args.max_n))
        print("  unlabeled:", compiled.sequence(args.max_n, labeled=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
