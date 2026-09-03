"""Command-line interface for UnaryFOMC."""

from __future__ import annotations

import argparse
from typing import Sequence

from .c1f import compile_c1f_fomc, compile_c1f_ufomc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unaryfomc",
        description="Count finite models of a C1=[f] sentence.",
    )
    parser.add_argument("formula", help="closed one-variable sentence")
    parser.add_argument("domain_size", type=int, help="size of the labeled domain")
    parser.add_argument(
        "--function-symbol",
        default="f",
        help="unary function symbol used in the sentence (default: f)",
    )
    parser.add_argument(
        "--unlabeled",
        action="store_true",
        help="count isomorphism classes (monadic relational vocabularies only)",
    )
    parser.add_argument(
        "--no-profile-pruning",
        action="store_true",
        help="disable universal-profile pruning",
    )
    parser.add_argument(
        "--no-twin-compression",
        action="store_true",
        help="disable exact strong-twin profile compression",
    )
    parser.add_argument(
        "--profile-statistics",
        action="store_true",
        help="print profile counts to stderr",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    compiler = compile_c1f_ufomc if args.unlabeled else compile_c1f_fomc
    compiled = compiler(
        args.formula,
        function_symbol=args.function_symbol,
        profile_pruning=not args.no_profile_pruning,
        twin_compression=not args.no_twin_compression,
    )
    count = (
        compiled.ufomc(args.domain_size)
        if args.unlabeled
        else compiled.fomc(args.domain_size)
    )
    print(count)
    if args.profile_statistics:
        import sys

        for branch, statistics in enumerate(compiled.profile_statistics()):
            print(f"branch {branch}: {statistics}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
