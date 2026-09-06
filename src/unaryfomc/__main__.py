"""Command-line interface for UnaryFOMC."""

from __future__ import annotations

import argparse
from typing import Sequence

from .c1f import compile_c1f_fomc, compile_c1f_ufomc
from .mso import MSOCompilerLimits, compile_mso


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unaryfomc",
        description="Count finite models with one unary function (guarded C1 or monadic MSO).",
    )
    parser.add_argument("formula", help="closed logical sentence")
    parser.add_argument("domain_size", type=int, help="size of the domain")
    parser.add_argument("--logic", choices=("c1f", "mso"), default="c1f",
                        help="counting backend (default: c1f)")
    parser.add_argument("--compiler-statistics", action="store_true",
                        help="print MSO compilation statistics to stderr")
    parser.add_argument("--mso-max-states", type=int, default=None,
                        help="override the intermediate MSO state limit")
    parser.add_argument("--mso-max-transitions", type=int, default=None,
                        help="override the MSO compilation work limit")
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.logic == "mso":
        if args.profile_statistics or args.no_profile_pruning or args.no_twin_compression:
            parser.error("profile flags apply only to --logic c1f")
        overrides = {name: getattr(args, "mso_" + name) for name in ("max_states", "max_transitions")
                     if getattr(args, "mso_" + name) is not None}
        if any(value < 1 for value in overrides.values()):
            parser.error("MSO compilation limits must be positive")
        compiled = compile_mso(args.formula, function_symbol=args.function_symbol,
                               limits=MSOCompilerLimits(**overrides))
    else:
        if args.compiler_statistics or args.mso_max_states is not None or args.mso_max_transitions is not None:
            parser.error("MSO compiler flags require --logic mso")
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
    if args.compiler_statistics:
        import sys

        print(compiled.statistics, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
