"""Reproduce the numerical examples from arXiv:2608.30580."""

from __future__ import annotations

import argparse
from math import comb
from typing import Iterable, Sequence

from unaryfomc import c1f_fomc, compile_c1f_fomc


def iterate_f(depth: int) -> str:
    term = "x"
    for _ in range(depth):
        term = f"f({term})"
    return term


def prescribed_three_cycles(domain_size: int) -> int:
    witness_count = 3 * (domain_size // 3)
    return int(
        c1f_fomc(
            f"exists={witness_count} x: (f(f(f(x))) = x & f(x) != x)",
            domain_size,
        )
    )


def compile_quasi_kernel(size: int):
    return compile_c1f_fomc(
        f"exists={size} x: Q(x) & "
        "forall x: (Q(x) -> ~Q(f(x))) & "
        "forall x: (~Q(x) -> (Q(f(x)) | Q(f(f(x))))) & "
        "forall x: f(f(x)) != x & "
        "forall x: f(f(f(x))) != x"
    )


def compile_bounded_height(height: int):
    return compile_c1f_fomc(
        f"forall x: {iterate_f(height + 1)} = {iterate_f(height)} & "
        "forall x: (f(f(x)) != f(x) -> "
        "(B(x) <-> (B(f(x)) | B(f(f(x))))))"
    )


def domain_sizes(maximum: int) -> Iterable[int]:
    return range(1, maximum + 1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=15)
    args = parser.parse_args(argv)
    if args.max_n < 1:
        parser.error("--max-n must be positive")

    print("example_2_prescribed_three_cycles")
    print("n\tfomc")
    for n in domain_sizes(args.max_n):
        print(f"{n}\t{prescribed_three_cycles(n)}")

    print("\nexample_3_prescribed_quasi_kernel")
    quasi = {k: compile_quasi_kernel(k) for k in (2, 3)}
    print("n\tk=2\tk=3")
    for n in domain_sizes(args.max_n):
        values = []
        for k in (2, 3):
            if n < k:
                values.append("-")
                continue
            labeled = int(quasi[k].fomc(n))
            divisor = comb(n, k)
            if labeled % divisor:
                raise ArithmeticError("quasi-kernel normalization is not integral")
            values.append(str(labeled // divisor))
        print(f"{n}\t" + "\t".join(values))

    print("\nexample_4_bounded_height_forests")
    forests = {k: compile_bounded_height(k) for k in (2, 3, 4, 5)}
    print("n\tk=2\tk=3\tk=4\tk=5")
    for n in domain_sizes(args.max_n):
        values = [str(int(forests[k].fomc(n))) for k in forests]
        print(f"{n}\t" + "\t".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
