"""Print small exact unlabeled C1=[f] counting sequences."""

from __future__ import annotations

import argparse
from typing import Sequence

from unaryfomc import compile_c1f_ufomc


EXAMPLES = {
    "endofunctions": "TRUE",
    "unary_coloured_identity": (
        "forall x: f(x) = x & "
        "forall x: (P(x) | ~P(x))"
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=15)
    args = parser.parse_args(argv)
    if args.max_n < 0:
        parser.error("--max-n must be non-negative")

    compiled = {
        name: compile_c1f_ufomc(sentence)
        for name, sentence in EXAMPLES.items()
    }
    print("n\t" + "\t".join(compiled))
    for n in range(args.max_n + 1):
        values = [str(int(problem.ufomc(n))) for problem in compiled.values()]
        print(f"{n}\t" + "\t".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
