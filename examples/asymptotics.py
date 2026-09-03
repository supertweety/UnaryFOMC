"""Run representative labeled, unlabeled, and periodic asymptotic analyses."""

from __future__ import annotations

import argparse
from typing import Sequence

from unaryfomc import c1f_labeled_asymptotics, c1f_unlabeled_asymptotics


LABELED_EXAMPLES = {
    "unrestricted endofunctions": "TRUE",
    "bounded indegree two": "forall x: exists<=2 y: (f(y) = x & TRUE)",
    "two Q-predecessors at every P-vertex": (
        "forall x: (P(x) -> exists=2 y: (f(y) = x & Q(y)))"
    ),
}

PERIODIC = (
    "forall x: ((exists=0 y: (f(y) = x & TRUE)) | "
    "(exists=3 y: (f(y) = x & TRUE)))"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--certify",
        action="store_true",
        help="attach Arb interval certificates (requires the certify extra)",
    )
    args = parser.parse_args(argv)

    print("labeled")
    print("example\tregime\tgrowth\tpower\tconstant\tdominance")
    for name, sentence in LABELED_EXAMPLES.items():
        result = c1f_labeled_asymptotics(
            sentence,
            certify=args.certify,
            expansion_order=2,
        )
        dominance = (
            result.certificate.dominance_certified
            if result.certificate is not None
            else "not requested"
        )
        print(
            f"{name}\t{result.regime}\t{result.exponential_growth:.10g}\t"
            f"{result.power:g}\t{result.leading_constant:.10g}\t{dominance}"
        )

    periodic = c1f_labeled_asymptotics(
        PERIODIC,
        certify=args.certify,
        expansion_order=2,
    )
    print("\nperiodic")
    print("period", periodic.periodicity)
    print("residue powers", periodic.residue_powers)
    print("estimates at n=100,102", periodic.estimate(100), periodic.estimate(102))

    unlabeled = c1f_unlabeled_asymptotics(
        "TRUE",
        certify=args.certify,
        expansion_order=2,
    )
    print("\nunlabeled endofunctions")
    print("growth", unlabeled.exponential_growth)
    print("power", unlabeled.power)
    print("constant", unlabeled.leading_constant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
