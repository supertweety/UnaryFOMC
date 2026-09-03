"""Synchronize the allowlisted UnaryFOMC sources from a WFOMC checkout."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Sequence


Transform = Callable[[str], str]


@dataclass(frozen=True)
class Export:
    source: str
    destination: str
    transform: Transform


def transform_module(text: str) -> str:
    old = "from .tail_signature_opt27 import mpq, mpz"
    new = "from .arithmetic import mpq, mpz"
    if text.count(old) != 1:
        raise ValueError(f"expected one private arithmetic import, found {text.count(old)}")
    text = text.replace(old, new)
    text = remove_section(
        text,
        "def _falling_factorial",
        "@dataclass(frozen=True)\nclass _CardinalityMarker",
    )
    text = remove_section(
        text,
        "    def evaluate_population",
        "\n\ndef _compile_monadic_c1f",
    )
    replacements = (
        (
            "    def evaluate(self, domain_size: int, evaluation_backend: str):\n"
            "        if evaluation_backend == \"egf\":\n"
            "            monadic_count = mpq(self.monadic.evaluate_egf(domain_size))\n"
            "        elif evaluation_backend == \"population\":\n"
            "            monadic_count = mpq(self.monadic.evaluate_population(domain_size))\n"
            "        else:\n"
            "            raise ValueError(f\"Unsupported C1F evaluation backend {evaluation_backend!r}\")\n",
            "    def evaluate(self, domain_size: int):\n"
            "        monadic_count = mpq(self.monadic.evaluate_egf(domain_size))\n",
        ),
        (
            "    branches: Tuple[_CompiledC1FBranch, ...]\n"
            "    evaluation_backend: str = \"egf\"\n\n"
            "    def fomc(self, domain_size: int, *, evaluation_backend: Optional[str] = None):\n"
            "        return evaluate_c1f_fomc(\n"
            "            self,\n"
            "            domain_size,\n"
            "            evaluation_backend=evaluation_backend,\n"
            "        )\n",
            "    branches: Tuple[_CompiledC1FBranch, ...]\n\n"
            "    def fomc(self, domain_size: int):\n"
            "        return evaluate_c1f_fomc(self, domain_size)\n",
        ),
        (
            "    twin_compression: bool = True,\n"
            "    evaluation_backend: str = \"egf\",\n"
            ") -> CompiledC1F:\n"
            "    _validate_identifier(function_symbol, \"Function symbol\")\n"
            "    if evaluation_backend not in {\"egf\", \"population\"}:\n"
            "        raise ValueError(\n"
            "            \"C1F evaluation_backend must be 'egf' or 'population'\"\n"
            "        )\n",
            "    twin_compression: bool = True,\n"
            ") -> CompiledC1F:\n"
            "    _validate_identifier(function_symbol, \"Function symbol\")\n",
        ),
        (
            "        branches=tuple(branches),\n"
            "        evaluation_backend=evaluation_backend,\n",
            "        branches=tuple(branches),\n",
        ),
        (
            "def evaluate_c1f_fomc(\n"
            "    compiled: CompiledC1F,\n"
            "    domain_size: int,\n"
            "    *,\n"
            "    evaluation_backend: Optional[str] = None,\n"
            "):\n"
            "    if domain_size < 0:\n"
            "        raise ValueError(\"Domain size must be non-negative\")\n"
            "    backend = compiled.evaluation_backend if evaluation_backend is None else evaluation_backend\n"
            "    if backend not in {\"egf\", \"population\"}:\n"
            "        raise ValueError(\"C1F evaluation_backend must be 'egf' or 'population'\")\n",
            "def evaluate_c1f_fomc(compiled: CompiledC1F, domain_size: int):\n"
            "    if domain_size < 0:\n"
            "        raise ValueError(\"Domain size must be non-negative\")\n",
        ),
        (
            "        total += branch.evaluate(domain_size, backend)\n",
            "        total += branch.evaluate(domain_size)\n",
        ),
        (
            "    twin_compression: bool = True,\n"
            "    evaluation_backend: str = \"egf\",\n"
            "):\n",
            "    twin_compression: bool = True,\n"
            "):\n",
        ),
        (
            "            twin_compression=twin_compression,\n"
            "            evaluation_backend=evaluation_backend,\n",
            "            twin_compression=twin_compression,\n",
        ),
    )
    for before, after in replacements:
        if text.count(before) != 1:
            raise ValueError(
                "private C1F layout changed; expected one public-export pattern, "
                f"found {text.count(before)}"
            )
        text = text.replace(before, after)
    return text


def remove_section(text: str, start: str, end: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(
            f"private C1F layout changed around {start!r} or {end!r}"
        )
    before, remainder = text.split(start, 1)
    _removed, after = remainder.split(end, 1)
    return before + end + after


def preserve_module(text: str) -> str:
    return text


EXPORTS = (
    Export("src/wfomc/c1f.py", "src/unaryfomc/c1f.py", transform_module),
    Export(
        "src/wfomc/specification.py",
        "src/unaryfomc/specification.py",
        preserve_module,
    ),
    Export(
        "src/wfomc/asymptotics.py",
        "src/unaryfomc/asymptotics.py",
        preserve_module,
    ),
    Export(
        "src/wfomc/interval_asymptotics.py",
        "src/unaryfomc/interval_asymptotics.py",
        preserve_module,
    ),
    Export(
        "src/wfomc/puiseux.py",
        "src/unaryfomc/puiseux.py",
        preserve_module,
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    source_root = args.source.resolve()
    destination_root = Path(__file__).resolve().parents[1]
    stale = []
    for export in EXPORTS:
        source = source_root / export.source
        destination = destination_root / export.destination
        if not source.is_file():
            raise FileNotFoundError(f"allowlisted source does not exist: {source}")
        expected = export.transform(source.read_text(encoding="utf-8"))
        actual = destination.read_text(encoding="utf-8") if destination.exists() else None
        if actual == expected:
            print(f"unchanged {export.destination}")
            continue
        stale.append(export.destination)
        if args.apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(expected, encoding="utf-8")
            print(f"updated   {export.destination}")
        else:
            print(f"stale     {export.destination}")

    if stale and args.check:
        print("run again with --apply to update allowlisted files", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
