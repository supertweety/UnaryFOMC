"""Exact arithmetic with an optional GMP-backed implementation."""

from __future__ import annotations

try:  # pragma: no cover - availability depends on the installation
    import gmpy2

    mpz = gmpy2.mpz
    mpq = gmpy2.mpq
except ImportError:  # pragma: no cover - exercised in dependency-free installs
    from fractions import Fraction

    mpz = int
    mpq = Fraction
