from __future__ import annotations

from contextlib import redirect_stdout
from inspect import signature
from io import StringIO
from pathlib import Path
import subprocess
import sys
import unittest

import unaryfomc
from unaryfomc import compile_c1f_fomc
from unaryfomc.__main__ import main


REPO_ROOT = Path(__file__).resolve().parents[1]


class PublicRepositoryTests(unittest.TestCase):
    def test_public_asymptotic_api_is_exported(self):
        expected = {
            "C1FAsymptoticCertificate",
            "C1FAsymptoticCertificationError",
            "C1FAsymptoticTerm",
            "C1FCertifiedAsymptoticTerm",
            "C1FLabeledAsymptotics",
            "C1FRealInterval",
            "C1FUnlabeledAsymptotics",
            "c1f_labeled_asymptotics",
            "c1f_unlabeled_asymptotics",
        }
        self.assertTrue(expected.issubset(set(unaryfomc.__all__)))

    def test_private_population_backend_is_not_exported(self):
        self.assertNotIn("evaluation_backend", signature(compile_c1f_fomc).parameters)

    def test_cli_prints_exact_count(self):
        output = StringIO()
        with redirect_stdout(output):
            status = main(["forall x: x = f(x)", "5"])
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue(), "1\n")

    def test_cli_prints_unlabeled_count(self):
        output = StringIO()
        with redirect_stdout(output):
            status = main(["TRUE", "6", "--unlabeled"])
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue(), "130\n")

    def test_private_export_is_in_sync(self):
        private_repo = REPO_ROOT.parent / "WFOMC"
        if not private_repo.is_dir():
            self.skipTest("private WFOMC checkout is not adjacent")
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "sync_from_wfomc.py"),
                "--source",
                str(private_repo),
                "--check",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
