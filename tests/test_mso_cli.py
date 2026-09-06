from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unaryfomc.__main__ import main


class MSOCLITests(unittest.TestCase):
    def test_counts_and_statistics(self):
        for flags, expected in (([], "57"), (["--unlabeled"], "5")):
            out, err = StringIO(), StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                status = main(["--logic", "mso", "--compiler-statistics",
                               "exists X: forall x: (X(f(x)) <-> ~X(x))", "4", *flags])
            self.assertEqual(status, 0)
            self.assertEqual(out.getvalue().strip(), expected)
            self.assertIn("formula-congruence", err.getvalue())

    def test_backend_specific_flags(self):
        for flags in (("--logic", "mso", "--profile-statistics"),
                      ("--compiler-statistics",), ("--mso-max-states", "100"),
                      ("--logic", "mso", "--mso-max-states", "0")):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                main(["TRUE", "1", *flags])
            self.assertEqual(error.exception.code, 2)

    def test_default_c1_backend_unchanged(self):
        out = StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["forall x: f(x)=x", "3"]), 0)
        self.assertEqual(out.getvalue().strip(), "1")


if __name__ == "__main__":
    unittest.main()
