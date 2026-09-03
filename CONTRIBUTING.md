# Contributing

UnaryFOMC is a focused public extraction of the unary-function model-counting
implementation used in the accompanying paper.

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

The optional `speed` extra installs GMP-backed exact arithmetic:

```bash
python -m pip install -e '.[speed]'
```

The optional `certify` extra installs Arb through `python-flint` on Python
3.11 or newer:

```bash
python -m pip install -e '.[certify]'
```

## Synchronizing research changes

When an implementation change originates in the private WFOMC research
repository, synchronize only the allowlisted unary-function modules:

```bash
python tools/sync_from_wfomc.py --source ../WFOMC --check
python tools/sync_from_wfomc.py --source ../WFOMC --apply
```

After applying an update:

1. Inspect the complete Git diff.
2. Confirm that only the intended unary-function code changed.
3. Port relevant regression tests deliberately; tests are not copied by the
   synchronizer.
4. Run the complete test suite.
5. Run `examples/paper_examples.py` and compare the overlapping values with the
   paper.
6. Run `examples/unlabeled_counts.py` and verify the initial unlabeled mapping
   sequence.
7. Check that no private paths, credentials, generated data, or unrelated
   experimental code entered the public tree.

Do not copy the private repository's Git history into this repository.

## Pull requests

Keep changes narrowly scoped and include a regression test for behavioral
changes. Counts must remain exact. Performance improvements should report the
sentences, domain sizes, arithmetic backend, and machine used for comparison.
