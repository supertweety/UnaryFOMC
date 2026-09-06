"""Five compact MSO counting examples with global function-graph properties."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from unaryfomc import compile_mso


def reach(x, y):
    return (f"(forall ReachSet: ((ReachSet({x}) & "
            "(forall w: (ReachSet(w) -> ReachSet(f(w))))) "
            f"-> ReachSet({y})))")


def same_component(x, y):
    return (f"(forall Component: ((Component({x}) & "
            "(forall w: (Component(w) <-> Component(f(w))))) "
            f"-> Component({y})))")


ROOTED_TREE = f"exists r: (f(r)=r & (forall x: {reach('x', 'r')}))"
EVEN_LEAVES = (
    "exists A: exists r: (f(r)=r & A(r) & "
    f"(forall x: {reach('x', 'r')}) & "
    "(forall x: (f(x)!=x -> (A(x) <-> ~A(f(x))))) & "
    "(forall x: ((forall y: (f(y)=x -> y=x)) -> A(x))))"
)
TWO_COMPONENTS = (
    f"exists x,y: (~{same_component('x', 'y')} & "
    f"(forall z: ({same_component('x', 'z')} | {same_component('y', 'z')})))"
)
THREE_PHASES = (
    "exists A,B,C: forall x: ((A(x) | B(x) | C(x)) & "
    "~(A(x) & B(x)) & ~(A(x) & C(x)) & ~(B(x) & C(x)) & "
    "(A(x) -> B(f(x))) & (B(x) -> C(f(x))) & (C(x) -> A(f(x))))"
)
CYCLE_TRANSVERSAL = (
    f"forall x: exists p: (P(p) & {reach('x', 'p')} & "
    f"(forall q: ((P(q) & {reach('x', 'q')}) -> q=p)))"
)

EXAMPLES = {
    "rooted_trees": ROOTED_TREE,
    "rooted_trees_even_depth_leaves": EVEN_LEAVES,
    "exactly_two_components": TWO_COMPONENTS,
    "cycle_lengths_divisible_by_three": THREE_PHASES,
    "one_marked_cycle_vertex_per_component": CYCLE_TRANSVERSAL,
}


def run_example(name, sentence, bound):
    started = perf_counter()
    compiled = compile_mso(sentence)
    compile_seconds = perf_counter() - started
    started = perf_counter()
    labeled = compiled.sequence(bound)
    unlabeled = compiled.sequence(bound, labeled=False)
    return dict(name=name, sentence=sentence, bound=bound, labeled=labeled,
                unlabeled=unlabeled, compile_seconds=compile_seconds,
                evaluation_seconds=perf_counter() - started,
                statistics=asdict(compiled.statistics))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.max_n < 0:
        parser.error("--max-n must be nonnegative")
    results = []
    for name, sentence in EXAMPLES.items():
        result = run_example(name, sentence, args.max_n)
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
