"""A linear-size sentence family forcing nonelementary composition algebras.

Each run has both compilation-work limits and a separate wall-clock limit.
See docs/mso-compilation-stress.md for the state lower-bound proof.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

from unaryfomc import MSOCompilationLimitError, MSOCompilerLimits, MSOSpecification, compile_mso


def child(vertex, parent):
    return f"(f({vertex})={parent} & {vertex}!={parent})"


def branch_equivalence(depth, pair=("x", "y")):
    """Compare sets of depth-(depth-1) child types, ignoring multiplicity.

    The recursive subformula occurs once, and four variable names suffice.
    Inner binders deliberately shadow variables no longer used there.
    """
    if type(depth) is not int or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    if depth == 0:
        return "TRUE"
    x, y = pair
    u, v = ("u", "v") if pair == ("x", "y") else ("x", "y")
    if depth == 1:
        # The first comparison only distinguishes leaves from nonleaves.
        return f"((exists {u}: {child(u, x)}) <-> (exists {v}: {child(v, y)}))"
    opposite = (f"(({child(u, x)} & {child(v, y)}) | "
                f"({child(u, y)} & {child(v, x)}))")
    return (f"(forall {u}: (({child(u, x)} | {child(u, y)}) -> "
            f"(exists {v}: ({opposite} & {branch_equivalence(depth - 1, (u, v))}))))")


def duplicate_branches_sentence(depth):
    """Two distinct fixed points whose incoming trees have equal depth types."""
    return ("exists x,y: (x!=y & f(x)=x & f(y)=y & "
            f"{branch_equivalence(depth)})")


def graph_state_lower_bound(depth):
    """Return a compact expression; never materialize enormous tower integers."""
    if type(depth) is not int or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    value = 1
    for level in range(depth + 1):
        if value > 16:
            expression = str(value)
            for _ in range(depth + 1 - level):
                expression = f"2^({expression})"
            return expression + "+1"
        value = 1 << value
    return str(value + 1)


def duplicate_branches_specification(depth, *, max_graph_states=1000):
    """An independent, specialized specification, NOT the sentence compiler.

    This realizes the graph-state bound and checks the generic compiler on
    small instances. Refuse large tables before constructing them.
    """
    if type(depth) is not int or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    sizes = [1]
    for _ in range(depth):
        if sizes[-1] >= max_graph_states.bit_length():
            raise ValueError("The direct specification exceeds its graph-state limit")
        sizes.append(1 << sizes[-1])
    types = sizes[-1]
    if types >= max_graph_states.bit_length() or (1 << types) + 1 > max_graph_states:
        raise ValueError("The direct specification exceeds its graph-state limit")

    def truncate(level, code):
        if level == 1:
            return 0
        return sum(1 << value for value in {
            truncate(level - 1, child_code) for child_code in range(sizes[level - 1])
            if code >> child_code & 1})

    duplicate = 1 << types
    long_cycle = types
    forest_add = tuple(tuple(a | b for b in range(types)) for a in range(types))
    forget = tuple(1 << truncate(depth, t) for t in range(types)) if depth else (0,)
    return MSOSpecification(
        predicates=(), forest_add=forest_add, root=(tuple(range(types)),),
        forget=forget, tree_to_dotted=tuple(range(types)),
        dotted_product=tuple((long_cycle,) * (types + 1) for _ in range(types + 1)),
        close=tuple(1 << t for t in range(types)) + (0,),
        graph_add=tuple(tuple(duplicate if a == duplicate or b == duplicate or a & b else a | b
                              for b in range(duplicate + 1)) for a in range(duplicate + 1)),
        accepting=frozenset((duplicate,)))


def _compile_worker(connection, sentence, options):
    started = perf_counter()
    try:
        compiled = compile_mso(sentence, limits=MSOCompilerLimits(**options))
        result = dict(status="compiled", compile_seconds=perf_counter() - started,
                      statistics=asdict(compiled.statistics),
                      labeled_through_4=compiled.sequence(4),
                      unlabeled_through_4=compiled.sequence(4, labeled=False))
    except MSOCompilationLimitError as error:
        result = dict(status="compilation_limit", error=str(error))
    except Exception as error:
        result = dict(status="error", error=f"{type(error).__name__}: {error}")
    result["worker_seconds"] = perf_counter() - started
    connection.send(result)
    connection.close()


def benchmark(depth, *, seconds=15.0, max_states=5000, max_transitions=2000000):
    sentence = duplicate_branches_sentence(depth)
    options = dict(max_states=max_states, max_transitions=max_transitions)
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    worker = context.Process(target=_compile_worker, args=(writer, sentence, options))
    started = perf_counter()
    worker.start()
    writer.close()
    try:
        if reader.poll(seconds):
            try:
                result = reader.recv()
            except EOFError:
                result = dict(status="worker_failed")
        else:
            result = dict(status="wall_time_limit")
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join()
        reader.close()
    return dict(depth=depth, source_characters=len(sentence),
                graph_state_lower_bound=graph_state_lower_bound(depth),
                limits=dict(**options, wall_seconds=seconds),
                wall_seconds=perf_counter() - started, **result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--max-states", type=int, default=5000)
    parser.add_argument("--max-transitions", type=int, default=2000000)
    parser.add_argument("--formula-only", action="store_true")
    parser.add_argument("--output", type=Path, help="write the same results as a JSON array")
    args = parser.parse_args(argv)
    if (any(d < 0 for d in args.depths) or args.seconds <= 0
            or args.max_states < 1 or args.max_transitions < 1):
        parser.error("depths must be nonnegative and limits positive")
    results = []
    for depth in args.depths:
        if args.formula_only:
            print(duplicate_branches_sentence(depth))
        else:
            result = benchmark(depth, seconds=args.seconds, max_states=args.max_states,
                               max_transitions=args.max_transitions)
            results.append(result)
            print(json.dumps(result), flush=True)
    if args.output:
        if args.formula_only:
            parser.error("--output cannot be combined with --formula-only")
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
