"""Command-line entry point: ``avsim <command>``.

::

    avsim list                      list the scenarios
    avsim run signal_red --plot out.png --json out.json
    avsim suite --out results/      run everything, write reports and figures
    avsim study 3                   run one of the lecture's study tasks
    avsim contract                  print the model contract
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .eval.kpi import summarize
from .eval.runner import run_all, run_scenario
from .eval.scenarios import SCENARIOS
from .eval.studies import STUDIES, run_all_studies


def _cmd_list(args) -> int:
    for name, build in SCENARIOS.items():
        setup = build(0)
        print(f"{name:24s} {setup.duration:5.1f} s   {setup.description}")
        if setup.notes:
            print(f"{'':24s}         note: {setup.notes}")
    return 0


def _write_figures(result, out_dir: str) -> None:
    try:
        from .viz.render import plot_behavior_timeline, plot_run
    except ImportError:
        print("  (matplotlib not installed: skipping figures)", file=sys.stderr)
        return
    os.makedirs(out_dir, exist_ok=True)
    plot_run(result, os.path.join(out_dir, f"{result.setup.name}.png"))
    if result.telemetry:
        plot_behavior_timeline(result, os.path.join(out_dir, f"{result.setup.name}_behavior.png"))


def _cmd_run(args) -> int:
    result = run_scenario(args.scenario, seed=args.seed, verbose=True)
    print(result.report.to_table())
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(result.report.to_json())
        print(f"wrote {args.json}")
    if args.plot:
        from .viz.render import plot_run

        plot_run(result, args.plot)
        print(f"wrote {args.plot}")
    if args.animate:
        from .viz.render import animate_run

        animate_run(result, args.animate)
        print(f"wrote {args.animate}")
    return 0 if result.passed else 1


def _cmd_suite(args) -> int:
    names = args.only.split(",") if args.only else None
    results = run_all(names, seed=args.seed, verbose=True)
    print()
    print(summarize([r.report for r in results]))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        payload = {r.setup.name: r.report.to_dict() for r in results}
        path = os.path.join(args.out, "results.json")
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=float)
        print(f"\nwrote {path}")
        if not args.no_figures:
            for r in results:
                _write_figures(r, args.out)
            try:
                from .viz.render import plot_kpi_matrix

                plot_kpi_matrix([r.report for r in results],
                                os.path.join(args.out, "kpi_matrix.png"))
            except ImportError:
                pass
            print(f"wrote figures to {args.out}/")
    return 0 if all(r.passed for r in results) else 1


def _cmd_study(args) -> int:
    if args.task:
        STUDIES[args.task](verbose=True)
    else:
        run_all_studies(verbose=True)
    return 0


def _cmd_contract(args) -> int:
    from .eval.studies import study_5_model_contract

    study_5_model_contract(verbose=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="avsim", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list scenarios").set_defaults(func=_cmd_list)

    p_run = sub.add_parser("run", help="run one scenario")
    p_run.add_argument("scenario", choices=sorted(SCENARIOS))
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--json", help="write the KPI report as JSON")
    p_run.add_argument("--plot", help="write a summary figure")
    p_run.add_argument("--animate", help="write a GIF")
    p_run.set_defaults(func=_cmd_run)

    p_suite = sub.add_parser("suite", help="run the whole scenario suite")
    p_suite.add_argument("--seed", type=int, default=0)
    p_suite.add_argument("--out", help="directory for reports and figures")
    p_suite.add_argument("--only", help="comma-separated subset of scenarios")
    p_suite.add_argument("--no-figures", action="store_true")
    p_suite.set_defaults(func=_cmd_suite)

    p_study = sub.add_parser("study", help="run a study task from the lecture")
    p_study.add_argument("task", nargs="?", choices=sorted(STUDIES))
    p_study.set_defaults(func=_cmd_study)

    sub.add_parser("contract", help="print the model contract").set_defaults(func=_cmd_contract)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
