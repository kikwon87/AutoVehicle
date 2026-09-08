"""Command-line entry point: ``avsim <command>``.

::

    avsim list                      list the scenarios
    avsim run signal_red --plot out.png --json out.json
    avsim suite --out results/      run everything, write reports and figures
    avsim study 3                   run one of the lecture's study tasks
    avsim contract                  print the model contract
    avsim platform                  start the browser test platform
    avsim platform --headless --preset unprotected_left --controller mpc
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


def _cmd_platform(args) -> int:
    """Start the test platform, or run one of its presets without a browser.

    ``--headless`` exists so a controller can be scored from a script or a CI
    job: the platform's value is the comparison, and a comparison that only
    happens in a browser cannot be automated.
    """
    if not args.headless:
        from .platform.server import serve

        serve(host=args.host, port=args.port, open_browser=not args.no_browser)
        return 0

    from .platform.session import RunConfig, RunSession

    config = RunConfig(
        preset=args.preset,
        controller=args.controller,
        seed=args.seed,
        n_vehicles=args.n_vehicles,
        duration=args.duration,
    )
    result = RunSession(config).run()
    print(f"{result.preset} / {result.controller}: {result.finish_reason} "
          f"after {result.duration:.1f} s")
    for key, value in sorted(result.metrics.items()):
        if isinstance(value, float):
            print(f"  {key:24s} {value:10.3f}")
        else:
            print(f"  {key:24s} {value!s:>10}")
    print(f"  {'SCORE':24s} {result.score.total:10.1f}")
    for cat, value in result.score.categories.items():
        print(f"    {cat:22s} {value:10.1f}")
    for note in result.score.notes:
        print(f"  note: {note}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(include_log=args.log), fh, indent=2, default=str)
        print(f"wrote {args.json}")
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

    p_platform = sub.add_parser("platform", help="the browser test platform")
    p_platform.add_argument("--host", default="127.0.0.1")
    p_platform.add_argument("--port", type=int, default=8770)
    p_platform.add_argument("--no-browser", action="store_true",
                            help="do not open a browser window")
    p_platform.add_argument("--headless", action="store_true",
                            help="run one preset and print its KPIs instead of serving")
    p_platform.add_argument("--preset", default="grid_random")
    p_platform.add_argument("--controller", default="mpc",
                            help="a builtin key, or the path to a .py plug-in")
    p_platform.add_argument("--seed", type=int, default=0)
    p_platform.add_argument("--n-vehicles", type=int, default=None, dest="n_vehicles")
    p_platform.add_argument("--duration", type=float, default=None)
    p_platform.add_argument("--json", help="write the result as JSON")
    p_platform.add_argument("--log", action="store_true",
                            help="include the per-tick log in --json")
    p_platform.set_defaults(func=_cmd_platform)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
