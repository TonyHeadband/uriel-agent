"""Turn every saved eval run into one self-contained HTML page: a scenario-by-run heatmap of pass rates,
the selected run's table with the change from the run before it, and each failing run's turns.

    uv run python -m evals.report [--results evals/results] [--out evals/results/report.html]
"""

import argparse
import json
from pathlib import Path

from evals.harness import Check, Run, TurnRun, summarise

HERE = Path(__file__).parent
TEMPLATE = HERE / "report_template.html"
DATA_SLOT = "/*REPORT_DATA*/null"


def load_run(raw: dict) -> Run:
    fields = TurnRun.__dataclass_fields__
    turns = [TurnRun(**{k: v for k, v in t.items() if k in fields}) for t in raw["turns"]]
    checks = [Check(**c) for c in raw["checks"]]
    return Run(raw["scenario"], turns, checks, raw.get("label", ""), raw.get("model", ""))


def build(results: Path) -> dict:
    files = []
    for path in sorted(results.glob("*.json")):  # timestamped names sort chronologically
        raw = json.loads(path.read_text())
        runs = [load_run(r) for r in raw]
        files.append(
            {
                "id": path.stem,
                "label": runs[0].label if runs else "",
                "rows": summarise(runs),
                "runs": [r | {"failures": run.failures()} for r, run in zip(raw, runs, strict=True)],
            }
        )
    return {"files": files}


def render(data: dict) -> str:
    # "</" would end the inline <script> early if a reply or argument happened to contain "</script>".
    payload = json.dumps(data).replace("</", "<\\/")
    return TEMPLATE.read_text().replace(DATA_SLOT, payload)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=None, help="default: report.html in the results directory")
    args = ap.parse_args()
    results = Path(args.results)
    out = Path(args.out) if args.out else results / "report.html"
    data = build(results)
    if not data["files"]:
        raise SystemExit(f"no result files in {results}; run `python -m evals.run` first")
    out.write_text(render(data))
    print(f"{len(data['files'])} result file(s) -> {out}")


if __name__ == "__main__":
    main()
