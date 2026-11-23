"""An HTML report of a run's saved results. CPU.

    python scripts/report.py emotion_vectors --config <vectors config>.yaml
    python scripts/report.py direction --config <direction config>.yaml

- Writes results/<run_id>/report.html. See `emotion_neurons.reports`.
"""

import sys

from emotion_neurons.config import load_run_config
from emotion_neurons.reports import REPORTS, write_report


def main():
    kind, argv = sys.argv[1], sys.argv[2:]
    assert kind in REPORTS, f"unknown report {kind!r}; one of {sorted(REPORTS)}"
    report = REPORTS[kind]
    cfg = load_run_config(report.config, argv)
    path = write_report(report.template, report.build(cfg), cfg.result_dir / "report.html")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
