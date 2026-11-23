"""An HTML report of a run's saved results. CPU.

    python scripts/report.py --config <extraction config>.yaml

- Writes results/<run_id>/report.html. See `emotion_neurons.reports`.
"""

import sys

from emotion_neurons.config import ExtractionRunConfig, load_run_config
from emotion_neurons.reports import REPORTS, write_report


def main():
    cfg = load_run_config(ExtractionRunConfig, sys.argv[1:])
    report = REPORTS[cfg.extraction.kind]
    path = write_report(report.template, report.build(cfg), cfg.result_dir / "report.html")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
