"""End-to-end modes report with checkpoints: harvest -> modes -> judge -> report.

    python scripts/pipeline.py --config configs/harvest_<emotion>_l25.yaml
    python scripts/pipeline.py --config ... --pipeline.until modes                        # stop after a stage
    python scripts/pipeline.py --config ... --pipeline.s3-prefix s3://bucket/emotion-neurons   # checkpoints on S3

- Rerun the same command after a crash or on another machine: finished stages are skipped.
- See `emotion_neurons.pipeline` for the checkpoints and the S3 sync.
"""

import sys

from emotion_neurons.config import ModesRunConfig, load_run_config
from emotion_neurons.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline(load_run_config(ModesRunConfig, sys.argv[1:]))
