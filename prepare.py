"""
Compatibility helper for the hybrid autoresearch-mlx / UCMD baseline repo.

The canonical baseline now lives entirely in train.py, so no mandatory prepare
step remains. This script can optionally materialize a synthetic dataset
snapshot outside the repo if UCMD_DATA or UCMD_WRITE_DATA is set.
"""

import os


def main():
    target = os.getenv("UCMD_DATA", "")

    if not target and os.getenv("UCMD_WRITE_DATA", "0") == "1":
        from train import default_dataset_path

        target = default_dataset_path()

    if target:
        from train import Config, build_samples

        config = Config()
        config.dataset_path = target
        samples = build_samples(config)
        print(f"UCMD synthetic dataset ready at: {target}")
        print(f"samples: {len(samples)}")
        return

    print("No separate prepare step is required for the UCMD baseline.")
    print("Run `uv run train.py` to generate synthetic supervision, train, evaluate, and log metrics.")
    print("Optional: set `UCMD_DATA=/path/to/ucmd.jsonl` before running if you want a saved dataset snapshot.")


if __name__ == "__main__":
    main()
