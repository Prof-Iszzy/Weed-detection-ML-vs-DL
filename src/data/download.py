"""
Download DeepWeeds (primary) and MFWD (secondary, cross-dataset eval).
Phase 2, Step 9 of project_execution_algorithm.md.

TODO (Phase 2):
- DeepWeeds: https://github.com/AlexOlsen/DeepWeeds
- MFWD: https://doi.org/10.5281/zenodo.7598372
"""

import argparse


def download_deepweeds(output_dir: str) -> None:
    raise NotImplementedError("Fill in during Phase 2 — data acquisition")


def download_mfwd(output_dir: str) -> None:
    raise NotImplementedError("Fill in during Phase 2 — data acquisition")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    download_deepweeds(f"{args.output_dir}/deepweeds")
    download_mfwd(f"{args.output_dir}/mfwd")
