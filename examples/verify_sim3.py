"""Reproduce a ten-landmark alignment without models, Open3D windows or CUDA."""

import json
from pathlib import Path

import numpy as np

from gs_merger.gaussian_io import split_similarity
from gs_merger.similarity import fit_control_points


def main():
    example = json.loads(Path(__file__).with_name("ten_landmarks.json").read_text())
    transform, mask, residuals = fit_control_points(
        example["source"], example["target"],
        threshold=example["control_threshold"], seed=example["seed"],
    )
    np.testing.assert_allclose(transform, example["expected_transform"], atol=1e-12)
    scale, _, _ = split_similarity(transform)
    print(f"scale={scale:.12f}; inliers={int(mask.sum())}/{len(mask)}")
    print(f"RMSE={np.sqrt(np.mean(residuals ** 2)):.12f} (reference coordinate units)")
    print("Saved transform reproduced successfully.")


if __name__ == "__main__":
    main()
