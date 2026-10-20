"""Positive-scale Umeyama fitting and RANSAC for paired manual landmarks."""


import numpy as np


def estimate_similarity(source, target, estimate_scale=True):
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Control points must be paired arrays with shape (N, 3)")
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("At least three finite corresponding control points are required")
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    centered_source, centered_target = source - source_mean, target - target_mean
    for points in (centered_source, centered_target):
        singular = np.linalg.svd(points, compute_uv=False)
        if singular[0] <= 1e-12 or singular[1] <= singular[0] * 1e-6:
            raise ValueError("Control points are coincident or collinear; use spread-out landmarks")
    covariance = centered_target.T @ centered_source / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    variance = np.mean(np.sum(centered_source**2, axis=1))
    scale = float(singular @ sign / variance) if estimate_scale else 1.0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Cannot estimate a positive uniform scale")
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = target_mean - scale * rotation @ source_mean
    return transform


def fit_control_points(source, target, threshold, estimate_scale=True, seed=123, iterations=1000):
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    estimate_similarity(source, target, estimate_scale)  # Validate shape and geometry first.
    if not np.isfinite(threshold) or threshold <= 0 or iterations < 1:
        raise ValueError("Control threshold and RANSAC iterations must be positive")
    rng = np.random.default_rng(seed)
    best_mask, best_score = None, (-1, -np.inf)
    for _ in range(iterations):
        sample = rng.choice(len(source), 3, replace=False)
        try:
            transform = estimate_similarity(source[sample], target[sample], estimate_scale)
        except ValueError:
            continue
        residuals = np.linalg.norm(source @ transform[:3, :3].T + transform[:3, 3] - target, axis=1)
        mask = residuals <= threshold
        score = (int(mask.sum()), -float(residuals[mask].mean()) if mask.any() else -np.inf)
        if score > best_score:
            best_mask, best_score = mask, score
        if mask.all():
            break
    if best_mask is None or best_mask.sum() < 3:
        raise ValueError("No consistent, noncollinear control-point alignment found")
    for _ in range(10):
        transform = estimate_similarity(source[best_mask], target[best_mask], estimate_scale)
        residuals = np.linalg.norm(source @ transform[:3, :3].T + transform[:3, 3] - target, axis=1)
        mask = residuals <= threshold
        if mask.sum() < 3:
            raise ValueError("Control-point refinement lost support")
        if np.array_equal(mask, best_mask):
            break
        best_mask = mask
    transform = estimate_similarity(source[best_mask], target[best_mask], estimate_scale)
    residuals = np.linalg.norm(source @ transform[:3, :3].T + transform[:3, 3] - target, axis=1)
    return transform, residuals <= threshold, residuals
