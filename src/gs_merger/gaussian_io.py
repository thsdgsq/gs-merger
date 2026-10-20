"""Read and transform standard Graphdeco 3DGS PLYs without losing Gaussian attributes.

Only NumPy and plyfile are needed here; this module and the merge worker do not
load Open3D, learned registration models, PyTorch or CUDA extensions.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


XYZ = ["x", "y", "z"]
ROT = [f"rot_{i}" for i in range(4)]  # w, x, y, z
SCALE = [f"scale_{i}" for i in range(3)]  # natural logarithms
DC = [f"f_dc_{i}" for i in range(3)]
NORMALS = ["nx", "ny", "nz"]
CHUNK_SIZE = 65536


def columns(vertices, names):
    return np.column_stack([vertices[name] for name in names])


def set_columns(vertices, names, values):
    for i, name in enumerate(names):
        vertices[name] = values[:, i]


def sh_degree(vertices):
    """Validate the scalar schema, including channel-major SH field ordering."""
    names = set(vertices.dtype.names or ())
    required = XYZ + ROT + SCALE + DC + ["opacity"]
    missing = set(required) - names
    if missing:
        raise ValueError(f"Not a standard 3DGS PLY; missing fields: {sorted(missing)}")
    for name in names:
        dtype = vertices.dtype.fields[name][0]
        if dtype.kind not in "fiu" or dtype.subdtype:
            raise ValueError(f"Only scalar numeric vertex fields are supported: {name}")
    for name in required + [n for n in NORMALS if n in names]:
        if vertices.dtype.fields[name][0].kind != "f":
            raise ValueError(f"3DGS field must be floating point: {name}")
    for prefix, expected in (("rot_", ROT), ("scale_", SCALE), ("f_dc_", DC)):
        if {n for n in names if n.startswith(prefix)} != set(expected):
            raise ValueError(f"Unexpected {prefix} fields; expected {expected}")
    rest = [name for name in names if name.startswith("f_rest_")]
    if set(rest) != {f"f_rest_{i}" for i in range(len(rest))}:
        raise ValueError("f_rest_* fields must be consecutively numbered from zero")
    root = math.isqrt(len(rest) // 3 + 1)
    degree = root - 1
    if len(rest) != 3 * (root * root - 1) or not 0 <= degree <= 4:
        raise ValueError("Expected a complete RGB SH basis of degree 0 through 4")
    for name in rest:
        if vertices.dtype.fields[name][0].kind != "f":
            raise ValueError(f"SH field must be floating point: {name}")
    if any(n in names for n in NORMALS) and not all(n in names for n in NORMALS):
        raise ValueError("Normals must include nx, ny, and nz together")
    return degree


def validate_values(vertices):
    """Fail before exporting unusable Gaussians, including zero quaternions."""
    for start in range(0, len(vertices), CHUNK_SIZE):
        block = vertices[start : start + CHUNK_SIZE]
        for name in vertices.dtype.names:
            if vertices.dtype.fields[name][0].kind == "f":
                if not np.isfinite(block[name]).all():
                    raise ValueError(f"Non-finite Gaussian attribute: {name}")
        q = columns(block, ROT).astype(np.float64)
        if np.any(np.linalg.norm(q, axis=1) < 1e-12):
            raise ValueError("Zero-length Gaussian quaternion")


def read_gaussians(path):
    ply = PlyData.read(str(path))
    if len(ply.elements) != 1 or ply.elements[0].name != "vertex":
        raise ValueError("Expected a vertex-only 3DGS PLY; mesh elements are unsupported")
    vertices = ply["vertex"].data
    if not len(vertices):
        raise ValueError(f"Empty Gaussian model: {path}")
    sh_degree(vertices)
    return vertices


def write_vertices(path, vertices, comment="gs-merger"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertices, "vertex")],
        text=False,
        byte_order="<",
        comments=[comment],
    ).write(str(path))


def write_point_cloud(path, points, colors=None):
    """Write a viewing proxy, never use this XYZ/RGB PLY as a Gaussian model."""
    fields = [(name, "f4") for name in XYZ]
    if colors is not None:
        fields += [(name, "u1") for name in ("red", "green", "blue")]
    vertices = np.empty(len(points), dtype=fields)
    set_columns(vertices, XYZ, points)
    if colors is not None:
        rgb = np.round(np.clip(colors, 0, 1) * 255).astype(np.uint8)
        set_columns(vertices, ["red", "green", "blue"], rgb)
    write_vertices(path, vertices, "XYZ proxy only; not a 3DGS model")


def export_view_cloud(source, destination, minimum_opacity=0.05, max_points=0, seed=123):
    """Stream original-coordinate XYZ/DC RGB for picking, without CUDA or voxelization.

    Zero max_points keeps every usable center. This is a viewing cloud, not a
    rendered surface or a Gaussian model; no synthetic points are introduced.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or (destination.exists() and destination.samefile(source)):
        raise ValueError("Viewing output would overwrite its input model")
    if not isinstance(max_points, (int, np.integer)) or max_points < 0:
        raise ValueError("Maximum viewing points must be an integer >= 0 (0 means all)")
    vertices = read_gaussians(source)

    def valid_centers(block):
        mask = np.isfinite(columns(block, XYZ + DC)).all(axis=1)
        mask &= np.isfinite(block["opacity"]) & opacity_mask(block, minimum_opacity)
        return mask

    counts = [
        int(valid_centers(vertices[start : start + CHUNK_SIZE]).sum())
        for start in range(0, len(vertices), CHUNK_SIZE)
    ]
    valid_count = sum(counts)
    if not valid_count:
        raise ValueError(f"No usable viewing centers in {source}; try lowering --min-opacity")
    count = min(max_points, valid_count) if max_points else valid_count
    ranks = (
        np.sort(np.random.default_rng(seed).choice(valid_count, count, replace=False))
        if count < valid_count
        else None
    )
    dtype = np.dtype(
        [(name, "<f4") for name in XYZ] + [(name, "u1") for name in ("red", "green", "blue")]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < count * dtype.itemsize + 1024**2:
        raise OSError("Not enough free disk space for the viewing cloud")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment Original-coordinate XYZ/DC RGB centers; not a 3DGS model\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".gs_merger_view_", suffix=".ply",
            dir=destination.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(header.encode("ascii"))
            offset = 0
            for i, valid in enumerate(counts):
                block = vertices[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
                selected = np.flatnonzero(valid_centers(block))
                if ranks is not None:
                    lo, hi = np.searchsorted(ranks, [offset, offset + valid])
                    selected = selected[ranks[lo:hi] - offset]
                offset += valid
                block = block[selected]
                cloud = np.empty(len(block), dtype=dtype)
                set_columns(cloud, XYZ, columns(block, XYZ))
                colors = np.clip(0.5 + 0.28209479177387814 * columns(block, DC), 0, 1)
                set_columns(
                    cloud, ["red", "green", "blue"], np.round(colors * 255).astype(np.uint8)
                )
                stream.write(cloud.data)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "source": str(source),
        "output": str(destination),
        "input_gaussians": len(vertices),
        "usable_centers": valid_count,
        "exported_centers": count,
        "max_points": int(max_points),
        "minimum_opacity": minimum_opacity,
        "seed": seed,
        "coordinate_frame": "original PLY; no centering, scaling or registration",
        "color": "clamp(0.5 + SH_C0 * f_dc, 0, 1); excludes view-dependent SH",
    }


def split_similarity(transform):
    """Return s, R, t from p_target = s R p_source + t; reject affine/shear."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7, rtol=0):
        raise ValueError("Invalid homogeneous transform bottom row")
    determinant = np.linalg.det(transform[:3, :3])
    if determinant <= 0:
        raise ValueError("Transform must have positive scale and no reflection")
    scale = float(np.cbrt(determinant))
    rotation = transform[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError("Only rigid or uniform-scale transforms are supported")
    return scale, rotation, transform[:3, 3]


def rotation_quaternion(rotation):
    """Convert SO(3) to a wxyz unit quaternion, including 180-degree rotations."""
    candidates = np.array(
        [
            1 + np.trace(rotation),
            1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2],
            1 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2],
            1 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2],
        ]
    )
    i = int(np.argmax(candidates))
    q = np.zeros(4)
    q[i] = 0.5 * np.sqrt(max(candidates[i], 0))
    denominator = 4 * q[i]
    if i == 0:
        q[1:] = [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
        q[1:] /= denominator
    else:
        axis = i - 1
        j, k = (axis + 1) % 3, (axis + 2) % 3
        q[0] = (rotation[k, j] - rotation[j, k]) / denominator
        q[j + 1] = (rotation[j, axis] + rotation[axis, j]) / denominator
        q[k + 1] = (rotation[k, axis] + rotation[axis, k]) / denominator
    return q / np.linalg.norm(q)


def quaternion_left_multiply(left, right):
    w, v = left[0], left[1:]
    rw, rv = right[:, :1], right[:, 1:]
    return np.column_stack((w * rw[:, 0] - rv @ v, w * rv + rw * v + np.cross(v, rv)))


def real_sh_basis(degree, directions):
    """Normalized real SH, l ascending then m=-l..l, with Condon-Shortley phase.

    This matches Graphdeco's basis: degree 1 is [-C*y, C*z, -C*x].
    Associated Legendre recurrence avoids a dependency on SH rotation libraries.
    """
    directions = np.asarray(directions, dtype=np.float64)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    z = np.clip(directions[:, 2], -1, 1)
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    basis = np.empty((len(directions), (degree + 1) ** 2))
    pmm = np.ones(len(directions))
    for m in range(degree + 1):
        if m:
            pmm = -(2 * m - 1) * np.sqrt(np.maximum(0, 1 - z * z)) * pmm
        previous = np.zeros_like(pmm)
        current = pmm
        for band in range(m, degree + 1):
            norm = np.sqrt(
                (2 * band + 1) / (4 * np.pi) * math.factorial(band - m) / math.factorial(band + m)
            )
            if m == 0:
                basis[:, band * band + band] = norm * current
            else:
                value = np.sqrt(2) * norm * current
                basis[:, band * band + band - m] = value * np.sin(m * phi)
                basis[:, band * band + band + m] = value * np.cos(m * phi)
            following = ((2 * band + 1) * z * current - (band + m) * previous) / (band - m + 1)
            previous, current = current, following
    return basis


def sh_rotation_blocks(degree, rotation):
    """Solve Y(d) D_l = Y(R^T d) for each degree in float64.

    The finite-dimensional SH identity is exact; redundant sphere samples only
    introduce numerical roundoff. Coefficients then transform as c' = D_l c.
    """
    count = 128
    indices = np.arange(count) + 0.5
    z = 1 - 2 * indices / count
    phi = indices * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(1 - z * z)
    directions = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
    before = real_sh_basis(degree, directions @ rotation)
    after = real_sh_basis(degree, directions)
    blocks = []
    for band in range(1, degree + 1):
        section = slice(band * band, (band + 1) ** 2)
        blocks.append(np.linalg.lstsq(after[:, section], before[:, section], rcond=None)[0])
    return blocks


def transform_gaussians(vertices, transform, sh_blocks=None):
    """Transform means, covariance orientations, log scales, normals, and SH."""
    degree = sh_degree(vertices)
    scale, rotation, translation = split_similarity(transform)
    validate_values(vertices)
    transformed = vertices.copy()
    set_columns(
        transformed,
        XYZ,
        columns(vertices, XYZ).astype(np.float64) @ (scale * rotation).T + translation,
    )
    q = columns(vertices, ROT).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q = quaternion_left_multiply(rotation_quaternion(rotation), q)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    set_columns(transformed, ROT, q)
    set_columns(transformed, SCALE, columns(vertices, SCALE).astype(np.float64) + np.log(scale))
    if all(name in vertices.dtype.names for name in NORMALS):
        set_columns(transformed, NORMALS, columns(vertices, NORMALS) @ rotation.T)
    if degree:
        if sh_blocks is None:
            sh_blocks = sh_rotation_blocks(degree, rotation)
        names = [f"f_rest_{i}" for i in range(3 * ((degree + 1) ** 2 - 1))]
        coeffs = columns(vertices, names).astype(np.float64).reshape(len(vertices), 3, -1)
        for band, block in enumerate(sh_blocks, start=1):
            section = slice(band * band - 1, (band + 1) ** 2 - 1)
            coeffs[:, :, section] = coeffs[:, :, section] @ block.T
        set_columns(transformed, names, coeffs.reshape(len(vertices), -1))
    validate_values(transformed)  # Also detect overflow when casting back to PLY float32.
    return transformed


def opacity_mask(vertices, minimum):
    if not np.isfinite(minimum) or not 0 <= minimum < 1:
        raise ValueError("Minimum opacity must be in [0, 1)")
    if minimum == 0:
        return np.ones(len(vertices), dtype=bool)
    # Graphdeco stores logits, not probabilities.
    return vertices["opacity"] >= np.log(minimum / (1 - minimum))


def merge_gaussian_files(source, target, transform, output_dir, minimum_opacity=0, *, log=None):
    """Concatenate target + transformed source using a disk-backed output buffer."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / "source_aligned.ply", output_dir / "merged.ply"]
    inputs = {Path(source).resolve(), Path(target).resolve()}
    if any(path.resolve() in inputs for path in outputs):
        raise ValueError("Output would overwrite an input model; choose another directory")
    def report(message):
        if log is not None:
            log(f"[merge] {message}")

    report("读取高斯 PLY 结构 …")
    src, tgt = read_gaussians(source), read_gaussians(target)
    if set(src.dtype.names) != set(tgt.dtype.names):
        raise ValueError("PLY schemas differ; both models must have the same SH degree and fields")
    for name in src.dtype.names:
        if src.dtype.fields[name][0] != tgt.dtype.fields[name][0]:
            raise ValueError(f"PLY field types differ: {name}")
    report(f"校验新模型 {len(src):,} 个高斯的全部属性 …")
    validate_values(src)
    report(f"校验已有模型 {len(tgt):,} 个高斯的全部属性 …")
    validate_values(tgt)
    src_mask, tgt_mask = opacity_mask(src, minimum_opacity), opacity_mask(tgt, minimum_opacity)
    n_src, n_tgt = int(src_mask.sum()), int(tgt_mask.sum())
    if not n_src or not n_tgt:
        raise ValueError("Opacity filtering removed an entire model")
    scale, rotation, _ = split_similarity(transform)
    blocks = sh_rotation_blocks(sh_degree(src), rotation)
    # Buffer + merged PLY + aligned source PLY all coexist until writing completes.
    required_bytes = (2 * (n_src + n_tgt) + n_src) * tgt.dtype.itemsize + 64 * 1024**2
    if shutil.disk_usage(output_dir).free < required_bytes:
        raise OSError(f"Merge needs about {required_bytes / 1e9:.2f} GB of free output-disk space")
    report(
        f"已有模型 {n_tgt:,} 个 + 新模型 {n_src:,} 个 = {n_tgt + n_src:,} 个高斯；"
        f"merged.ply 约 {(n_tgt + n_src) * tgt.dtype.itemsize / 1e9:.2f} GB；"
        f"临时空间预算 {required_bytes / 1e9:.2f} GB。"
    )

    def progress(stage, done, total, offset, started, last_report):
        if log is None:
            return last_report
        now = time.monotonic()
        if done == total or now - last_report >= 2:
            elapsed = max(now - started, 1e-9)
            remaining = (total - done) * elapsed / max(done, 1)
            report(
                f"{stage}：{done:,}/{total:,} ({100 * done / total:.1f}%)；"
                f"高斯准备进度 {100 * offset / (n_tgt + n_src):.1f}%；"
                f"{done * tgt.dtype.itemsize / elapsed / 1e6:.1f} MB/s；"
                f"本阶段预计剩余 {remaining:.0f} 秒"
            )
            return now
        return last_report

    # Millions of Gaussians can exceed RAM if several concatenated copies are made.
    with tempfile.TemporaryDirectory(prefix="gs_merger_merge_", dir=output_dir) as temporary:
        report(f"准备磁盘映射缓冲区：{Path(temporary) / 'vertices.bin'}")
        merged = np.memmap(
            Path(temporary) / "vertices.bin", mode="w+", dtype=tgt.dtype, shape=(n_tgt + n_src,)
        )
        offset = 0
        report("已有模型分块复制 …")
        started = time.monotonic()
        last_report = started - 2
        for start in range(0, len(tgt), CHUNK_SIZE):
            block = tgt[start : start + CHUNK_SIZE][tgt_mask[start : start + CHUNK_SIZE]]
            merged[offset : offset + len(block)] = block
            offset += len(block)
            last_report = progress("已有模型分块复制", offset, n_tgt, offset, started, last_report)
        report("新模型分块变换 …")
        # report("新模型分块变换（位置、四元数、尺度、球谐系数） …")
        started = time.monotonic()
        last_report = started - 2
        for start in range(0, len(src), CHUNK_SIZE):
            block = src[start : start + CHUNK_SIZE][src_mask[start : start + CHUNK_SIZE]]
            block = transform_gaussians(block, transform, blocks)
            destination = merged[offset : offset + len(block)]
            # Name-based copies also support different property ordering.
            for name in tgt.dtype.names:
                destination[name] = block[name]
            offset += len(block)
            last_report = progress(
                "新模型分块变换", offset - n_tgt, n_src, offset, started, last_report
            )
        report(f"高斯准备完成；写入对齐源模型，约 {n_src * tgt.dtype.itemsize / 1e9:.2f} GB …")
        write_vertices(outputs[0], merged[n_tgt:])
        report(f"写入完整 merged.ply，约 {(n_tgt + n_src) * tgt.dtype.itemsize / 1e9:.2f} GB …")
        write_vertices(outputs[1], merged)
        del destination, merged
        report("完整 PLY 写入完成。")
    return {
        "source_input_count": len(src),
        "target_input_count": len(tgt),
        "source_output_count": n_src,
        "target_output_count": n_tgt,
        "merged_count": n_src + n_tgt,
        "minimum_opacity": minimum_opacity,
        "uniform_scale": scale,
        "sh_degree": sh_degree(src),
        "source_aligned": str(outputs[0].resolve()),
        "merged": str(outputs[1].resolve()),
    }
