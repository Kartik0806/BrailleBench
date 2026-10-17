"""Random helpers, dtype helpers and quad geometry shared by every stage."""

import numpy as np
import cv2
import random


# ===========================================================================
# UTILITIES
# ===========================================================================


def rand(lo, hi):
    return random.uniform(lo, hi)


def randi(lo, hi):
    return random.randint(lo, hi)


def coin(p=0.5):
    return random.random() < p


def to_float(img_np):
    """uint8 → float32 [0,1]"""
    return img_np.astype(np.float32) / 255.0


def to_uint8(img_f):
    """float32 [0,1] → uint8"""
    return np.clip(img_f * 255, 0, 255).astype(np.uint8)


def ensure_3ch(img):
    """Ensure numpy image is HxWx3 uint8."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _order_quad(points: np.ndarray) -> np.ndarray:
    """Return four points in TL, TR, BR, BL order."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    pts = np.roll(pts, -start, axis=0)
    # After rotating to top-left, ensure the next point is the top-right.
    if pts[1, 0] < pts[-1, 0]:
        pts = np.concatenate([pts[:1], pts[:0:-1]], axis=0)
    return pts.astype(np.float32)


def _sample_quad_perimeter(quad: np.ndarray, samples_per_edge: int = 12) -> np.ndarray:
    """Densely sample a quad boundary so nonlinear warps cannot escape its corners."""
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    samples_per_edge = max(2, int(samples_per_edge))
    t = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False, dtype=np.float32)
    edges = [
        q[i][None, :] * (1.0 - t[:, None]) + q[(i + 1) % 4][None, :] * t[:, None]
        for i in range(4)
    ]
    return np.concatenate(edges, axis=0)


def _fit_enclosing_quad(points: np.ndarray) -> np.ndarray:
    """Minimum-area rectangle enclosing all supplied points."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3 or not np.isfinite(pts).all():
        x0, y0 = np.nanmin(pts, axis=0)
        x1, y1 = np.nanmax(pts, axis=0)
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    return _order_quad(cv2.boxPoints(rect))


def _expand_quad(quad: np.ndarray, padding: float) -> np.ndarray:
    """Expand an oriented quad by `padding` pixels on every side."""
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if padding <= 0:
        return _order_quad(q)
    center, size, angle = cv2.minAreaRect(q)
    expanded = (max(0.0, size[0] + 2.0 * padding), max(0.0, size[1] + 2.0 * padding))
    return _order_quad(cv2.boxPoints((center, expanded, angle)))


def _clip_quad_to_image(quad: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip a convex quad to the visible image and return an enclosing quad."""
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if width <= 0 or height <= 0:
        return q
    bounds = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float32,
    )
    try:
        _, intersection = cv2.intersectConvexConvex(
            cv2.convexHull(q).reshape(-1, 2), bounds
        )
    except cv2.error:
        intersection = None
    if intersection is not None and len(intersection) >= 3:
        return _fit_enclosing_quad(intersection.reshape(-1, 2))
    q[:, 0] = np.clip(q[:, 0], 0.0, width - 1.0)
    q[:, 1] = np.clip(q[:, 1], 0.0, height - 1.0)
    return _order_quad(q)


def _sample_remap_field(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinearly sample a cv2.remap coordinate field at floating-point positions."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    map_x = pts[:, 0].reshape(-1, 1)
    map_y = pts[:, 1].reshape(-1, 1)
    return cv2.remap(
        field,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1)


def _invert_remap_points(
    source_points: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    iterations: int = 10,
) -> np.ndarray:
    """Solve map(destination)=source for near-identity smooth remap fields."""
    src = np.asarray(source_points, dtype=np.float32).reshape(-1, 2)
    dst = src.copy()
    h, w = map_x.shape[:2]
    for _ in range(max(1, int(iterations))):
        mapped = np.column_stack(
            (_sample_remap_field(map_x, dst), _sample_remap_field(map_y, dst))
        ).astype(np.float32)
        delta = src - mapped
        dst += delta
        dst[:, 0] = np.clip(dst[:, 0], 0.0, w - 1.0)
        dst[:, 1] = np.clip(dst[:, 1], 0.0, h - 1.0)
        if float(np.max(np.linalg.norm(delta, axis=1))) < 0.05:
            break
    return dst


def _transform_quads_through_remap(
    quads: list[np.ndarray],
    map_x: np.ndarray,
    map_y: np.ndarray,
    samples_per_edge: int = 12,
) -> list[np.ndarray]:
    """Track quads through a backward remap using dense perimeter inversion."""
    h, w = map_x.shape[:2]
    transformed = []
    for quad in quads:
        perimeter = _sample_quad_perimeter(quad, samples_per_edge=samples_per_edge)
        moved = _invert_remap_points(perimeter, map_x, map_y)
        transformed.append(_clip_quad_to_image(_fit_enclosing_quad(moved), w, h))
    return transformed


def _fit_within_limits(
    width: int, height: int, max_width: int, max_height: int
) -> tuple[int, int, float]:
    """Return a size that fits within the requested bounds while preserving aspect ratio."""
    scale = min(1.0, max_width / max(width, 1), max_height / max(height, 1))
    fitted_width = max(1, int(np.floor(width * scale)))
    fitted_height = max(1, int(np.floor(height * scale)))
    return fitted_width, fitted_height, scale
