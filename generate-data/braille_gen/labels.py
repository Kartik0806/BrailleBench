"""Bounding-box label records and visualisation."""

import numpy as np
import cv2
from typing import Optional
from .braille_text import BRAILLE_SPACE
from .utils import _clip_quad_to_image, _expand_quad, ensure_3ch


# ===========================================================================
# BOUNDING-BOX RECORD ASSEMBLY + VISUALIZATION
# ===========================================================================


def _quad_bbox(
    quad: np.ndarray, img_w: Optional[int] = None, img_h: Optional[int] = None
) -> list[int]:
    """Conservative [x, y, width, height] bbox from a quad."""
    q = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    x0 = int(np.floor(float(q[:, 0].min())))
    y0 = int(np.floor(float(q[:, 1].min())))
    # +1 keeps the last antialiased pixel whose centre can lie just beyond
    # the transformed continuous boundary. The upper edge remains exclusive.
    x1 = int(np.ceil(float(q[:, 0].max()))) + 1
    y1 = int(np.ceil(float(q[:, 1].max()))) + 1
    if img_w is not None:
        x0 = int(np.clip(x0, 0, img_w))
        x1 = int(np.clip(x1, x0, img_w))
    if img_h is not None:
        y0 = int(np.clip(y0, 0, img_h))
        y1 = int(np.clip(y1, y0, img_h))
    return [x0, y0, max(0, x1 - x0), max(0, y1 - y0)]


def _rect_quad(x0, y0, x1, y1) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def build_bbox_record(
    text: str,
    word_records: list[dict],
    img_w: int,
    img_h: int,
    sample_id: int,
    fname: str,
    bg_used: bool,
    bg_source: Optional[str],
    source_meta: Optional[dict] = None,
    sample_meta: Optional[dict] = None,
) -> dict:
    """Assemble clipped, conservative word/line/sentence/paragraph boxes."""
    word_bboxes = []
    lines_map: dict[int, list[int]] = {}
    clipped_quads: list[np.ndarray] = []

    for idx, rec in enumerate(word_records):
        quad = _clip_quad_to_image(
            _expand_quad(np.asarray(rec["quad"], dtype=np.float32), 1.0),
            img_w,
            img_h,
        )
        clipped_quads.append(quad)
        word_bboxes.append(
            {
                "text": rec["text"],
                "braille_text": rec["braille_text"],
                "bbox": _quad_bbox(quad, img_w, img_h),
                "quad": quad.tolist(),
                "sentence_idx": 0,
                "line_idx": rec["line_idx"],
            }
        )
        lines_map.setdefault(rec["line_idx"], []).append(idx)

    line_bboxes = []
    for line_idx in sorted(lines_map):
        idxs = lines_map[line_idx]
        pts = np.concatenate([clipped_quads[i] for i in idxs], axis=0)
        x0 = max(0.0, float(pts[:, 0].min()))
        y0 = max(0.0, float(pts[:, 1].min()))
        x1 = min(float(img_w), float(pts[:, 0].max()))
        y1 = min(float(img_h), float(pts[:, 1].max()))
        line_quad = np.asarray(_rect_quad(x0, y0, x1, y1), dtype=np.float32)
        words_en = [word_bboxes[i]["text"] for i in idxs]
        words_br = [word_bboxes[i]["braille_text"] for i in idxs]
        line_bboxes.append(
            {
                "text": " ".join(words_en),
                "braille_text": BRAILLE_SPACE.join(words_br),
                "bbox": _quad_bbox(line_quad, img_w, img_h),
                "quad": line_quad.tolist(),
                "para_idx": 0,
                "line_idx": line_idx,
                "word_indices": idxs,
            }
        )

    all_line_indices = list(range(len(line_bboxes)))
    if line_bboxes:
        all_pts = np.concatenate(
            [np.asarray(lb["quad"], dtype=np.float32) for lb in line_bboxes], axis=0
        )
        fx0 = max(0.0, float(all_pts[:, 0].min()))
        fy0 = max(0.0, float(all_pts[:, 1].min()))
        fx1 = min(float(img_w), float(all_pts[:, 0].max()))
        fy1 = min(float(img_h), float(all_pts[:, 1].max()))
        full_quad_np = np.asarray(_rect_quad(fx0, fy0, fx1, fy1), dtype=np.float32)
        full_bbox = _quad_bbox(full_quad_np, img_w, img_h)
        full_quad = full_quad_np.tolist()
    else:
        full_bbox, full_quad = [0, 0, 0, 0], []

    sentence_bboxes = [
        {
            "sentence_idx": 0,
            "text": text,
            "bbox": full_bbox,
            "quad": full_quad,
            "line_indices": all_line_indices,
            "global_line_indices": all_line_indices,
        }
    ]
    para_bboxes = [
        {
            "para_idx": 0,
            "text": text,
            "bbox": full_bbox,
            "quad": full_quad,
            "global_line_indices": all_line_indices,
        }
    ]
    relation_graph = [[lines_map[li] for li in sorted(lines_map)]]

    record = {
        "source_text": text,
        "braille_text": BRAILLE_SPACE.join(w["braille_text"] for w in word_bboxes),
        "word_bboxes": word_bboxes,
        "line_bboxes": line_bboxes,
        "sentence_bboxes": sentence_bboxes,
        "para_bboxes": para_bboxes,
        "relation_graph": relation_graph,
        "image_width": img_w,
        "image_height": img_h,
        "rendering_path": "braille_dataset_generator",
        "background_category": "photo" if bg_used else "none",
        "background_source": bg_source or "",
        "id": f"{sample_id:08d}",
        "image": f"images/{fname}",
    }
    if source_meta is not None:
        record["source"] = source_meta
    if sample_meta is not None:
        record["augmentation"] = sample_meta
        record["braille_grade"] = sample_meta.get("braille_grade", 1)
        record["braille_backend"] = sample_meta.get("braille_backend", "")
    return record


def draw_visualization(img: np.ndarray, record: dict) -> np.ndarray:
    """Overlay word quads (green) and line bboxes (blue) for QA."""
    vis = ensure_3ch(img.copy())
    for wb in record["word_bboxes"]:
        pts = (
            np.rint(np.array(wb["quad"], dtype=np.float32))
            .astype(np.int32)
            .reshape(-1, 1, 2)
        )
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    for lb in record["line_bboxes"]:
        x, y, bw, bh = lb["bbox"]
        cv2.rectangle(
            vis,
            (x, y),
            (max(x, x + bw - 1), max(y, y + bh - 1)),
            color=(255, 0, 0),
            thickness=2,
        )
    return vis
