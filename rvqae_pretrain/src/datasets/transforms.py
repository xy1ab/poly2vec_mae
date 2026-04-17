"""Data augmentation transforms for triangle-based polygon samples.

This module provides deterministic-shape-preserving geometric augmentation in
normalized coordinate space `[-1, 1]`.
"""

from __future__ import annotations

import math
import random

import numpy as np


def augment_triangles(tris: np.ndarray) -> np.ndarray:
    """Apply rotation, scale, and translation jitter to triangle coordinates.

    Args:
        tris: Triangle tensor shaped `[T, 3, 2]`.

    Returns:
        Augmented triangle tensor with the same shape.
    """
    angle = random.uniform(0.0, 2.0 * math.pi)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    tris_shape = tris.shape
    tris_flat = tris.reshape(-1, 2).dot(rot_matrix)

    min_c = tris_flat.min(axis=0)
    max_c = tris_flat.max(axis=0)

    scale = random.uniform(0.5, 1.0)
    tris_flat = tris_flat * scale
    min_c *= scale
    max_c *= scale

    max_tx = 1.0 - max_c[0]
    min_tx = -1.0 - min_c[0]
    max_ty = 1.0 - max_c[1]
    min_ty = -1.0 - min_c[1]

    tx = random.uniform(min_tx, max_tx) if max_tx >= min_tx else 0.0
    ty = random.uniform(min_ty, max_ty) if max_ty >= min_ty else 0.0

    tris_flat += np.array([tx, ty], dtype=np.float32)
    return tris_flat.reshape(tris_shape).astype(np.float32)
