"""Safe serialization compatibility helpers.

PyTorch 2.6+ enables stricter object safety during torch.load. This module
registers NumPy-related globals that are required to load legacy datasets.
"""

from __future__ import annotations

import numpy as np
import torch


def register_numpy_safe_globals() -> None:
    """Register NumPy globals for legacy `torch.load(..., weights_only=False)`.

    This function is intentionally best-effort and should never raise on failure,
    because strict registration behavior may vary across NumPy versions.
    """
    if not hasattr(torch.serialization, "add_safe_globals"):
        return

    try:
        torch.serialization.add_safe_globals([np.ndarray, np.dtype])
        if hasattr(np, "_core"):
            import numpy._core.multiarray  # type: ignore

            torch.serialization.add_safe_globals(
                [
                    numpy._core.multiarray._reconstruct,  # type: ignore[attr-defined]
                    numpy._core.multiarray.scalar,  # type: ignore[attr-defined]
                ]
            )
        elif hasattr(np, "core"):
            import numpy.core.multiarray  # type: ignore

            torch.serialization.add_safe_globals(
                [
                    numpy.core.multiarray._reconstruct,  # type: ignore[attr-defined]
                    numpy.core.multiarray.scalar,  # type: ignore[attr-defined]
                ]
            )

        torch.serialization.add_safe_globals([type(np.dtype(np.float32)), type(np.dtype(np.float64)), np.float32, np.float64])
    except Exception:
        # Keep silent by design to avoid breaking runtime.
        return
