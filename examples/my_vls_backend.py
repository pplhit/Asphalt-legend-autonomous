"""Template for YOUR calibrated 54->8 VLS photonic dot-product implementation.

This is intentionally not fake optical inference. Modify `infer` after mapping
VLS encoding, optical calibration, parallel multiplexing and CMOS readout.

Usage:
  python drive.py --checkpoint runs/track1/best.pt \
    --backend hardware --vls-plugin examples.my_vls_backend:infer
"""
import numpy as np


def infer(vectors: np.ndarray) -> np.ndarray:
    """Input [N,54] float32 convolution patches, output [N,8] float32.

    To reproduce conv3 exactly, return vectors @ W.T + b with calibrated
    signed real values. W and b are exported by export_vls.py.

    Physical constraints:
    - 8 output channels for each position (N approx 60 per 160x96 frame).
    - VLS 54-dimensional coefficient input must preserve sign/dynamic range.
    - CMOS intensities alone do not encode signed dot-products without
      differential/reference or another calibrated conversion.
    - SLM refresh time and readout bandwidth must be budgeted.
    """
    raise NotImplementedError('Connect VLS SLM/CMOS hardware here.')
