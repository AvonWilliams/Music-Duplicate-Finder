# Music Duplicate Finder — fingerprint_graph.py
# Windowed chromaprint similarity curve widget.
# Used in the results dialog to show how closely a duplicate track matches
# the reference track across its duration.

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

if TYPE_CHECKING:
    import numpy as np


def _decode(fp_str: str):
    """Decode a compressed chromaprint string to a numpy uint32 array."""
    if not fp_str:
        return None
    try:
        import numpy as _np
        from acoustid import chromaprint as _cp
        data = fp_str.encode("ascii") if isinstance(fp_str, str) else fp_str
        ints, _version = _cp.decode_fingerprint(data)
        if not ints:
            return None
        return _np.asarray(ints, dtype=_np.uint32)
    except Exception:
        return None


def _popcount(arr):
    """Vectorised popcount for a numpy uint32 array (works on all numpy versions)."""
    c = arr - ((arr >> 1) & 0x55555555)
    c = (c & 0x33333333) + ((c >> 2) & 0x33333333)
    c = (c + (c >> 4)) & 0x0F0F0F0F
    return ((c * 0x01010101) >> 24).astype("int32")


def _best_offset(a, b, max_off: int) -> int:
    """Return the alignment offset that maximises overall similarity."""
    best_off, best_sim = 0, -1.0
    for off in range(-max_off, max_off + 1):
        if off >= 0:
            L = min(len(a) - off, len(b))
            if L < 30:
                continue
            bits = int(_popcount(a[off:off + L] ^ b[:L]).sum())
        else:
            L = min(len(a), len(b) + off)
            if L < 30:
                continue
            bits = int(_popcount(a[:L] ^ b[-off:-off + L]).sum())
        sim = 1.0 - bits / (L * 32.0)
        if sim > best_sim:
            best_sim, best_off = sim, off
    return best_off


def compute_similarity_curve(
    fp_ref: str,
    fp_other: str,
    window: int = 40,
    step: int = 8,
    align: bool = True,
) -> list[float] | None:
    """
    Return per-window Hamming similarity between two chromaprint strings.
    window=40 frames ≈ 5 s, step=8 frames ≈ 1 s per data point.
    Values are 0.0 (no match) – 1.0 (perfect match).
    Returns None if either fingerprint is missing or too short.
    align=False skips the offset search and compares at position 0 (faster,
    suitable when the files are similar-but-not-identical, e.g. CLAP results).
    """
    a = _decode(fp_ref)
    b = _decode(fp_other)
    if a is None or b is None or len(a) < window or len(b) < window:
        return None

    off = _best_offset(a, b, min(40, len(a) // 4, len(b) // 4)) if align else 0

    if off >= 0:
        a_al, b_al = a[off:], b[:len(a) - off]
    else:
        b_al, a_al = b[-off:], a[:len(b) + off]

    L = min(len(a_al), len(b_al))
    if L < window:
        return None
    a_al, b_al = a_al[:L], b_al[:L]

    curve = []
    for i in range(0, L - window + 1, step):
        bits = int(_popcount(a_al[i:i + window] ^ b_al[i:i + window]).sum())
        sim = max(0.0, min(1.0, 1.0 - bits / (window * 32.0)))
        curve.append(sim)

    return curve or None


def _sim_color(v: float, alpha: int = 180) -> QColor:
    """Red (0.0) → yellow (0.5) → green (1.0), semi-transparent fill."""
    v = max(0.0, min(1.0, v))
    if v >= 0.5:
        t = (v - 0.5) * 2.0
        r = int(255 * (1.0 - t) + 26  * t)
        g = int(200 * (1.0 - t) + 127 * t)
        b = int(0   * (1.0 - t) + 55  * t)
    else:
        t = v * 2.0
        r = int(207 * (1.0 - t) + 255 * t)
        g = int(34  * (1.0 - t) + 200 * t)
        b = int(46  * (1.0 - t) + 0   * t)
    return QColor(r, g, b, alpha)


class FingerprintGraphWidget(QWidget):
    """Painted line graph showing per-window similarity over track time."""

    _PL = 32    # left padding  — Y-axis labels
    _PR = 4     # right padding
    _PT = 4     # top padding
    _PB = 16    # bottom padding — X-axis labels

    def __init__(
        self,
        fp_ref: str,
        fp_other: str,
        is_best: bool,
        ref_duration: float = 0.0,
        align: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._is_best   = is_best
        self._duration  = ref_duration
        self._curve: list[float] | None = None
        if not is_best:
            self._curve = compute_similarity_curve(fp_ref, fp_other, align=align)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(160, 80)
        self.setToolTip(
            "Chromaprint similarity vs. reference track over time.\n"
            "Top = 100% match  ·  Bottom = 0% match"
        )

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        pl, pr, pt, pb = self._PL, self._PR, self._PT, self._PB

        p.fillRect(0, 0, W, H, QColor("#f5f5f5"))
        p.setPen(QPen(QColor("#ccc"), 1))
        p.drawRect(0, 0, W - 1, H - 1)

        if self._is_best:
            p.setPen(QColor("#1a7f37"))
            p.setFont(QFont("sans-serif", 8))
            p.drawText(0, 0, W, H, Qt.AlignmentFlag.AlignCenter, "Reference track")
            return

        if not self._curve:
            p.setPen(QColor("#aaa"))
            p.setFont(QFont("sans-serif", 8))
            p.drawText(0, 0, W, H, Qt.AlignmentFlag.AlignCenter, "No fingerprint data")
            return

        gx = pl              # graph area left edge
        gy = pt              # graph area top edge
        gw = W - pl - pr     # graph area width
        gh = H - pt - pb     # graph area height
        n  = len(self._curve)

        # ── Axis labels + guide lines ─────────────────────────────────────
        lbl_font = QFont("sans-serif", 7)
        p.setFont(lbl_font)

        # Y-axis: labels at 100%/75%/50%/25%/0%, horizontal guides for middle three
        for pct, label in [(1.0, "100%"), (0.75, "75%"), (0.5, "50%"), (0.25, "25%"), (0.0, "0%")]:
            yg = gy + int(gh * (1.0 - pct))
            p.setPen(QColor("#777"))
            p.drawText(0, yg - 6, pl - 3, 12,
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       label)
            if 0.0 < pct < 1.0:
                p.setPen(QPen(QColor("#e0e0e0"), 1))
                p.drawLine(gx, yg, gx + gw, yg)

        # X-axis: "0:00" left, duration right, time splits at 25%/50%/75%
        p.setPen(QColor("#777"))
        p.drawText(gx, H - pb + 1, 28, pb - 1,
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   "0:00")
        if self._duration > 0:
            m, s = divmod(int(self._duration), 60)
            p.drawText(gx + gw - 28, H - pb + 1, 28, pb - 1,
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f"{m}:{s:02d}")
            for frac in (0.25, 0.5, 0.75):
                t = int(self._duration * frac)
                xm = gx + int(gw * frac)
                mm, ss = divmod(t, 60)
                p.setPen(QPen(QColor("#e8e8e8"), 1))
                p.drawLine(xm, gy, xm, gy + gh)
                p.setPen(QColor("#777"))
                p.drawText(xm - 14, H - pb + 1, 28, pb - 1,
                           Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter,
                           f"{mm}:{ss:02d}")

        # ── Axis lines ────────────────────────────────────────────────────
        p.setPen(QPen(QColor("#bbb"), 1))
        p.drawLine(gx, gy, gx, gy + gh)
        p.drawLine(gx, gy + gh, gx + gw, gy + gh)

        # ── Point positions ───────────────────────────────────────────────
        pts = [
            (gx + int(i * gw / max(n - 1, 1)), gy + int((1.0 - v) * gh))
            for i, v in enumerate(self._curve)
        ]

        # ── Dynamic coloured fill: one trapezoid per segment ──────────────
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            yb = gy + gh
            v_mid = (self._curve[i] + self._curve[i + 1]) / 2.0
            fill = QPainterPath()
            fill.moveTo(x0, yb)
            fill.lineTo(x0, y0)
            fill.lineTo(x1, y1)
            fill.lineTo(x1, yb)
            fill.closeSubpath()
            p.fillPath(fill, _sim_color(v_mid))

        # ── Line on top ───────────────────────────────────────────────────
        mean_v = sum(self._curve) / len(self._curve)
        line_col = (QColor("#1a7f37") if mean_v >= 0.80
                    else QColor("#9a6700") if mean_v >= 0.65
                    else QColor("#cf222e"))
        p.setPen(QPen(line_col, 1.5))
        for i in range(len(pts) - 1):
            p.drawLine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
