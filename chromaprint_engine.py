# Music Duplicate Finder — chromaprint_engine.py
# V2.0: AcoustID/Chromaprint-based duplicate detection.
#
# Unlike CLAP (which detects genre/production similarity), Chromaprint is
# a true track-identity fingerprint. Two recordings of the same master
# score > 0.95 regardless of codec, bitrate, or container. Genre-matched
# but unrelated tracks score < 0.30. The gap in the middle is what makes
# it good at this job.
#
# Pipeline:
#   1. Input: dict[path -> compressed_fingerprint_string] sourced from
#      Picard file.metadata['acoustid_fingerprint'].
#   2. Decode each fingerprint to a uint32 array (via pyacoustid's
#      chromaprint.decode_fingerprint).
#   3. Pad/pack to a batched tensor, compute Hamming-based similarity
#      across alignment offsets, take the best offset per pair.
#   4. Feed similarity matrix to the shared complete-linkage clusterer.
#
# Dependencies:
#   numpy         — always required
#   pyacoustid    — required for fingerprint decode (Picard ships with this)
#   torch         — required for GPU; CPU fallback only if the user accepts
#                   the runtime warning.
#
# Alignment window selection (chromaprint timestep = ~0.124s per int32):
#   "off":         no offset tried    (exact only)
#   "narrow":      ±4                 (~0.5s tolerance)
#   "standard":    ±20                (~2.5s; what AcoustID itself uses)
#   "wide":        ±80                (~10s)
#   "exhaustive":  ±200               (~25s; debug only)

from __future__ import annotations

import time
from typing import Any, Callable

from .diag import get_logger

_log = get_logger("chromaprint_engine")


# Mapping from config string -> offset half-window (integer timesteps).
ALIGNMENT_WINDOWS = {
    "off":        0,
    "narrow":     4,
    "standard":  20,
    "wide":      80,
    "exhaustive": 200,
}
DEFAULT_ALIGNMENT = "standard"


# Per-offset overlap must cover at least this many timesteps for a
# comparison to be considered valid. Prevents tiny overlaps at extreme
# offsets from producing spurious "perfect" scores.
MIN_OVERLAP_STEPS = 30  # ~3.7 seconds of audio


class MissingDependencyError(RuntimeError):
    """Raised when a required package is missing."""


class NoFingerprintsError(RuntimeError):
    """Raised when no input files had usable fingerprints."""


# ──────────────────────────────────────────────────────────────────────────
# Dependency detection
# ──────────────────────────────────────────────────────────────────────────

def check_dependencies() -> list[str]:
    """Return a list of missing package names. Empty = all good."""
    missing: list[str] = []
    for pkg in ("numpy", "acoustid"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def torch_available() -> bool:
    """True if torch is importable and at least one CUDA device is visible."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:  # noqa: BLE001
        return False


# ──────────────────────────────────────────────────────────────────────────
# Fingerprint decoding
# ──────────────────────────────────────────────────────────────────────────

def _decode_fingerprint(compressed: str):
    """
    Decode Picard's compressed-base64 chromaprint string into a numpy
    uint32 array. Returns None on decode failure.

    Picard stores acoustid_fingerprint in the compressed form that
    AcoustID submits over the wire (bytes start with a version header,
    then the raw 32-bit differential-encoded integers).
    """
    try:
        from acoustid import chromaprint
    except ImportError as exc:
        raise MissingDependencyError(
            "pyacoustid is required for AcoustID duplicate detection. "
            "Install with:  pip install pyacoustid"
        ) from exc
    import numpy as np

    try:
        # chromaprint.decode_fingerprint returns (ints, version)
        decoded = chromaprint.decode_fingerprint(
            compressed.encode("ascii") if isinstance(compressed, str) else compressed
        )
    except Exception as exc:  # noqa: BLE001
        _log.debug("decode_fingerprint failed: %s: %s", type(exc).__name__, exc)
        return None

    if not decoded:
        return None
    ints = decoded[0] if isinstance(decoded, tuple) else decoded
    if not ints:
        return None
    arr = np.asarray(ints, dtype=np.uint32)
    if arr.size < MIN_OVERLAP_STEPS:
        _log.debug(
            "fingerprint too short (%d steps < %d minimum), skipping",
            arr.size, MIN_OVERLAP_STEPS,
        )
        return None
    return arr


# ──────────────────────────────────────────────────────────────────────────
# CPU popcount (numpy bit-twiddle)
# ──────────────────────────────────────────────────────────────────────────

def _popcount32_np(x):
    """Hamming weight of a numpy uint32 array. Returns same shape as int32."""
    import numpy as np
    x = x.astype(np.uint32, copy=False)
    # Promote to 64-bit for intermediate math to avoid overflow on the multiply.
    x64 = x.astype(np.uint64)
    x64 = x64 - ((x64 >> 1) & np.uint64(0x5555555555555555))
    x64 = (x64 & np.uint64(0x3333333333333333)) + ((x64 >> 2) & np.uint64(0x3333333333333333))
    x64 = (x64 + (x64 >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x64 * np.uint64(0x0101010101010101)) >> 56).astype(np.int32)


# ──────────────────────────────────────────────────────────────────────────
# CuPy fused CUDA kernel (Option 3 — no intermediate tensor allocations)
# ──────────────────────────────────────────────────────────────────────────
#
# Each GPU thread handles one (query_i, target_j) pair. It loops over every
# alignment offset in registers, runs __popc on the XOR of the two 32-bit
# fingerprint ints directly, and writes only the final best-similarity float.
# No (Q, N, L) XOR tensors are ever materialised in VRAM.

_CUPY_KERNEL_SRC = r"""
extern "C" __global__
void chromaprint_sim(
    const int* __restrict__ fp_q,
    const int* __restrict__ fp_t,
    const unsigned char* __restrict__ vq,
    const unsigned char* __restrict__ vt,
    float* __restrict__ out,
    int Q, int N, int L,
    int max_offset, int min_overlap
) {
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = blockIdx.y * blockDim.y + threadIdx.y;
    if (qi >= Q || ti >= N) return;

    const int*           fq  = fp_q + (ptrdiff_t)qi * L;
    const int*           ft  = fp_t + (ptrdiff_t)ti * L;
    const unsigned char* vqi = vq   + (ptrdiff_t)qi * L;
    const unsigned char* vti = vt   + (ptrdiff_t)ti * L;

    float best = 0.0f;
    for (int off = -max_offset; off <= max_offset; off++) {
        int q0  = off < 0 ? -off : 0;
        int t0  = off > 0 ?  off : 0;
        int len = L - (off < 0 ? -off : off);
        if (len <= 0) continue;

        int diff = 0, steps = 0;
        for (int l = 0; l < len; l++) {
            if (vqi[q0 + l] & vti[t0 + l]) {
                diff += __popc((unsigned int)fq[q0 + l] ^ (unsigned int)ft[t0 + l]);
                steps++;
            }
        }
        if (steps >= min_overlap) {
            float sim = 1.0f - (float)diff / (float)(steps * 32);
            if (sim > best) best = sim;
        }
    }
    out[(ptrdiff_t)qi * N + ti] = best;
}
"""

_cupy_kernel_cache = None


def _get_cupy_kernel():
    global _cupy_kernel_cache
    if _cupy_kernel_cache is None:
        import cupy as cp
        _cupy_kernel_cache = cp.RawKernel(_CUPY_KERNEL_SRC, "chromaprint_sim")
    return _cupy_kernel_cache


def _cupy_install_hint() -> str:
    """Return the correct pip install command for CuPy based on the detected CUDA version."""
    try:
        import torch
        cuda_ver = getattr(torch.version, "cuda", None) or ""
        major = cuda_ver.split(".")[0] if "." in cuda_ver else cuda_ver
        if major.isdigit():
            return f"pip install cupy-cuda{major}x"
    except Exception:  # noqa: BLE001
        pass
    return "pip install cupy-cuda12x  (replace '12' with your CUDA major version)"


# ──────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────

class ChromaprintEngine:
    """
    GPU-accelerated (torch) AcoustID fingerprint comparison engine.
    Falls back to CPU (numpy) when torch is unavailable or caller asked for it.
    """

    def __init__(
        self,
        alignment: str          = DEFAULT_ALIGNMENT,
        use_gpu: bool           = True,
        gpu_index: int          = 0,
        query_tile: int         = 64,     # query rows per GPU tile
        db_tile: int            = 512,    # target rows per GPU tile (limits VRAM)
        progress_cb: "Callable[[str], None] | None" = None,
    ):
        if alignment not in ALIGNMENT_WINDOWS:
            raise ValueError(f"alignment must be one of {list(ALIGNMENT_WINDOWS)}")
        self._alignment = alignment
        self._max_offset = ALIGNMENT_WINDOWS[alignment]
        self._want_gpu = use_gpu
        self._gpu_index = gpu_index
        self._tile = query_tile
        self._db_tile = db_tile
        self._progress_cb = progress_cb

    def _report(self, msg: str) -> None:
        if self._progress_cb:
            try:
                self._progress_cb(msg)
            except Exception:  # noqa: BLE001
                pass

    # ── Public entrypoint ─────────────────────────────────────────────────

    def find_duplicates(
        self,
        fingerprint_map: dict[str, str],
        certain_threshold: float,
        likely_threshold: float,
        unsure_threshold: float,
        abort_flag: "Callable[[], bool] | None" = None,
    ) -> dict[str, Any]:
        """
        fingerprint_map: path -> compressed chromaprint string.
        Returns the same dict shape as LocalInferenceEngine.find_duplicates.
        """
        missing = check_dependencies()
        if missing:
            raise MissingDependencyError(
                "AcoustID engine missing: {0}. Install with:  pip install {1}".format(
                    ", ".join(missing), " ".join(missing),
                )
            )

        import numpy as np

        t_start = time.time()

        # ── Decode fingerprints ──────────────────────────────────────────
        self._report("Decoding {0} fingerprints…".format(len(fingerprint_map)))

        paths: list[str] = []
        arrays: list = []
        decode_failures = 0
        for path, compressed in fingerprint_map.items():
            if abort_flag and abort_flag():
                return self._empty_result(t_start)
            if not compressed:
                decode_failures += 1
                continue
            arr = _decode_fingerprint(compressed)
            if arr is None:
                decode_failures += 1
                _log.debug("skipping %r — fingerprint decode failed", path)
                continue
            paths.append(path)
            arrays.append(arr)

        n = len(paths)
        _log.info(
            "decoded %d fingerprints  (failures=%d, alignment=%s, max_offset=%d steps)",
            n, decode_failures, self._alignment, self._max_offset,
        )

        if n < 2:
            raise NoFingerprintsError(
                "Only {0} file(s) had usable AcoustID fingerprints — need at "
                "least 2 to compare. Use Picard → Tools → Calculate AcoustID "
                "Fingerprints to populate them.".format(n)
            )

        # ── Pad to common length ─────────────────────────────────────────
        lengths = np.array([a.size for a in arrays], dtype=np.int32)
        L_max = int(lengths.max())
        L_min = int(lengths.min())
        L_mean = int(lengths.mean())
        _log.info(
            "fingerprint length stats: min=%d  mean=%d  max=%d  (timesteps @ ~0.124s each)",
            L_min, L_mean, L_max,
        )

        fps = np.zeros((n, L_max), dtype=np.uint32)
        valid = np.zeros((n, L_max), dtype=np.uint8)  # 1 where real data
        for i, a in enumerate(arrays):
            fps[i, : a.size] = a
            valid[i, : a.size] = 1

        # Free the per-file arrays now that we've packed them.
        del arrays

        # ── Dispatch to GPU or CPU implementation ────────────────────────
        use_gpu = self._want_gpu and torch_available()
        if self._want_gpu and not torch_available():
            _log.warning(
                "GPU requested but torch/CUDA not available — falling back to CPU. "
                "This will be significantly slower for large libraries."
            )

        self._report(
            "Comparing {0} fingerprints on {1} "
            "(alignment={2}, {3} offsets)…".format(
                n, "GPU" if use_gpu else "CPU",
                self._alignment, 2 * self._max_offset + 1,
            )
        )

        t_compare0 = time.time()
        if use_gpu:
            sim = self._compare_gpu(fps, valid, abort_flag)
            import torch
            torch.cuda.empty_cache()
            _log.info("[VRAM-DIAG] after chromaprint GPU + empty_cache: %s",
                      self._vram_mb(torch.device("cuda:{0}".format(self._gpu_index))))
        else:
            sim = self._compare_cpu(fps, valid, abort_flag)

        if abort_flag and abort_flag():
            return self._empty_result(t_start)

        t_compare = time.time() - t_compare0
        _log.info(
            "pairwise comparison complete on %s: %d files -> %d pairs "
            "in %.2fs  (%.1f pairs/sec)",
            "GPU" if use_gpu else "CPU",
            n, n * (n - 1) // 2, t_compare,
            (n * (n - 1) / 2) / max(t_compare, 0.001),
        )

        # Diagnostic stats on the similarity matrix
        self._log_similarity_stats(sim, paths)

        # ── Cluster ──────────────────────────────────────────────────────
        from .clustering import cluster_complete_linkage
        self._report("Clustering into duplicate groups…")
        raw_groups = cluster_complete_linkage(
            sim               = sim,
            paths             = paths,
            certain_threshold = certain_threshold,
            likely_threshold  = likely_threshold,
            unsure_threshold  = unsure_threshold,
            log               = _log,
            log_prefix        = "chromaprint",
        )

        # Warn if thresholds are set below the noise floor.
        if sim.shape[0] > 1:
            off = sim[np.triu_indices_from(sim, k=1)]
            if off.size > 0:
                mean_off = float(off.mean())
                if unsure_threshold <= mean_off + 0.05:
                    _log.warning(
                        "unsure_threshold %.3f is at or below the mean pair "
                        "similarity %.3f — expect many false positives. "
                        "Try raising to at least %.3f.",
                        unsure_threshold, mean_off, mean_off + 0.15,
                    )

        elapsed = time.time() - t_start
        return {
            "groups":          raw_groups,
            "scanned":         n,
            "embedded":        n,                 # chromaprint has no embed step
            "embed_failures":  decode_failures,
            "elapsed_seconds": elapsed,
        }

    def _empty_result(self, t_start: float) -> dict:
        return {
            "groups":          [],
            "scanned":         0,
            "embedded":        0,
            "embed_failures":  0,
            "elapsed_seconds": time.time() - t_start,
        }

    # ── Similarity matrix diagnostics ─────────────────────────────────────

    def _log_similarity_stats(self, sim, paths: list[str]) -> None:
        import numpy as np
        if sim.shape[0] < 2:
            return

        # On large libraries sampling is fine — a 50k x 50k matrix is 10 GB
        # in float32 and we don't need perfect precision for a diagnostic.
        n = sim.shape[0]
        if n <= 2000:
            off = sim[np.triu_indices_from(sim, k=1)]
        else:
            # Sample 200k pairs
            rng = np.random.default_rng(0)
            ii = rng.integers(0, n, size=200_000)
            jj = rng.integers(0, n, size=200_000)
            mask = ii < jj
            ii, jj = ii[mask], jj[mask]
            off = sim[ii, jj]

        if off.size == 0:
            return
        _log.info(
            "similarity matrix stats (%s):  min=%.4f  mean=%.4f  max=%.4f  p95=%.4f  p99=%.4f",
            "full" if n <= 2000 else "sampled 200k pairs",
            float(off.min()), float(off.mean()), float(off.max()),
            float(np.percentile(off, 95)), float(np.percentile(off, 99)),
        )

        # Top-5 strongest off-diagonal pairs (small N only — scan cost O(N²))
        if n <= 3000:
            import os
            tri = np.triu(sim, k=1)
            flat = tri.flatten()
            top_idx = np.argsort(-flat)[:5]
            _log.info("top-5 most-similar pairs:")
            for fi in top_idx:
                i, j = int(fi // n), int(fi % n)
                if i >= j or sim[i, j] <= 0:
                    continue
                _log.info(
                    "  %.4f   %r  <->  %r",
                    float(sim[i, j]),
                    os.path.basename(paths[i]),
                    os.path.basename(paths[j]),
                )

    # ── GPU path ──────────────────────────────────────────────────────────

    @staticmethod
    def _vram_mb(device) -> str:
        """Return a one-line VRAM summary for the log."""
        import torch
        res = torch.cuda.memory_reserved(device)  / 1024**2
        alc = torch.cuda.memory_allocated(device) / 1024**2
        return f"reserved={res:.0f} MB  allocated={alc:.0f} MB"

    def _compare_gpu(self, fps, valid, abort_flag):
        """Dispatch: CuPy fused kernel first, PyTorch offset-loop fallback."""
        try:
            import cupy as cp  # noqa: F401
            _get_cupy_kernel()  # compile now so failure is caught here
            return self._compare_cupy(fps, valid, abort_flag)
        except ModuleNotFoundError:
            _log.info(
                "CuPy not installed — AcoustID scan using the slower PyTorch path. "
                "Speed up with:  %s", _cupy_install_hint(),
            )
            return self._compare_torch(fps, valid, abort_flag)
        except Exception as exc:
            _log.info("CuPy kernel unavailable (%s: %s) — using PyTorch loop",
                      type(exc).__name__, exc)
            return self._compare_torch(fps, valid, abort_flag)

    def _compare_cupy(self, fps, valid, abort_flag):
        """
        Fused CUDA kernel path — one kernel call per (Q_tile × N_tile) pair.
        Each thread handles one (qi, ti) pair and loops over all offsets
        internally: no (Q, N, L) intermediate tensors are ever allocated.
        """
        import numpy as np
        import torch
        import cupy as cp

        device = torch.device("cuda:{0}".format(self._gpu_index))
        n, L   = fps.shape

        _log.info("[VRAM-DIAG] _compare_cupy entry  n=%d L=%d  %s", n, L, self._vram_mb(device))

        # int32 is sufficient — chromaprint values are uint32, bit patterns identical,
        # and the kernel uses (unsigned int) casts for XOR/popcount. Halves VRAM vs int64.
        with cp.cuda.Device(self._gpu_index):
            fps_cp   = cp.asarray(fps.view(np.int32))   # (n, L) int32  on GPU
            valid_cp = cp.asarray(valid)                  # (n, L) uint8  on GPU

        sim = torch.zeros((n, n), dtype=torch.float32, device=device)
        sim.fill_diagonal_(1.0)

        _log.info(
            "[VRAM-DIAG] after uploads  fps=%.0f MB  valid=%.0f MB  sim=%.0f MB  %s",
            fps_cp.nbytes / 1024**2, valid_cp.nbytes / 1024**2,
            sim.nbytes / 1024**2, self._vram_mb(device),
        )

        q_tile  = max(1, self._tile)
        db_tile = max(1, self._db_tile)
        kernel  = _get_cupy_kernel()
        BLOCK   = 16   # 16×16 = 256 threads per block

        total_tiles = (n + q_tile - 1) // q_tile
        for t_idx, q_start in enumerate(range(0, n, q_tile)):
            if abort_flag and abort_flag():
                return sim.cpu().numpy()
            q_end  = min(q_start + q_tile, n)
            Q      = q_end - q_start
            fp_q_cp = fps_cp  [q_start:q_end]   # (Q, L)
            vq_cp   = valid_cp[q_start:q_end]   # (Q, L)

            for n_start in range(0, n, db_tile):
                n_end   = min(n_start + db_tile, n)
                N       = n_end - n_start
                fp_t_cp = fps_cp  [n_start:n_end]   # (N, L)
                vt_cp   = valid_cp[n_start:n_end]   # (N, L)
                out_cp  = cp.empty((Q, N), dtype=cp.float32)

                grid = ((Q + BLOCK - 1) // BLOCK, (N + BLOCK - 1) // BLOCK)
                kernel(
                    grid, (BLOCK, BLOCK),
                    (fp_q_cp, fp_t_cp, vq_cp, vt_cp, out_cp,
                     np.int32(Q), np.int32(N), np.int32(L),
                     np.int32(self._max_offset), np.int32(MIN_OVERLAP_STEPS)),
                )

                sim[q_start:q_end, n_start:n_end] = torch.from_dlpack(out_cp)

            if (t_idx + 1) % max(1, total_tiles // 20) == 0:
                self._report(
                    "Comparing fingerprints… {0}/{1} tiles".format(t_idx + 1, total_tiles)
                )

        sim = torch.maximum(sim, sim.t())
        return sim.cpu().numpy()

    def _compare_torch(self, fps, valid, abort_flag):
        """PyTorch fallback — original offset for-loop."""
        import numpy as np
        import torch

        device = torch.device("cuda:{0}".format(self._gpu_index))
        n, L = fps.shape

        _log.info("[VRAM-DIAG] _compare_torch entry  n=%d L=%d  %s",
                  n, L, self._vram_mb(device))

        fps_t   = torch.from_numpy(fps.astype(np.int64, copy=False)).to(device)
        valid_t = torch.from_numpy(valid).to(device)
        _log.info("[VRAM-DIAG] after fps_t+valid_t  (fps_t=%.0f MB  valid_t=%.0f MB)  %s",
                  fps_t.nbytes / 1024**2, valid_t.nbytes / 1024**2, self._vram_mb(device))

        sim = torch.zeros((n, n), dtype=torch.float32, device=device)
        sim.fill_diagonal_(1.0)
        _log.info("[VRAM-DIAG] after sim alloc  (sim=%.0f MB)  %s",
                  sim.nbytes / 1024**2, self._vram_mb(device))

        max_offset = self._max_offset
        offsets = list(range(-max_offset, max_offset + 1))
        q_tile  = max(1, self._tile)
        db_tile = max(1, self._db_tile)

        total_tiles = (n + q_tile - 1) // q_tile
        for t_idx, q_start in enumerate(range(0, n, q_tile)):
            if abort_flag and abort_flag():
                return sim.cpu().numpy()
            q_end = min(q_start + q_tile, n)
            q  = fps_t  [q_start:q_end]   # (Q, L) int64
            qv = valid_t[q_start:q_end]   # (Q, L) uint8

            best = torch.zeros((q_end - q_start, n), dtype=torch.float32, device=device)

            for n_start in range(0, n, db_tile):
                n_end  = min(n_start + db_tile, n)
                t_db   = fps_t  [n_start:n_end]
                tv_db  = valid_t[n_start:n_end]

                for offset in offsets:
                    if offset >= 0:
                        k = offset
                        q_sub  = q    [:, :L - k] if k > 0 else q
                        qv_sub = qv   [:, :L - k] if k > 0 else qv
                        t_sub  = t_db [:, k:]     if k > 0 else t_db
                        tv_sub = tv_db[:, k:]     if k > 0 else tv_db
                    else:
                        k = -offset
                        q_sub  = q    [:, k:]
                        qv_sub = qv   [:, k:]
                        t_sub  = t_db [:, :L - k]
                        tv_sub = tv_db[:, :L - k]

                    if t_idx == 0 and n_start == 0 and offset == offsets[0]:
                        _log.info(
                            "[VRAM-DIAG] before xor  q_sub=%s t_sub=%s  %s",
                            tuple(q_sub.shape), tuple(t_sub.shape), self._vram_mb(device),
                        )

                    xor  = q_sub.unsqueeze(1) ^ t_sub.unsqueeze(0)
                    pop  = self._popcount64(xor)
                    ov   = qv_sub.unsqueeze(1) & tv_sub.unsqueeze(0)
                    ov_f = ov.to(torch.int64)

                    diff_bits   = (pop * ov_f).sum(dim=-1)
                    total_steps = ov_f.sum(dim=-1)
                    total_bits  = total_steps * 32

                    denom    = total_bits.clamp_min(1).to(torch.float32)
                    this_sim = 1.0 - diff_bits.to(torch.float32) / denom
                    this_sim = torch.where(
                        total_steps >= MIN_OVERLAP_STEPS,
                        this_sim,
                        torch.zeros_like(this_sim),
                    )
                    best[:, n_start:n_end] = torch.maximum(
                        best[:, n_start:n_end], this_sim
                    )

                    if t_idx == 0 and n_start == 0 and offset == offsets[0]:
                        _log.info("[VRAM-DIAG] peak within offset  %s", self._vram_mb(device))

                    del xor, pop, ov, ov_f, diff_bits, total_steps, total_bits, this_sim

            sim[q_start:q_end, :] = best
            del best

            if (t_idx + 1) % max(1, total_tiles // 20) == 0:
                self._report(
                    "Comparing fingerprints… {0}/{1} tiles".format(t_idx + 1, total_tiles)
                )

        sim = torch.maximum(sim, sim.t())
        return sim.cpu().numpy()

    @staticmethod
    def _popcount64(x):
        """Hamming weight of a torch int64 tensor, same shape out."""
        import torch
        # Ensure we're in int64 for the shifts; the chromaprint values fit in 32.
        m1 = 0x5555555555555555
        m2 = 0x3333333333333333
        m4 = 0x0F0F0F0F0F0F0F0F
        h1 = 0x0101010101010101
        x = x & 0xFFFFFFFF               # chromaprint ints are 32 bits
        x = x - ((x >> 1) & m1)
        x = (x & m2) + ((x >> 2) & m2)
        x = (x + (x >> 4)) & m4
        return (x * h1) >> 56

    # ── CPU fallback ──────────────────────────────────────────────────────

    def _compare_cpu(self, fps, valid, abort_flag):
        """
        numpy fallback. Uses the same tiled scheme as GPU but without torch.
        Significantly slower — warn the user before invoking.
        """
        import numpy as np

        n, L = fps.shape
        sim = np.zeros((n, n), dtype=np.float32)
        np.fill_diagonal(sim, 1.0)

        max_offset = self._max_offset
        offsets = list(range(-max_offset, max_offset + 1))
        tile = max(1, self._tile)

        total_tiles = (n + tile - 1) // tile
        for t_idx, q_start in enumerate(range(0, n, tile)):
            if abort_flag and abort_flag():
                return sim
            q_end = min(q_start + tile, n)
            q  = fps  [q_start:q_end].astype(np.uint32, copy=False)
            qv = valid[q_start:q_end]

            best = np.zeros((q_end - q_start, n), dtype=np.float32)

            for offset in offsets:
                if offset >= 0:
                    k = offset
                    q_sub  = q [:, : L - k] if k > 0 else q
                    qv_sub = qv[:, : L - k] if k > 0 else qv
                    t_sub  = fps  [:,  k:] if k > 0 else fps
                    tv_sub = valid[:,  k:] if k > 0 else valid
                else:
                    k = -offset
                    q_sub  = q [:,  k:]
                    qv_sub = qv[:,  k:]
                    t_sub  = fps  [:, : L - k]
                    tv_sub = valid[:, : L - k]

                # XOR: (Q, 1, L') ^ (1, N, L') — memory-heavy
                xor = q_sub[:, None, :].astype(np.uint32) ^ t_sub[None, :, :].astype(np.uint32)
                pop = _popcount32_np(xor)                    # (Q, N, L') int32
                ov  = qv_sub[:, None, :] & tv_sub[None, :, :]
                ov32 = ov.astype(np.int32)

                diff_bits = (pop * ov32).sum(axis=-1)
                total_steps = ov32.sum(axis=-1)
                total_bits = total_steps.astype(np.int64) * 32

                denom = np.maximum(total_bits, 1).astype(np.float32)
                this_sim = 1.0 - diff_bits.astype(np.float32) / denom
                this_sim = np.where(total_steps >= MIN_OVERLAP_STEPS, this_sim, 0.0).astype(np.float32)
                np.maximum(best, this_sim, out=best)

            sim[q_start:q_end, :] = best

            if (t_idx + 1) % max(1, total_tiles // 20) == 0:
                self._report(
                    "Comparing fingerprints (CPU)… {0}/{1} tiles".format(
                        t_idx + 1, total_tiles,
                    )
                )

        sim = np.maximum(sim, sim.T)
        return sim
