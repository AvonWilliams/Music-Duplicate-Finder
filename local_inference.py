# Music Duplicate Finder — local_inference.py
# CLAP-based audio fingerprinting + cosine similarity search running
# entirely on the local machine's GPU (or CPU fallback).
#
# Dependencies (all optional — detected at runtime):
#   torch          pip install torch --index-url https://download.pytorch.org/whl/cu121
#   transformers   pip install transformers
#   soundfile      pip install soundfile          # primary audio decoder
#   scipy          pip install scipy              # resampling
#
# Audio decoding strategy (V1.4):
#   1. Try soundfile (libsndfile): handles FLAC/WAV/OGG/MP3/AIFF/etc.
#   2. Fall back to ffmpeg subprocess: handles MKA/Matroska, WebM, and
#      anything else ffmpeg can decode. ffmpeg must be on PATH.
#
# torchaudio is no longer used because in torchaudio >= 2.7 it delegates
# to torchcodec, which cannot decode .mka (Matroska Audio) — see handoff
# diagnostic log.
#
# Clustering strategy (V1.9):
#   Complete-linkage (clique) clustering on the cosine similarity matrix.
#   Every pair within a group is guaranteed to be >= unsure_threshold.
#   Previous versions used Union-Find which does transitive closure and
#   catastrophically over-merged on realistic libraries — see
#   scan_worker log after the V1.8 determinism fix for the symptom
#   ("1195334 pairs unioned -> 1 groups" on a 2030-file corpus).
#
# If any Python dep is missing, load_model() raises MissingDependencyError
# with a clear install message rather than crashing Picard silently.

from __future__ import annotations

import os
import time
from typing import Any, Callable

from .diag import get_logger

_log = get_logger("local_inference")

# NOTE: heavy ML imports (numpy, torch, transformers, soundfile, scipy) are
# done lazily inside methods so the plugin can load and run in remote mode
# without any of them installed. Only load_model()/find_duplicates() will
# actually import them — and they'll raise MissingDependencyError with a
# clear install message if anything is missing.

# ── Dependency detection ───────────────────────────────────────────────────

class MissingDependencyError(RuntimeError):
    """Raised when a required Python package is not installed."""


def check_dependencies() -> list[str]:
    """Return a list of missing package names (empty = all good)."""
    # Note the module name != pip name for soundfile (importlib name is
    # "soundfile", pip package is also "soundfile" — same for scipy).
    missing = []
    for pkg in ("numpy", "torch", "transformers", "soundfile", "scipy"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def check_ffmpeg() -> bool:
    """Return True if the `ffmpeg` binary is callable on PATH."""
    import shutil
    return shutil.which("ffmpeg") is not None


def available_gpus() -> list[str]:
    """Return list of CUDA device names, or [] if torch/CUDA unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        return [
            f"{i}: {torch.cuda.get_device_name(i)}"
            for i in range(torch.cuda.device_count())
        ]
    except Exception:  # noqa: BLE001
        return []


# ══════════════════════════════════════════════════════════════════════════
# Main inference engine
# ══════════════════════════════════════════════════════════════════════════

class LocalInferenceEngine:
    """
    Loads a CLAP model onto the chosen GPU, embeds audio files, and
    returns duplicate groups using cosine-similarity clustering.

    Output from find_duplicates() is identical in shape to the JSON
    returned by ServerClient.scan() so ScanWorker can use either
    without branching on the result format.
    """

    CLAP_MODEL_ID = "laion/clap-htsat-unfused"
    TARGET_SR     = 48_000   # CLAP expects 48 kHz
    CHUNK_SEC     = 30       # use first 30 s of each track for embedding
    BATCH_SIZE    = 8        # files per GPU batch

    def __init__(
        self,
        gpu_index: int = 0,
        model_cache_dir: str | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ):
        self._gpu_index      = gpu_index
        self._cache_dir      = model_cache_dir
        self._progress       = progress_cb or (lambda msg: None)
        self._model          = None
        self._processor      = None
        self._device         = None
        self._model_loaded   = False
        # transformers has churned the CLAP processor's audio kwarg between
        # 'audios' (older) and 'audio' (newer, deprecates audios). We probe
        # on the first file and cache the winning name here.
        self._audio_kwarg: str | None = None

    # ── Public API ─────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """
        Download (first time) and load the CLAP model onto the GPU.
        Raises MissingDependencyError if torch/transformers/soundfile/scipy
        are not installed.
        """
        missing = check_dependencies()
        if missing:
            _log.error("load_model: missing dependencies: %s", missing)
            pkgs = " ".join(missing)
            raise MissingDependencyError(
                f"Local GPU inference requires these packages which are not "
                f"installed in Picard's environment:\n\n  pip install {pkgs}\n\n"
                f"Install them into the same Python environment that runs Picard, "
                f"then try again."
            )

        import torch
        from transformers import ClapModel, ClapProcessor

        # ── Force deterministic, full-precision math ─────────────────────
        # CLAP embeddings must be identical across runs for bit-identical
        # input files. TF32 matmul (the Ampere/Ada default) introduces
        # ~1e-4 jitter per op which compounds to ~1% cosine-similarity
        # drift across the ~8 transformer layers of CLAP-HTSAT — enough
        # to push bit-identical files from 1.0000 down to 0.98-0.99, and
        # to produce different results on successive runs of the same
        # scan. Disable TF32 and non-deterministic cuDNN kernels.
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32       = False
            torch.backends.cudnn.deterministic    = True
            torch.backends.cudnn.benchmark        = False
            _log.info("Disabled TF32 and enabled cuDNN determinism for reproducible embeddings")

        # Device selection
        if torch.cuda.is_available() and self._gpu_index < torch.cuda.device_count():
            self._device = torch.device(f"cuda:{self._gpu_index}")
            gpu_name = torch.cuda.get_device_name(self._gpu_index)
            _log.info("Using CUDA device cuda:%d (%s)", self._gpu_index, gpu_name)
            self._progress(f"Loading CLAP model onto {gpu_name}…")
        else:
            self._device = torch.device("cpu")
            _log.info(
                "No CUDA device available (cuda.is_available=%s, device_count=%s) — using CPU",
                torch.cuda.is_available(),
                torch.cuda.device_count() if torch.cuda.is_available() else 0,
            )
            self._progress("No CUDA GPU found — running on CPU (will be slow)…")

        kwargs: dict[str, Any] = {"cache_dir": self._cache_dir} if self._cache_dir else {}
        _log.info("Loading CLAP model id=%s cache_dir=%s", self.CLAP_MODEL_ID,
                  self._cache_dir or "(default HF cache)")

        self._progress("Downloading/loading CLAP weights (first run may take a moment)…")
        t0 = time.time()
        self._processor = ClapProcessor.from_pretrained(self.CLAP_MODEL_ID, **kwargs)
        self._model     = ClapModel.from_pretrained(self.CLAP_MODEL_ID, **kwargs)
        self._model     = self._model.to(self._device)
        self._model.eval()
        self._model_loaded = True
        _log.info("CLAP model loaded in %.2fs on %s", time.time() - t0, self._device)
        self._progress("CLAP model ready.")

    def find_duplicates(
        self,
        win_paths: list[str],
        certain_threshold: float,
        likely_threshold: float,
        unsure_threshold: float,
        abort_flag: Callable[[], bool] | None = None,
    ) -> dict:
        """
        Embed all files and return a dict matching the server response shape.
        Paths in the output are the original Windows paths — no remapping.
        """
        if not self._model_loaded:
            self.load_model()

        _abort = abort_flag or (lambda: False)
        t0     = time.time()
        _log.info(
            "find_duplicates: %d files, thresholds=(certain=%.3f likely=%.3f unsure=%.3f)",
            len(win_paths), certain_threshold, likely_threshold, unsure_threshold,
        )

        # ── Embed all files ───────────────────────────────────────────────
        self._progress(f"Embedding {len(win_paths)} files…")
        embeddings: dict[str, "np.ndarray"] = {}
        embed_failures: list[str] = []

        for i, path in enumerate(win_paths):
            if _abort():
                _log.info("embedding aborted at %d/%d", i, len(win_paths))
                return {
                    "groups":          [],
                    "scanned":         i,
                    "embedded":        len(embeddings),
                    "embed_failures":  len(embed_failures),
                    "elapsed_seconds": time.time() - t0,
                }
            self._progress(f"Embedding {i + 1} / {len(win_paths)}  —  {os.path.basename(path)}")
            emb = self._embed_file(path)
            if emb is not None:
                embeddings[path] = emb

                # ── Determinism canary (first successful embed only) ────
                # Embed the same file a second time and log the cosine
                # similarity. If math is deterministic, it should be exactly
                # 1.0000 (bit-for-bit). Anything < 0.9999 indicates residual
                # GPU non-determinism that cross-file comparisons will inherit.
                if len(embeddings) == 1:
                    import numpy as _np
                    emb2 = self._embed_file(path)
                    if emb2 is not None:
                        a = emb  / (_np.linalg.norm(emb)  + 1e-12)
                        b = emb2 / (_np.linalg.norm(emb2) + 1e-12)
                        self_cos = float(_np.dot(a, b))
                        if self_cos >= 0.99999:
                            _log.info(
                                "determinism canary: same file embedded twice "
                                "-> cos=%.6f (math is deterministic ✓)",
                                self_cos,
                            )
                        else:
                            _log.warning(
                                "determinism canary: same file embedded twice "
                                "-> cos=%.6f (expected 1.0000). GPU non-"
                                "determinism is still present — low cross-file "
                                "scores may be partly due to jitter, not only "
                                "real audio differences.",
                                self_cos,
                            )
            else:
                embed_failures.append(path)

        valid_paths = list(embeddings.keys())
        n = len(valid_paths)
        _log.info(
            "embedding complete: %d/%d successful (%d failed)",
            n, len(win_paths), len(embed_failures),
        )
        if embed_failures:
            _log.warning("first few embedding failures:")
            for p in embed_failures[:5]:
                _log.warning("  FAILED to embed: %r", p)

        if n < 2:
            _log.warning(
                "Only %d files embedded successfully — cannot compare "
                "(need ≥ 2). Returning empty result.", n,
            )
            return {
                "groups":          [],
                "scanned":         len(win_paths),
                "embedded":        n,
                "embed_failures":  len(embed_failures),
                "elapsed_seconds": time.time() - t0,
            }

        if _abort():
            return {
                "groups":          [],
                "scanned":         n,
                "embedded":        n,
                "embed_failures":  len(embed_failures),
                "elapsed_seconds": time.time() - t0,
            }

        # ── Cosine similarity matrix on GPU ───────────────────────────────
        self._progress(f"Computing pairwise similarity for {n} files…")
        import numpy as np
        import torch
        import torch.nn.functional as F

        matrix_np = np.stack([embeddings[p] for p in valid_paths])   # (n, d)
        mat = torch.from_numpy(matrix_np).to(self._device)
        mat = F.normalize(mat, dim=1)
        sim = torch.mm(mat, mat.T).cpu().numpy()                      # (n, n)

        # ── Diagnostic: log similarity stats ──────────────────────────────
        # Off-diagonal values only (exclude self-similarity which is always 1.0)
        mask = ~np.eye(n, dtype=bool)
        off_diag = sim[mask]
        if off_diag.size > 0:
            _log.info(
                "similarity matrix stats (off-diagonal):  min=%.4f  mean=%.4f  "
                "max=%.4f  p95=%.4f  p99=%.4f",
                float(off_diag.min()), float(off_diag.mean()),
                float(off_diag.max()),
                float(np.percentile(off_diag, 95)),
                float(np.percentile(off_diag, 99)),
            )
            # Log the top-5 highest non-self pairs
            tri = np.triu(sim, k=1)           # upper triangle, no self-pairs
            flat = [(tri[i, j], i, j) for i in range(n) for j in range(i + 1, n)]
            flat.sort(reverse=True)
            _log.info("top-5 most-similar pairs:")
            for s, i, j in flat[:5]:
                _log.info("  %.4f   %r  <->  %r",
                          s, os.path.basename(valid_paths[i]),
                          os.path.basename(valid_paths[j]))

            # Count how many pairs would pass each threshold
            _log.info(
                "pairs passing thresholds: certain(≥%.3f)=%d  likely(≥%.3f)=%d  "
                "unsure(≥%.3f)=%d  (total pairs=%d)",
                certain_threshold, int((tri >= certain_threshold).sum()),
                likely_threshold,  int((tri >= likely_threshold).sum()),
                unsure_threshold,  int((tri >= unsure_threshold).sum()),
                n * (n - 1) // 2,
            )

            # ── Same-filename diagnostic ──────────────────────────────────
            # Find every pair of files that share the same basename (i.e.
            # the user likely expects them to be duplicates) and log their
            # similarity scores. Lets us see at a glance which supposed-
            # duplicate pairs are scoring anomalously low.
            from collections import defaultdict
            by_name: dict[str, list[int]] = defaultdict(list)
            for idx, p in enumerate(valid_paths):
                by_name[os.path.basename(p)].append(idx)
            same_name_pairs: list[tuple[float, str, str, str]] = []
            for basename, idxs in by_name.items():
                if len(idxs) < 2:
                    continue
                for a in range(len(idxs)):
                    for b in range(a + 1, len(idxs)):
                        i, j = idxs[a], idxs[b]
                        score = float(sim[i, j])
                        same_name_pairs.append((
                            score, basename,
                            os.path.dirname(valid_paths[i]),
                            os.path.dirname(valid_paths[j]),
                        ))
            if same_name_pairs:
                same_name_pairs.sort()  # ascending — worst first
                _log.info(
                    "same-filename pairs (%d found) — ranked worst-first "
                    "(these are the ones the user probably expects to be dupes):",
                    len(same_name_pairs),
                )
                for s, bn, d1, d2 in same_name_pairs:
                    _log.info("  %.4f   %r", s, bn)
                    _log.info("           %s", d1)
                    _log.info("           %s", d2)

        # ── Cluster with complete-linkage (clique) clustering ─────────────
        # V1.9 FIX: previous versions used Union-Find, which does transitive
        # closure — if A~B and B~C are both above threshold, A and C end up in
        # the same group even when sim(A,C) is low. On a small test library
        # this is hidden; on a 2030-file real library with mean pair cos=0.54
        # (CLAP embeds music into a narrow cone of the unit sphere), 58% of
        # all pairs sit above unsure_threshold=0.50 and union-find collapses
        # the entire corpus into one giant connected component via a handful
        # of "bridge" edges. That's the "all songs look like the same song"
        # symptom.
        #
        # Complete-linkage clustering fixes this by requiring every pair
        # within a group to be ≥ threshold (clique property). A candidate
        # only joins an existing group if it is above-threshold with EVERY
        # current member; two groups merge only if every cross-pair is above
        # threshold. This eliminates chain-merging.
        #
        # Confidence tier is assigned from the group's MINIMUM pairwise
        # similarity (weakest link), which is guaranteed to be ≥ the
        # clustering threshold and honestly reflects group quality. The
        # previous code used MAX, which let one coincidental 0.97 edge label
        # an otherwise-dubious group "certain".
        import numpy as _np

        t_cluster0 = time.time()

        # Collect above-threshold pairs (upper triangle only) and sort
        # strongest-first so the clique is seeded from the most confident
        # edges and weaker edges only extend cliques when they're consistent.
        tri_mask = _np.triu(sim, k=1) >= unsure_threshold
        pair_i, pair_j = _np.where(tri_mask)
        if pair_i.size > 0:
            pair_vals = sim[pair_i, pair_j]
            order_desc = _np.argsort(-pair_vals, kind="stable")
            pair_i = pair_i[order_desc]
            pair_j = pair_j[order_desc]
            pair_vals = pair_vals[order_desc]

        group_of = [-1] * n          # node idx -> group id (-1 == unassigned)
        groups_m: dict = {}          # group id -> list[int] of member indices
        _next_gid = [0]              # wrapped in list so inner helper can bump

        def _all_above(members_a, members_b, thresh):
            # True iff every cross-pair sim(a,b) is >= thresh.
            for _a in members_a:
                row = sim[_a]
                for _b in members_b:
                    if _a == _b:
                        continue
                    if row[_b] < thresh:
                        return False
            return True

        merges_done = 0
        for idx_p in range(pair_i.size):
            i = int(pair_i[idx_p])
            j = int(pair_j[idx_p])
            gi = group_of[i]
            gj = group_of[j]

            if gi == -1 and gj == -1:
                # Seed a fresh group from this pair.
                new_gid = _next_gid[0]
                _next_gid[0] += 1
                groups_m[new_gid] = [i, j]
                group_of[i] = new_gid
                group_of[j] = new_gid
                merges_done += 1
            elif gi == -1:
                # Try to add i into j's group: clique-complete with all members.
                members = groups_m[gj]
                ok = True
                row_i = sim[i]
                for _m in members:
                    if row_i[_m] < unsure_threshold:
                        ok = False
                        break
                if ok:
                    group_of[i] = gj
                    members.append(i)
                    merges_done += 1
            elif gj == -1:
                # Try to add j into i's group.
                members = groups_m[gi]
                ok = True
                row_j = sim[j]
                for _m in members:
                    if row_j[_m] < unsure_threshold:
                        ok = False
                        break
                if ok:
                    group_of[j] = gi
                    members.append(j)
                    merges_done += 1
            elif gi != gj:
                # Two distinct groups — only merge if every cross-pair is ok.
                if _all_above(groups_m[gi], groups_m[gj], unsure_threshold):
                    # Absorb smaller into larger.
                    if len(groups_m[gi]) >= len(groups_m[gj]):
                        dst, src = gi, gj
                    else:
                        dst, src = gj, gi
                    for _m in groups_m[src]:
                        group_of[_m] = dst
                    groups_m[dst].extend(groups_m[src])
                    del groups_m[src]
                    merges_done += 1
            # else: gi == gj, already together, nothing to do.

        t_cluster = time.time() - t_cluster0
        _log.info(
            "complete-linkage clustering: %d candidate pairs above %.3f -> "
            "%d groups formed via %d successful merges  (elapsed=%.2fs)",
            int(pair_i.size), unsure_threshold,
            len(groups_m), merges_done, t_cluster,
        )

        # ── Build output groups ───────────────────────────────────────────
        # For each group compute min / mean / max pairwise similarity.
        # Confidence tier is determined by MIN (weakest link in the clique).
        raw_groups = []
        size_histogram: dict = {}
        for _, members in groups_m.items():
            if len(members) < 2:
                continue
            members_sorted = sorted(members)
            pair_sims = []
            for a_idx in range(len(members_sorted)):
                a = members_sorted[a_idx]
                row = sim[a]
                for b_idx in range(a_idx + 1, len(members_sorted)):
                    pair_sims.append(float(row[members_sorted[b_idx]]))
            min_sim  = min(pair_sims)
            max_sim  = max(pair_sims)
            mean_sim = sum(pair_sims) / len(pair_sims)

            if min_sim >= certain_threshold:
                confidence = "certain"
            elif min_sim >= likely_threshold:
                confidence = "likely"
            else:
                confidence = "unsure"

            raw_groups.append({
                "confidence": confidence,
                "similarity": max_sim,     # preserved for existing UI display
                "min_similarity":  min_sim,
                "max_similarity":  max_sim,
                "mean_similarity": mean_sim,
                "files":      [valid_paths[m] for m in members_sorted],
            })

            sz = len(members_sorted)
            size_histogram[sz] = size_histogram.get(sz, 0) + 1

        # Sort: certain first, then by similarity
        order = {"certain": 0, "likely": 1, "unsure": 2}
        raw_groups.sort(key=lambda g: (order.get(g["confidence"], 9), -g["similarity"]))

        # ── Diagnostic logging: group size histogram + per-tier counts ────
        if size_histogram:
            hist_str = ", ".join(
                "{0} files x {1}".format(sz, cnt)
                for sz, cnt in sorted(size_histogram.items())
            )
            _log.info("group size distribution: %s", hist_str)

        by_tier: dict = {"certain": 0, "likely": 0, "unsure": 0}
        for g in raw_groups:
            by_tier[g["confidence"]] = by_tier.get(g["confidence"], 0) + 1
        _log.info(
            "group confidence tiers: certain=%d  likely=%d  unsure=%d  (total=%d)",
            by_tier["certain"], by_tier["likely"], by_tier["unsure"],
            len(raw_groups),
        )

        # Top-5 strongest groups (by min_similarity) — very useful for
        # eyeballing whether clustering looks sane.
        if raw_groups:
            top = sorted(raw_groups, key=lambda g: -g["min_similarity"])[:5]
            _log.info("top-5 tightest groups (by weakest-link similarity):")
            for g in top:
                _log.info(
                    "  tier=%s  size=%d  min=%.4f  mean=%.4f  max=%.4f",
                    g["confidence"], len(g["files"]),
                    g["min_similarity"], g["mean_similarity"], g["max_similarity"],
                )
                for p in g["files"][:3]:
                    _log.info("      %s", p)
                if len(g["files"]) > 3:
                    _log.info("      ... +%d more", len(g["files"]) - 3)

        # Warn if unsure_threshold is below the mean pair similarity: at that
        # point the threshold is no longer discriminating duplicates from
        # random music, and "unsure" groups will be largely noise.
        if off_diag.size > 0:
            _mean_off = float(off_diag.mean())
            if unsure_threshold <= _mean_off + 0.05:
                _log.warning(
                    "unsure_threshold (%.3f) is at or below the mean pair "
                    "similarity (%.3f). CLAP embeds music into a narrow cone "
                    "where unrelated tracks routinely score ~%.2f, so any "
                    "'unsure' groups surfaced are likely false positives. "
                    "Consider raising unsure_threshold to ~%.2f or higher.",
                    unsure_threshold, _mean_off, _mean_off,
                    min(0.85, _mean_off + 0.20),
                )

        elapsed = time.time() - t0
        _log.info(
            "clustering complete: %d candidate pairs -> %d groups  (elapsed=%.2fs)",
            int(pair_i.size), len(raw_groups), elapsed,
        )

        return {
            "groups":          raw_groups,
            "scanned":         len(win_paths),
            "embedded":        n,
            "embed_failures":  len(embed_failures),
            "elapsed_seconds": elapsed,
        }

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_audio(self, path: str) -> "tuple[np.ndarray, int]":
        """
        Load audio and return (mono_float32_waveform, sample_rate).

        Strategy (V1.4):
          1. Try soundfile (libsndfile) — fast, supports most common formats.
          2. If that fails (e.g. .mka / Matroska / WebM), fall back to an
             ffmpeg subprocess that decodes directly to mono f32le at the
             target sample rate.

        Raises RuntimeError on unrecoverable failures.
        """
        import numpy as np

        # ── Attempt 1: soundfile ─────────────────────────────────────────
        sf_exc: Exception | None = None
        try:
            import soundfile as sf
            waveform, sr = sf.read(path, dtype="float32", always_2d=False)
            # sf returns 1-D for mono, 2-D (samples, channels) for multichannel
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1).astype(np.float32)
            else:
                waveform = waveform.astype(np.float32, copy=False)
            if waveform.size == 0:
                raise RuntimeError("soundfile returned 0 samples")
            return waveform, int(sr)
        except Exception as exc:  # noqa: BLE001 — libsndfile raises RuntimeError for unsupported formats
            sf_exc = exc
            _log.debug(
                "soundfile failed for %r: %s: %s — trying ffmpeg fallback",
                path, type(exc).__name__, exc,
            )

        # ── Attempt 2: ffmpeg subprocess ─────────────────────────────────
        import subprocess
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel", "error",
            "-i", path,
            "-t", str(self.CHUNK_SEC),     # cap duration, saves decode work
            "-ac", "1",                     # mix to mono
            "-ar", str(self.TARGET_SR),     # resample to CLAP's target SR
            "-f", "f32le",                  # raw 32-bit float little-endian
            "-",                            # stdout
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, check=True, timeout=60,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg not found on PATH. soundfile could not decode this "
                "file (%s: %s) and ffmpeg is required as the fallback decoder. "
                "Install ffmpeg: `sudo apt install ffmpeg` on Debian/Ubuntu, "
                "`brew install ffmpeg` on macOS, or download from "
                "https://ffmpeg.org/ on Windows."
                % (type(sf_exc).__name__ if sf_exc else "unknown",
                   str(sf_exc) if sf_exc else "n/a")
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                "ffmpeg exited %d decoding %r: %s"
                % (exc.returncode, path, stderr[:500])
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "ffmpeg timed out decoding %r after 60s" % path
            ) from exc

        waveform = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        if waveform.size == 0:
            raise RuntimeError("ffmpeg produced 0 bytes of audio for %r" % path)
        # ffmpeg already resampled + mono-mixed, so sr == TARGET_SR
        return waveform, self.TARGET_SR

    def _call_processor(self, waveform):
        """
        Invoke the CLAP processor with the correct audio keyword argument.

        transformers 4.x has churned this name:
          - Older versions:  processor(audios=waveform, ...)
          - Newer versions:  processor(audio=waveform, ...)
          - The transitional release raises ValueError on `audios=` telling
            callers to switch to `audio=`.

        Probe once on the first call, cache the winner on self, use it for
        every subsequent file.
        """
        kwargs = {"return_tensors": "pt", "sampling_rate": self.TARGET_SR}

        if self._audio_kwarg is not None:
            return self._processor(**{self._audio_kwarg: waveform}, **kwargs)

        # First call — probe new name first (the one the current deprecation
        # path pushes callers toward).
        try:
            out = self._processor(audio=waveform, **kwargs)
            self._audio_kwarg = "audio"
            _log.info("ClapProcessor accepts 'audio=' (newer transformers API)")
            return out
        except (TypeError, ValueError) as exc:
            # TypeError = unknown kwarg on old versions
            # ValueError = some transitional versions with strict kwarg checking
            _log.debug(
                "processor(audio=...) rejected (%s: %s) — falling back to "
                "processor(audios=...)", type(exc).__name__, exc,
            )

        out = self._processor(audios=waveform, **kwargs)
        self._audio_kwarg = "audios"
        _log.info("ClapProcessor requires 'audios=' (older transformers API)")
        return out

    def _embed_file(self, path: str) -> "np.ndarray | None":
        """Load audio, compute CLAP embedding, return unit-norm numpy vector."""
        try:
            import numpy as np
            import torch
            import torch.nn.functional as F

            # ── Seed every RNG before each embed (determinism) ───────────
            # Some components downstream (ClapFeatureExtractor's SpecAugment,
            # possibly StochasticDepth/DropPath in the HTSAT audio tower)
            # use random number generators at inference time even with
            # model.eval(). Seeding identically per call guarantees that
            # same input -> same embedding, regardless of what stochastic
            # code runs inside the processor/model.
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            np.random.seed(0)
            import random as _random
            _random.seed(0)

            waveform, sr = self._load_audio(path)

            # Resample to 48 kHz if needed (ffmpeg branch already did this)
            if sr != self.TARGET_SR:
                import scipy.signal
                from math import gcd
                g = gcd(int(sr), int(self.TARGET_SR))
                up   = self.TARGET_SR // g
                down = int(sr) // g
                waveform = scipy.signal.resample_poly(waveform, up, down).astype(np.float32)

            # Truncate to CHUNK_SEC seconds
            max_samples = self.CHUNK_SEC * self.TARGET_SR
            waveform = waveform[:max_samples]

            # Feed to CLAP processor (uses adapter to handle transformers
            # API churn between 'audios=' and 'audio=')
            inputs = self._call_processor(waveform)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                features = self._model.get_audio_features(**inputs)

            # transformers has churned get_audio_features() return type too:
            #   - Older versions: returns a raw torch.Tensor of shape (1, D)
            #   - Newer versions: returns BaseModelOutputWithPooling with a
            #     `.audio_embeds` / `.pooler_output` / `.last_hidden_state`
            #     attribute carrying the actual tensor.
            if not isinstance(features, torch.Tensor):
                for attr in ("audio_embeds", "pooler_output", "last_hidden_state"):
                    candidate = getattr(features, attr, None)
                    if isinstance(candidate, torch.Tensor):
                        _log.debug(
                            "unwrapping get_audio_features() output: using .%s "
                            "(type=%s)", attr, type(features).__name__,
                        )
                        features = candidate
                        break
                else:
                    # Fall back to iterating via __getitem__(0) the way HF
                    # containers typically support.
                    try:
                        features = features[0]
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(
                            "get_audio_features() returned %s with no known "
                            "tensor attribute (audio_embeds / pooler_output / "
                            "last_hidden_state). Attrs: %s"
                            % (type(features).__name__,
                               [a for a in dir(features) if not a.startswith("_")])
                        ) from exc

            # If last_hidden_state (shape (B, T, D)), mean-pool over time
            if features.dim() == 3:
                features = features.mean(dim=1)

            # L2-normalise and return as CPU numpy
            features = F.normalize(features, dim=-1)
            return features.squeeze(0).cpu().numpy()

        except Exception as exc:  # noqa: BLE001
            # Log but don't crash — skip unreadable files
            _log.warning("could not embed %r: %s: %s",
                         path, type(exc).__name__, exc)
            return None
