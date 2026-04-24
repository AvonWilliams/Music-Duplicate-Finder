# Music Duplicate Finder — clustering.py
# V2.0: extracted shared complete-linkage (clique) clustering from V1.9's
# local_inference.py so both the CLAP engine and the new AcoustID engine
# produce identically-shaped output through the same proven algorithm.
#
# The clustering guarantees every pair inside a group is >= unsure_threshold
# (the clique property). Union-Find was rejected in V1.9 because transitive
# closure over CLAP's "narrow cone" embedding space collapsed entire 2k-file
# libraries into single groups via bridge edges.
#
# Confidence tier is assigned from the weakest-link pairwise similarity
# inside a group, which is a floor guarantee. Previous code used max, which
# let one coincidental high-similarity edge label an otherwise-dubious group
# "certain".

from __future__ import annotations

import logging
import time
from typing import Any


def cluster_complete_linkage(
    sim,                        # (N, N) float32 numpy array, symmetric, diag=1
    paths: list[str],           # length N, parallel to sim rows/cols
    certain_threshold: float,
    likely_threshold: float,
    unsure_threshold: float,
    log: logging.Logger,
    log_prefix: str = "",       # e.g. "chromaprint" or "clap" for log tags
) -> list[dict[str, Any]]:
    """
    Run complete-linkage clustering on a similarity matrix and return a list
    of raw groups in the shape the scan_worker._finalise() path expects.

    Returns a list of dicts:
        {
            "confidence":      "certain" | "likely" | "unsure",
            "similarity":      float     # max pairwise (preserved for UI compat)
            "min_similarity":  float,
            "max_similarity":  float,
            "mean_similarity": float,
            "files":           list[str],
        }

    Pre-logs: delegated to caller (they already have the similarity matrix
    in hand and can log their own stats).

    Internal logs emitted here:
    - "[<prefix>] complete-linkage clustering: N pairs above T -> M groups
       formed via K successful merges  (elapsed=Xs)"
    - "[<prefix>] group size distribution: ..."
    - "[<prefix>] group confidence tiers: certain=... likely=... unsure=..."
    - "[<prefix>] top-5 tightest groups (by weakest-link similarity): ..."
    """
    import numpy as _np

    n = sim.shape[0]
    tag = f"[{log_prefix}] " if log_prefix else ""

    t_cluster0 = time.time()

    # ── Collect above-threshold pairs, strongest-first ────────────────────
    tri_mask = _np.triu(sim, k=1) >= unsure_threshold
    pair_i, pair_j = _np.where(tri_mask)
    if pair_i.size > 0:
        pair_vals = sim[pair_i, pair_j]
        order_desc = _np.argsort(-pair_vals, kind="stable")
        pair_i = pair_i[order_desc]
        pair_j = pair_j[order_desc]

    group_of = [-1] * n          # node idx -> group id (-1 == unassigned)
    groups_m: dict[int, list[int]] = {}
    _next_gid = [0]

    def _all_above(members_a, members_b, thresh):
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
            new_gid = _next_gid[0]
            _next_gid[0] += 1
            groups_m[new_gid] = [i, j]
            group_of[i] = new_gid
            group_of[j] = new_gid
            merges_done += 1
        elif gi == -1:
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
            if _all_above(groups_m[gi], groups_m[gj], unsure_threshold):
                if len(groups_m[gi]) >= len(groups_m[gj]):
                    dst, src = gi, gj
                else:
                    dst, src = gj, gi
                for _m in groups_m[src]:
                    group_of[_m] = dst
                groups_m[dst].extend(groups_m[src])
                del groups_m[src]
                merges_done += 1

    t_cluster = time.time() - t_cluster0
    log.info(
        "%scomplete-linkage clustering: %d candidate pairs above %.3f -> "
        "%d groups formed via %d successful merges  (elapsed=%.2fs)",
        tag, int(pair_i.size), unsure_threshold,
        len(groups_m), merges_done, t_cluster,
    )

    # ── Build output groups ───────────────────────────────────────────────
    raw_groups: list[dict[str, Any]] = []
    size_histogram: dict[int, int] = {}
    for _, members in groups_m.items():
        if len(members) < 2:
            continue
        members_sorted = sorted(members)
        pair_sims: list[float] = []
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
            "confidence":      confidence,
            "similarity":      max_sim,
            "min_similarity":  min_sim,
            "max_similarity":  max_sim,
            "mean_similarity": mean_sim,
            "files":           [paths[m] for m in members_sorted],
        })

        sz = len(members_sorted)
        size_histogram[sz] = size_histogram.get(sz, 0) + 1

    order = {"certain": 0, "likely": 1, "unsure": 2}
    raw_groups.sort(key=lambda g: (order.get(g["confidence"], 9), -g["similarity"]))

    # ── Diagnostic logging ────────────────────────────────────────────────
    if size_histogram:
        hist_str = ", ".join(
            "{0} files x {1}".format(sz, cnt)
            for sz, cnt in sorted(size_histogram.items())
        )
        log.info("%sgroup size distribution: %s", tag, hist_str)

    by_tier = {"certain": 0, "likely": 0, "unsure": 0}
    for g in raw_groups:
        by_tier[g["confidence"]] = by_tier.get(g["confidence"], 0) + 1
    log.info(
        "%sgroup confidence tiers: certain=%d  likely=%d  unsure=%d  (total=%d)",
        tag, by_tier["certain"], by_tier["likely"], by_tier["unsure"],
        len(raw_groups),
    )

    if raw_groups:
        top = sorted(raw_groups, key=lambda g: -g["min_similarity"])[:5]
        log.info("%stop-5 tightest groups (by weakest-link similarity):", tag)
        for g in top:
            log.info(
                "%s  tier=%s  size=%d  min=%.4f  mean=%.4f  max=%.4f",
                tag, g["confidence"], len(g["files"]),
                g["min_similarity"], g["mean_similarity"], g["max_similarity"],
            )
            for p in g["files"][:3]:
                log.info("%s      %s", tag, p)
            if len(g["files"]) > 3:
                log.info("%s      ... +%d more", tag, len(g["files"]) - 3)

    return raw_groups
