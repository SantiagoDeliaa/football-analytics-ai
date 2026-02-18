#!/usr/bin/env python3
"""
M5: Regression check — compares two stats JSON files and reports deltas.

Usage:
    python tools/validate_outputs.py baseline_stats.json new_stats.json
"""

import argparse
import json
import math
import sys


def safe_get(d, *keys, default=None):
    """Navigate nested dict safely."""
    current = d
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k, default)
        if current is default:
            return default
    return current


def heatmap_center_of_mass(heatmap_2d):
    """Compute center-of-mass (x_m, y_m) from a 2D heatmap grid."""
    if heatmap_2d is None:
        return None
    total = 0.0
    cx = 0.0
    cy = 0.0
    rows = len(heatmap_2d)
    cols = len(heatmap_2d[0]) if rows > 0 else 0
    if rows == 0 or cols == 0:
        return None
    for i in range(rows):
        for j in range(cols):
            v = float(heatmap_2d[i][j])
            x_m = (i + 0.5) * 105.0 / rows
            y_m = (j + 0.5) * 68.0 / cols
            cx += v * x_m
            cy += v * y_m
            total += v
    if total < 1e-9:
        return None
    return (cx / total, cy / total)


def classify_delta(delta, threshold_good=0.02, threshold_bad=0.05, lower_is_better=True):
    """Return emoji classification for a delta value."""
    if lower_is_better:
        if delta < -threshold_good:
            return "improved"
        elif delta > threshold_bad:
            return "regressed"
        else:
            return "similar"
    else:
        if delta > threshold_good:
            return "improved"
        elif delta < -threshold_bad:
            return "regressed"
        else:
            return "similar"


EMOJI = {
    "improved": "OK",
    "similar": "~",
    "regressed": "FAIL",
}


def main():
    parser = argparse.ArgumentParser(description="Compare two stats JSON files for regression.")
    parser.add_argument("baseline", help="Path to baseline stats JSON")
    parser.add_argument("new", help="Path to new stats JSON")
    args = parser.parse_args()

    with open(args.baseline, "r") as f:
        baseline = json.load(f)
    with open(args.new, "r") as f:
        new = json.load(f)

    results = []

    # 1. homography_valid_ratio (higher is better)
    bv = safe_get(baseline, "homography_telemetry", "homography_valid_ratio", default=None)
    nv = safe_get(new, "homography_telemetry", "homography_valid_ratio", default=None)
    if bv is not None and nv is not None:
        delta = nv - bv
        status = classify_delta(delta, lower_is_better=False)
        results.append(("homography_valid_ratio", f"{bv:.3f}", f"{nv:.3f}", f"{delta:+.3f}", status))

    # 2. fallback_ratio (lower is better)
    bf = safe_get(baseline, "homography_telemetry", "fallback_ratio", default=None)
    nf = safe_get(new, "homography_telemetry", "fallback_ratio", default=None)
    if bf is not None and nf is not None:
        delta = nf - bf
        status = classify_delta(delta, lower_is_better=True)
        results.append(("fallback_ratio", f"{bf:.3f}", f"{nf:.3f}", f"{delta:+.3f}", status))

    # 3. avg_reproj_error (lower is better)
    br = safe_get(baseline, "homography_telemetry", "avg_reproj_error", default=None)
    nr = safe_get(new, "homography_telemetry", "avg_reproj_error", default=None)
    if br is not None and nr is not None:
        delta = nr - br
        status = classify_delta(delta, lower_is_better=True)
        results.append(("avg_reproj_error", f"{br:.2f}", f"{nr:.2f}", f"{delta:+.2f}", status))

    # 4. mean_block_depth team1 (delta %)
    bd = safe_get(baseline, "metrics", "team1", "block_depth_m", "mean", default=None)
    nd = safe_get(new, "metrics", "team1", "block_depth_m", "mean", default=None)
    if bd is not None and nd is not None and bd != 0:
        delta_pct = (nd - bd) / abs(bd) * 100
        status = "similar" if abs(delta_pct) <= 5 else ("regressed" if abs(delta_pct) > 15 else "similar")
        results.append(("mean_block_depth_t1", f"{bd:.1f}", f"{nd:.1f}", f"{delta_pct:+.1f}%", status))

    # 5. mean_block_width team1 (delta %)
    bw = safe_get(baseline, "metrics", "team1", "block_width_m", "mean", default=None)
    nw = safe_get(new, "metrics", "team1", "block_width_m", "mean", default=None)
    if bw is not None and nw is not None and bw != 0:
        delta_pct = (nw - bw) / abs(bw) * 100
        status = "similar" if abs(delta_pct) <= 5 else ("regressed" if abs(delta_pct) > 15 else "similar")
        results.append(("mean_block_width_t1", f"{bw:.1f}", f"{nw:.1f}", f"{delta_pct:+.1f}%", status))

    # 6. heatmap COM team1
    bh = safe_get(baseline, "scouting_heatmaps", "team1", default=None)
    nh = safe_get(new, "scouting_heatmaps", "team1", default=None)
    b_com = heatmap_center_of_mass(bh)
    n_com = heatmap_center_of_mass(nh)
    if b_com is not None and n_com is not None:
        dist = math.sqrt((b_com[0] - n_com[0]) ** 2 + (b_com[1] - n_com[1]) ** 2)
        status = "improved" if dist < 2.0 else ("regressed" if dist > 5.0 else "similar")
        results.append((
            "heatmap_COM_team1",
            f"({b_com[0]:.0f},{b_com[1]:.0f})",
            f"({n_com[0]:.0f},{n_com[1]:.0f})",
            f"d={dist:.1f}m",
            status
        ))

    # 7. out_of_bounds_ratio (lower is better)
    bo = safe_get(baseline, "homography_telemetry", "avg_oob_pct", default=None)
    no = safe_get(new, "homography_telemetry", "avg_oob_pct", default=None)
    if bo is not None and no is not None:
        delta = no - bo
        status = classify_delta(delta, lower_is_better=True)
        results.append(("out_of_bounds_ratio", f"{bo:.3f}", f"{no:.3f}", f"{delta:+.3f}", status))

    # 8. formation consistency team1
    bf1 = safe_get(baseline, "formations", "team1", "most_common", default=None)
    nf1 = safe_get(new, "formations", "team1", "most_common", default=None)
    if bf1 is not None and nf1 is not None:
        status = "improved" if bf1 == nf1 else "regressed"
        results.append(("formation_t1", bf1, nf1, "==" if bf1 == nf1 else "!=", status))

    # Print report
    print("\n=== Regression Report ===")
    counts = {"improved": 0, "similar": 0, "regressed": 0}
    for name, old_val, new_val, delta_str, status in results:
        emoji = EMOJI[status]
        counts[status] += 1
        print(f"  {name:.<30s} {old_val:>12s} -> {new_val:>12s}  ({delta_str:>10s})  [{emoji}]")

    total = sum(counts.values())
    print("---")
    print(f"Overall: {counts['improved']}/{total} OK, {counts['similar']}/{total} ~, {counts['regressed']}/{total} FAIL")

    if counts["regressed"] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
