"""Rebuild three paper figures on the 2026-09-15 rerun matrix, which
supersedes every earlier result pool (n=75, n=45, the 62-pair abstention
subset, 2026-09-08_sketch_vs_roma_bench and every 2026-09-14_* pool).

Reads, read-only, from results/2026-09-15_rerun_matrix/:

  matrix_canny_uavscenes_shipped_cc1_gate.jsonl     114 rows
  matrix_distill_uavscenes_shipped_cc1_gate.jsonl   114 rows
  matrix_learned_uavscenes_shipped_cc1_gate.jsonl   114 rows
  matrix_roma_uavscenes.jsonl                       114 rows

Nothing else is read. The UAV-VisLoc files in that directory carry
gt_in_estimator: true and citable: false, are not citable, and no panel here
uses them.

Naming. Three drawing arms now exist, so "the sketch matcher" is ambiguous.
In this file, and in the paper, the sketch matcher is the CANNY arm: the
shipped structure-based drawer, the arm the ancestor generators carried as
r["classical"]. "distill" and "learned" are the two trained drawers. "dense"
is RoMa v2.

Panels:

  PANEL 1  fig_gate_tradeoff_loso.pdf
      Coverage against median error for the confidence signal, per drawing
      arm, on UAVScenes.
      x = coverage, the share of that arm's answers admitted by a peak_ratio
      threshold; y = median position error of the admitted answers, log axis.
      The curve for an arm is that arm's peak_ratio swept over every distinct
      value it produced. One filled marker per arm is the leave-one-scene-out
      operating point: the four UAVScenes scenes (amtown01, amvalley01,
      hkairport, hkisland01) are held out one at a time; on the other three
      the lowest threshold admitting no answer worse than 30 m is chosen and
      applied unchanged to the held-out scene; the four held-out results are
      pooled. No threshold is shared across arms, and none can be: the peak
      ratio has a different range on each arm (see PEAK RATIO RANGES in the
      provenance file).

  PANEL 2  fig_abstention_terrain_pp.pdf
      Where RoMa v2 does and does not produce a geometric fit, by terrain
      class, over the same 114 pairs. Two bands, no threshold enters the
      figure. Bars carry 95 percent Wilson intervals on the fit-attempted
      share.

  PANEL 3  fig_reverse_scatter_uniform_pp.pdf
      Per-pair position error, RoMa v2 against the sketch matcher, on the
      pairs both answer. Log axes, equal aspect, dashed line of equality.
      Marker shape and fill give the terrain class.

Terrain class comes from the pid prefix before the first underscore:
amtown01 -> town, hkairport* -> airport, hkisland01 -> island,
amvalley01 -> valley. No terrain field exists on the rerun rows.

No mask is downsampled anywhere in this file, so lab retraction R15
(INTER_AREA, never nearest neighbour) has nothing to bite on here.

Outputs, all into this directory. Nothing in report/figures/ is written,
overwritten or read:

  fig_gate_tradeoff_loso.pdf            (+ .png preview)
  fig_abstention_terrain_pp.pdf         (+ .png preview)
  fig_reverse_scatter_uniform_pp.pdf    (+ .png preview)
  FIGURE_NUMBERS_2026-09-15.json        every number behind every panel

Citability: no run.json exists for this measurement. Per the project record-keeping policy,
section 1 nothing here is citable until that record exists.

CPU only, matplotlib Agg, vector PDF, 3.4 in single column, 8 pt type.

Usage: python make_figures_2026_09_15.py
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # never touch the GPU

import collections
import json
import math
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "2026-09-15_rerun_matrix"))
OUT_DIR = HERE

BAND_M = 30.0            # the error boundary the operating point is picked at
SCENES = ["amtown01", "amvalley01", "hkairport", "hkisland01"]
TERRAIN = {"amtown01": "town", "hkairport": "airport",
           "hkisland01": "island", "amvalley01": "valley"}
TERR = ["town", "airport", "island", "valley"]

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

NUM = {
    "_sources": {
        "sketch_arms": ["matrix_%s_uavscenes_shipped_cc1_gate.jsonl" % a
                        for a in ("canny", "distill", "learned")],
        "dense_arm": "matrix_roma_uavscenes.jsonl",
        "directory": "results/2026-09-15_rerun_matrix",
    },
    "_superseded_and_not_read": [
        "results/2026-09-08_sketch_vs_roma_bench/*",
        "results/2026-09-14_*",
        "any n=75, n=45 or 62-pair abstention pool",
    ],
    "_citability": ("no run.json exists for this measurement; per "
                    "the project record-keeping policy nothing here is citable "
                    "until that record exists"),
}


def jsonl(name):
    with open(os.path.join(SRC, name)) as f:
        return [json.loads(l) for l in f if l.strip()]


def scene_of(pid):
    return pid.split("_")[0]


def terrain_of(pid):
    return TERRAIN[scene_of(pid)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100.0 * max(0.0, c - h), 100.0 * min(1.0, c + h))


def save(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p)
    fig.savefig(p[:-4] + ".png", dpi=300)
    plt.close(fig)
    print("wrote", p)
    return name   # file name only: no absolute paths in released output


# ===========================================================================
# LOAD
# ===========================================================================
ARMS = ["canny", "distill", "learned"]
ARM_LABEL = {"canny": "sketch matcher", "distill": "distilled drawer",
             "learned": "learned drawer"}

sketch = {}
for a in ARMS:
    rows = jsonl("matrix_%s_uavscenes_shipped_cc1_gate.jsonl" % a)
    assert len(rows) == 114, (a, len(rows))
    # `answered` is the population filter. `answered_gated` is the shipped
    # 1.10 gate's own decision and must never enter here: it would silently
    # collapse canny to 33 rows and make every coverage figure wrong.
    ans = [r for r in rows if r["answered"]]
    assert all(r["shipped_err_m"] is not None and r["peak_ratio"] is not None
               for r in ans)
    sketch[a] = ans

dense = jsonl("matrix_roma_uavscenes.jsonl")
assert len(dense) == 114

NUM["population"] = {
    "uavscenes_pairs": 114,
    "answered_per_arm": {a: len(sketch[a]) for a in ARMS},
    "answered_note": ("learned answers 113 of 114; "
                      "hkairport_gnss_evening_000234_rtku has answered:false "
                      "with null error and null peak_ratio, and is excluded "
                      "from every learned quantity"),
    "dense_answered": sum(1 for r in dense if r["roma"]["answered"]),
    "peak_ratio_range_per_arm": {
        a: [round(min(r["peak_ratio"] for r in sketch[a]), 4),
            round(max(r["peak_ratio"] for r in sketch[a]), 4)] for a in ARMS},
    "peak_ratio_range_note": (
        "the shipped fixed gate of 1.10 is a canny-arm quantity. The distill "
        "and learned arms never reach 1.10 on any of the 114 pairs, so that "
        "threshold admits nothing from them and no threshold value is "
        "comparable across arms. Coverage is the only shared axis."),
    "ungated_median_err_m": {
        a: round(st.median([r["shipped_err_m"] for r in sketch[a]]), 2)
        for a in ARMS},
    "ungated_over_band": {
        a: sum(1 for r in sketch[a] if r["shipped_err_m"] > BAND_M)
        for a in ARMS},
}


# ===========================================================================
# PANEL 1 -- coverage against median error, per arm, with the LOSO point
# ===========================================================================
MIN_ADMITTED = 5         # a median over fewer than five answers is not drawn


def sweep(rows):
    """Every distinct peak_ratio as a threshold. Returns coverage %, median
    error, and the count worse than the band, one entry per threshold.

    Thresholds admitting fewer than MIN_ADMITTED answers are dropped: their
    median is a median of two or three numbers and swings several metres
    between neighbouring thresholds. The cut is stated in the caption."""
    n = len(rows)
    out = []
    for t in sorted(set(r["peak_ratio"] for r in rows)):
        adm = [r["shipped_err_m"] for r in rows if r["peak_ratio"] >= t]
        if len(adm) < MIN_ADMITTED:
            continue
        out.append({"threshold": round(t, 4),
                    "coverage_pct": 100.0 * len(adm) / n,
                    "n_admitted": len(adm),
                    "median_err_m": st.median(adm),
                    "n_over_band": sum(1 for e in adm if e > BAND_M)})
    return out


def loso(rows):
    """Leave one scene out. On the other scenes pick the LOWEST threshold that
    admits nothing worse than the band; apply it unchanged to the held-out
    scene; pool the four held-out results."""
    pooled, per_scene = [], {}
    for held in SCENES:
        train = [r for r in rows if scene_of(r["pid"]) != held]
        test = [r for r in rows if scene_of(r["pid"]) == held]
        thr = None
        for t in sorted(set(r["peak_ratio"] for r in train)):
            adm = [r for r in train if r["peak_ratio"] >= t]
            if adm and all(r["shipped_err_m"] <= BAND_M for r in adm):
                thr = t
                break
        adm_test = [] if thr is None else [r for r in test
                                           if r["peak_ratio"] >= thr]
        per_scene[held] = {
            "threshold": None if thr is None else round(thr, 4),
            "held_out_n": len(test),
            "admitted": len(adm_test),
            "over_band": sum(1 for r in adm_test
                             if r["shipped_err_m"] > BAND_M),
        }
        pooled += adm_test
    errs = [r["shipped_err_m"] for r in pooled]
    return {
        "per_scene": per_scene,
        "n_pool": len(rows),
        "admitted": len(pooled),
        "coverage_pct": 100.0 * len(pooled) / len(rows),
        "median_err_m": round(st.median(errs), 2) if errs else None,
        "over_band": sum(1 for e in errs if e > BAND_M),
        "max_err_m": round(max(errs), 1) if errs else None,
    }


STYLE = {"canny": ("-", "o", "black"),
         "distill": ("--", "s", "0.45"),
         "learned": (":", "^", "black")}

NUM["panel1_gate_tradeoff"] = {"sweep": {}, "loso": {}}
fig, ax = plt.subplots(figsize=(3.4, 2.7))
handles, op_lines = [], []
for a in ARMS:
    sw = sweep(sketch[a])
    lo = loso(sketch[a])
    NUM["panel1_gate_tradeoff"]["sweep"][a] = [
        {k: (round(v, 2) if isinstance(v, float) else v)
         for k, v in s.items()} for s in sw]
    NUM["panel1_gate_tradeoff"]["loso"][a] = lo
    ls, mk, col = STYLE[a]
    ax.plot([s["coverage_pct"] for s in sw], [s["median_err_m"] for s in sw],
            ls=ls, lw=1.0, color=col, zorder=2)
    ax.scatter([lo["coverage_pct"]], [lo["median_err_m"]], marker=mk, s=42,
               facecolor=col, edgecolor="black", linewidth=0.7, zorder=5)
    handles.append(matplotlib.lines.Line2D(
        [], [], ls=ls, lw=1.0, color=col, marker=mk, ms=5.0,
        markerfacecolor=col, markeredgecolor="black", markeredgewidth=0.7,
        label="%s ($n$=%d)" % (ARM_LABEL[a], len(sketch[a]))))
    op_lines.append("%s  %d/%d, %d over 30 m"
                    % (ARM_LABEL[a].split()[0], lo["admitted"], lo["n_pool"],
                       lo["over_band"]))

ax.set_yscale("log")
ax.set_xlim(0, 104)
ax.set_ylim(0.8, 40)
ax.set_xlabel("coverage: answers admitted (%)")
ax.set_ylabel("median error of the\nadmitted answers (m)")
ax.set_yticks([1, 2, 5, 10, 20])
ax.set_yticklabels(["1", "2", "5", "10", "20"])
ax.legend(handles=handles, loc="upper left", frameon=False, handlelength=2.4,
          fontsize=6.6, borderpad=0.2, labelspacing=0.3)
# The per-arm admitted counts live in the caption, not here: every corner of
# this axes is crossed by a curve at some coverage, so an inline block would
# sit on the data. op_lines is kept for the caption and the record.
ax.text(0.99, 0.02, "marker: leave-one-scene-out operating point",
        transform=ax.transAxes, fontsize=6.2, ha="right", va="bottom")
NUM["panel1_gate_tradeoff"]["operating_point_lines_for_the_caption"] = op_lines
ax.spines[["top", "right"]].set_visible(False)
NUM["panel1_gate_tradeoff"]["path"] = save(fig, "fig_gate_tradeoff_loso.pdf")
NUM["panel1_gate_tradeoff"]["axes"] = {
    "x": "coverage, percent of that arm's answered pairs admitted",
    "y": "median position error of the admitted answers, metres, log",
    "y_limits_m": [0.8, 40],
    "curve": ("one point per distinct peak_ratio value that arm produced, "
              "restricted to thresholds admitting at least %d answers"
              % MIN_ADMITTED),
    "min_admitted_for_a_plotted_point": MIN_ADMITTED,
}


# ===========================================================================
# PANEL 2 -- where the dense matcher produces no fit, by terrain
# ===========================================================================
GROUPS = TERR + ["all"]
counts = {}
for g in GROUPS:
    rr = dense if g == "all" else [r for r in dense
                                   if terrain_of(r["pid"]) == g]
    fitted = [r for r in rr if r["roma"]["answered"]]
    nofit = [r for r in rr if not r["roma"]["answered"]]
    counts[g] = {
        "n": len(rr),
        "fit_attempted": len(fitted),
        "no_fit": len(nofit),
        # the two mechanisms behind a no-fit, recomputed rather than quoted
        "no_fit_few_correspondences": sum(1 for r in nofit
                                          if r["roma"]["n_conf"] < 2),
        "no_fit_few_inliers": sum(1 for r in nofit
                                  if r["roma"]["n_conf"] >= 2
                                  and r["roma"]["n_inliers"] < 2),
        "fit_attempted_wilson_pct": [round(v) for v in
                                     wilson(len(fitted), len(rr))],
    }
assert (counts["all"]["no_fit_few_correspondences"]
        + counts["all"]["no_fit_few_inliers"] == counts["all"]["no_fit"])

fig, ax = plt.subplots(figsize=(3.4, 2.5))
xs = list(range(len(GROUPS)))
bands = {
    "no fit": [100.0 * counts[g]["no_fit"] / counts[g]["n"] for g in GROUPS],
    "fit attempted": [100.0 * counts[g]["fit_attempted"] / counts[g]["n"]
                      for g in GROUPS],
}
bottom = [0.0] * len(GROUPS)
for key, color, hatch in [("no fit", "0.25", "xxx"),
                          ("fit attempted", "0.92", "")]:
    ax.bar(xs, bands[key], 0.62, bottom=bottom, facecolor=color,
           edgecolor="black", linewidth=0.6, hatch=hatch, label=key)
    bottom = [b + v for b, v in zip(bottom, bands[key])]

for i, g in enumerate(GROUPS):
    lo, hi = wilson(counts[g]["fit_attempted"], counts[g]["n"])
    x = i + 0.40
    ax.plot([x, x], [100 - hi, 100 - lo], color="black", lw=1.0,
            solid_capstyle="butt")
    ax.plot([x - 0.06, x + 0.06], [100 - hi] * 2, color="black", lw=1.0)
    ax.plot([x - 0.06, x + 0.06], [100 - lo] * 2, color="black", lw=1.0)
    ax.text(i, 101.5, "%d/%d" % (counts[g]["fit_attempted"], counts[g]["n"]),
            ha="center", va="bottom", fontsize=6.5)

ax.set_xticks(xs)
ax.set_xticklabels(["%s\n$n$=%d" % (g, counts[g]["n"]) for g in GROUPS])
ax.set_ylabel("share of pairs (%)")
ax.set_ylim(0, 108)
ax.axvline(3.5, color="black", lw=0.5, ls=":")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=2,
          frameon=False, handlelength=1.5, columnspacing=1.0)
ax.spines[["top", "right"]].set_visible(False)
NUM["panel2_abstention_terrain"] = {
    "path": save(fig, "fig_abstention_terrain_pp.pdf"),
    "counts": counts,
    "no_threshold_enters_this_figure": True,
    "sketch_arm_answer_counts_for_the_caption": {
        a: len(sketch[a]) for a in ARMS},
}


# ===========================================================================
# PANEL 3 -- per-pair error, dense against the sketch matcher
# ===========================================================================
canny_by_pid = {r["pid"]: r for r in sketch["canny"]}

# The two arms must be scored on the same pair, the same crop size and the
# same jitter, or the scatter compares different problems. The ancestor
# pipelines treated this as load-bearing: merge_ppfix.py crop-verified, and
# results/2026-09-15_windowed_roma/ existed only to match the boxes. Checked
# here per pid rather than assumed.
assert {r["pid"] for r in dense} == set(canny_by_pid), \
    "pid sets differ between the dense and sketch records"
for _d in dense:
    _c = canny_by_pid.get(_d["pid"])
    if _c is None:
        continue
    assert _d["region_m"] == _c["region_m"], (_d["pid"], "region_m")
    assert _d["jitter"] == _c["jitter"], (_d["pid"], "jitter")
    assert _d["eff"] == _c["eff"], (_d["pid"], "eff")
    assert _d["convention"] == "principal_point", (_d["pid"], "convention")
    assert _d["roma_crop"] == "shared", (_d["pid"], "roma_crop")

both = [{"pid": r["pid"],
         "terrain": terrain_of(r["pid"]),
         "sketch_err_m": canny_by_pid[r["pid"]]["shipped_err_m"],
         "dense_err_m": r["roma"]["err_m"]}
        for r in dense
        if r["roma"]["answered"] and r["pid"] in canny_by_pid]

MARK = {"town": "o", "airport": "s", "island": "^", "valley": "D"}
FACE = {"town": "white", "airport": "0.35", "island": "0.75",
        "valley": "black"}
lim = (0.15, 500.0)    # unchanged from the superseded figure
FLOOR = lim[0] * 1.2   # sub-floor values are drawn here, just inside the axis

fig, ax = plt.subplots(figsize=(3.4, 3.4))
ax.plot(lim, lim, ls="--", lw=0.7, color="0.4")
for t in TERR:
    pts = [b for b in both if b["terrain"] == t]
    # a dense error of 0.0 m exists and cannot be drawn on a log axis; it and
    # any other sub-floor value are drawn just inside the floor, and counted
    # in the provenance record below.
    ax.scatter([max(b["sketch_err_m"], FLOOR) for b in pts],
               [max(b["dense_err_m"], FLOOR) for b in pts],
               marker=MARK[t], s=16, facecolor=FACE[t], edgecolor="black",
               linewidth=0.6, label="%s (%d)" % (t, len(pts)), zorder=3)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*lim)
ax.set_ylim(*lim)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("sketch matcher error (m)")
ax.set_ylabel("dense error (m)")
ax.legend(loc="lower right", frameon=True, framealpha=1.0, edgecolor="0.7",
          fontsize=6.3, handlelength=1.0, borderpad=0.3, labelspacing=0.25)
ax.spines[["top", "right"]].set_visible(False)

per_terr = {}
for t in TERR:
    pts = [b for b in both if b["terrain"] == t]
    se = [b["sketch_err_m"] for b in pts]
    de = [b["dense_err_m"] for b in pts]
    per_terr[t] = {
        "n": len(pts),
        "sketch_median_m": round(st.median(se), 2),
        "dense_median_m": round(st.median(de), 2),
        "sketch_closer": sum(1 for b in pts
                             if b["sketch_err_m"] < b["dense_err_m"]),
        "dense_closer": sum(1 for b in pts
                            if b["dense_err_m"] < b["sketch_err_m"]),
        "tied": sum(1 for b in pts
                    if b["sketch_err_m"] == b["dense_err_m"]),
    }

NUM["panel3_reverse_scatter"] = {
    "path": save(fig, "fig_reverse_scatter_uniform_pp.pdf"),
    "n_plotted": len(both),
    "sketch_arm": "canny",
    "sketch_median_m": round(st.median([b["sketch_err_m"] for b in both]), 2),
    "dense_median_m": round(st.median([b["dense_err_m"] for b in both]), 2),
    "sketch_closer": sum(1 for b in both
                         if b["sketch_err_m"] < b["dense_err_m"]),
    "dense_closer": sum(1 for b in both
                        if b["dense_err_m"] < b["sketch_err_m"]),
    "tied": sum(1 for b in both
                if b["sketch_err_m"] == b["dense_err_m"]),
    "per_terrain": per_terr,
    "axis_limits_m": list(lim),
    "points_below_axis_floor": {
        "sketch": sum(1 for b in both if b["sketch_err_m"] < lim[0]),
        "dense": sum(1 for b in both if b["dense_err_m"] < lim[0]),
        "clamped_to_floor": True,
        "which": [b["pid"] for b in both
                  if b["dense_err_m"] < lim[0] or b["sketch_err_m"] < lim[0]],
    },
    "points_above_axis_ceiling": {
        "sketch": sum(1 for b in both if b["sketch_err_m"] > lim[1]),
        "dense": sum(1 for b in both if b["dense_err_m"] > lim[1]),
    },
    "dense_median_reporting_precision": {
        "prior_evidence": (
            "two earlier runs of the dense arm over this same 114-pair pool "
            "at one matched scoring convention gave town 163.9 against "
            "196.1 m and island 97.5 against 65.2 m, while airport gave 3.5 "
            "against 3.9 and valley 22.0 against 23.2. The RANSAC seed is "
            "randomized per process. Recorded as UNSTABLE in "
            "results/2026-09-14_rtk114_ppfix_table/make_ppfix_table_fig.py"),
        "town": {
            "value_m": per_terr["town"]["dense_median_m"],
            "report_as": float("%.2g" % per_terr["town"]["dense_median_m"]),
            "note": ("inside the documented 163.9 to 196.1 m seed band, so "
                     "the two-significant-figure convention carries over "
                     "unchanged"),
        },
        "island": {
            "value_m": per_terr["island"]["dense_median_m"],
            "report_as": per_terr["island"]["dense_median_m"],
            "note": ("NOT explained by seed instability. 6.5 m against 97.5 "
                     "and 65.2 m in the two earlier runs is an "
                     "order-of-magnitude move, far outside that band. The "
                     "population also differs: 11 island pairs both arms "
                     "answer here. Report 6.5 m plainly and raise the "
                     "discrepancy rather than absorbing it into the "
                     "seed-noise label"),
        },
    },
}

with open(os.path.join(OUT_DIR, "FIGURE_NUMBERS_2026-09-15.json"), "w") as f:
    json.dump(NUM, f, indent=1)
print("wrote", os.path.join(OUT_DIR, "FIGURE_NUMBERS_2026-09-15.json"))

print("\n--- panel 1, leave-one-scene-out ---")
for a in ARMS:
    lo = NUM["panel1_gate_tradeoff"]["loso"][a]
    print("  %-8s %d/%d admitted (%.1f%%), median %.2f m, %d over 30 m"
          % (a, lo["admitted"], lo["n_pool"], lo["coverage_pct"],
             lo["median_err_m"], lo["over_band"]))
print("--- panel 2, dense fit attempted ---")
for g in GROUPS:
    print("  %-8s %d/%d" % (g, counts[g]["fit_attempted"], counts[g]["n"]))
print("  no fit split: %d few correspondences, %d few inliers"
      % (counts["all"]["no_fit_few_correspondences"],
         counts["all"]["no_fit_few_inliers"]))
print("--- panel 3 ---")
print("  %d pairs both answer; sketch closer on %d, dense on %d, %d tied"
      % (NUM["panel3_reverse_scatter"]["n_plotted"],
         NUM["panel3_reverse_scatter"]["sketch_closer"],
         NUM["panel3_reverse_scatter"]["dense_closer"],
         NUM["panel3_reverse_scatter"]["tied"]))
