"""R9 in-window recount of the RUN 032 rerun matrix (RoMa v2 vs sketch).

Faculty comment R9 (docs/AMA_REVIEW_2026-09-16.md): compare both methods on
the same allowed area; a dense answer outside the sketch matcher's reachable
position box is counted as unanswered.

CPU only. No matcher is re-run. No GPU. Read-only except this directory.

INPUTS, read-only
  ../2026-09-15_rerun_matrix/matrix_roma_uavscenes.jsonl            114 rows
  ../2026-09-15_rerun_matrix/matrix_roma_visloc_combined125.jsonl   125 rows
  ../2026-09-15_rerun_matrix/matrix_canny_uavscenes_shipped_cc1_gate.jsonl
  ../2026-09-15_rerun_matrix/matrix_canny_visloc_combined125_oracle_cc1_gate.jsonl
  ../2026-09-09_rtk_uniform_pool_pilot/prepack/<pid>.npz, prepack_ortho/<pid>.npz
      UAVScenes: `off`, `eff`, `x0y0_big`, `anchor_EN`.
  ../2026-09-08_sketch_vs_roma_bench/visloc_bench_data/{prepack,pairs}/<pid>.npz
      UAV-VisLoc: only 5 of the 125 pids exist locally. Used as a direct check.

BOX DERIVATION (same as ../2026-09-15_equal_position_range/equal_position_range.py)
  rotation_probe.prep_pid:  true_c = (off[0] - x0) / 8,  true_r = (off[1] - y0) / 8
  cell pitch = 8 * eff;  (oh, ow) = sketch row grid_shape
  E in [tE - true_c*cell, tE + (ow-1-true_c)*cell]
  N in [tN - (oh-1-true_r)*cell, tN + true_r*cell]
  with (tE, tN) = the dense arm's own truth, roma.true_EN (== prepack anchor_EN).
  The crop x0, y0 come from the roma row itself; both runners draw the crop
  with random.Random(B._stable_pid_seed(pid)) and jitter_px = SEARCH_MARGIN_M
  / eff, so the crop is the same in both arms (verified by check 2 below).

UAV-VisLoc `off` WITHOUT A PREPACK
  visloc_build_site_hpc.py lines 558-559, 585, 588:
      offx = int(round(anchor_x)); offy = int(round(anchor_y))
      x0y0_big = (0, 0);  anchor_EN = (anchor_x * G_pair, -anchor_y * G_pair)
  so off = (round(tE / eff), round(-tN / eff)) exactly, from the roma row's
  own true_EN and eff. Verified against the 5 local prepacks (check 4) and
  against the sketch row's center_hit_err_m on every row that carries
  true_EN (check 2). Rows with no true_EN (fit_failed) get no box, and need
  none: there is no answer to classify.

OUTPUT  R9_RECOUNT.json and R9_RECOUNT.md in this directory.
Usage:  python r9_recount.py
"""
import io
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, ".."))
MAT = os.path.join(RES, "2026-09-15_rerun_matrix")
POOL = os.path.join(RES, "2026-09-09_rtk_uniform_pool_pilot")
VBD = os.path.join(RES, "2026-09-08_sketch_vs_roma_bench", "visloc_bench_data")
F = 8               # TRAIN_COARSE_F
TIE_M = 0.1

POPS = {
    "uavscenes": dict(
        roma="matrix_roma_uavscenes.jsonl",
        sketch="matrix_canny_uavscenes_shipped_cc1_gate.jsonl",
        sketch_err="shipped_err_m", n=114,
        sketch_label="sketch shipped heading (citable)"),
    "visloc": dict(
        roma="matrix_roma_visloc_combined125.jsonl",
        sketch="matrix_canny_visloc_combined125_oracle_cc1_gate.jsonl",
        sketch_err="oracle_diag_err_m", n=125,
        sketch_label="sketch ORACLE angle: GT-SELECTED CEILING, "
                     "gt_in_estimator=true, not a system result"),
}


def med(v):
    v = [x for x in v if x is not None]
    return round(float(np.median(v)), 1) if v else None


def mn(v):
    v = [x for x in v if x is not None]
    return round(float(min(v)), 1) if v else None


def mx(v):
    v = [x for x in v if x is not None]
    return round(float(max(v)), 1) if v else None


def load_jsonl(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def uav_prepack(pid):
    for cand in (pid, pid.rsplit("_", 1)[0]):
        for sub in ("prepack", "prepack_ortho"):
            p = os.path.join(POOL, sub, cand + ".npz")
            if os.path.exists(p):
                return p, sub
    raise SystemExit("FATAL: no UAVScenes prepack for %s" % pid)


def terrain_of(pid, pop):
    return pid.split("_")[0]


def geometry(pop, pid, rm, checks):
    """Returns (off, eff, tEN, source) or None when no box is derivable."""
    if pop == "uavscenes":
        p, sub = uav_prepack(pid)
        z = np.load(p, allow_pickle=True)
        off = np.asarray(z["off"], dtype=np.float64)
        eff = float(z["eff"])
        anchor = np.asarray(z["anchor_EN"], dtype=np.float64)
        x0y0 = np.asarray(z["x0y0_big"], dtype=np.float64)
        if rm.get("true_EN") is not None:
            d = float(np.abs(np.asarray(rm["true_EN"]) - anchor).max())
            checks["trueEN_vs_anchorEN_max_abs_m"] = max(
                checks["trueEN_vs_anchorEN_max_abs_m"], d)
            d2 = float(np.abs(np.asarray(rm["x0y0_big"]) - x0y0).max())
            checks["x0y0_vs_prepack_max_abs_px"] = max(
                checks["x0y0_vs_prepack_max_abs_px"], d2)
        return off, eff, anchor, "prepack_" + sub
    # ---- UAV-VisLoc
    pp = os.path.join(VBD, "prepack", pid + ".npz")
    pr = os.path.join(VBD, "pairs", pid + ".npz")
    if rm.get("true_EN") is None:
        if os.path.exists(pp) and os.path.exists(pr):
            z = np.load(pp, allow_pickle=True)
            zp = np.load(pr, allow_pickle=True)
            return (np.asarray(z["off"], np.float64), float(z["eff"]),
                    np.asarray(zp["anchor_EN"], np.float64), "local_prepack")
        return None
    tE, tN = rm["true_EN"]
    eff = float(rm.get("eff_row"))
    off = np.array([int(round(tE / eff)), int(round(-tN / eff))], np.float64)
    src = "off_from_row_trueEN"
    if os.path.exists(pp) and os.path.exists(pr):
        z = np.load(pp, allow_pickle=True)
        zp = np.load(pr, allow_pickle=True)
        d_off = float(np.abs(np.asarray(z["off"], np.float64) - off).max())
        d_anc = float(np.abs(np.asarray(zp["anchor_EN"], np.float64)
                             - np.array([tE, tN])).max())
        checks["visloc_local_prepack_n"] += 1
        checks["visloc_off_reconstructed_vs_prepack_max_abs_px"] = max(
            checks["visloc_off_reconstructed_vs_prepack_max_abs_px"], d_off)
        checks["visloc_trueEN_vs_pairs_anchorEN_max_abs_m"] = max(
            checks["visloc_trueEN_vs_pairs_anchorEN_max_abs_m"], d_anc)
        src = "off_from_row_trueEN+local_prepack_verified"
    return off, eff, np.array([tE, tN]), src


def analyse(pop, cfg):
    roma = load_jsonl(os.path.join(MAT, cfg["roma"]))
    sk = {r["pid"]: r for r in load_jsonl(os.path.join(MAT, cfg["sketch"]))}
    if len(roma) != cfg["n"] or len(sk) != cfg["n"]:
        raise SystemExit("FATAL: %s expected %d rows, got roma=%d sketch=%d"
                         % (pop, cfg["n"], len(roma), len(sk)))
    checks = dict(
        n_rows=len(roma),
        n_rows_with_box=0,
        n_rows_without_box=0,
        check1_truth_inside_box_n=0,
        check1_truth_outside_box_n=0,
        check2_center_hit_max_abs_diff_m=0.0,
        check2_center_hit_n_compared=0,
        check3_roma_err_recomputed_max_abs_diff_m=0.0,
        trueEN_vs_anchorEN_max_abs_m=0.0,
        x0y0_vs_prepack_max_abs_px=0.0,
        visloc_local_prepack_n=0,
        visloc_off_reconstructed_vs_prepack_max_abs_px=0.0,
        visloc_trueEN_vs_pairs_anchorEN_max_abs_m=0.0,
        eff_values=set(), grid_shapes=set(), geometry_sources={},
        roma_answered_without_box_n=0,
        roma_error_rows_n=0,
    )
    pairs = []
    for r in roma:
        pid = r["pid"]
        if r.get("error"):
            checks["roma_error_rows_n"] += 1
        rm = r.get("roma") or {}
        rm["eff_row"] = r.get("eff")
        s = sk[pid]
        oh, ow = s["grid_shape"]
        checks["grid_shapes"].add("%dx%d" % (oh, ow))
        checks["eff_values"].add(float(r.get("eff")))
        crop = r["crop"]
        rec = dict(pid=pid, terrain=terrain_of(pid, pop), eff=r.get("eff"),
                   crop=crop, grid_shape=[oh, ow],
                   sketch_answered=bool(s.get("answered")),
                   sketch_answered_gated=bool(s.get("answered_gated")),
                   sketch_err_m=s.get(cfg["sketch_err"]),
                   sketch_shipped_err_m=s.get("shipped_err_m"),
                   center_hit_err_m=s.get("center_hit_err_m"),
                   roma_answered=bool(rm.get("answered")),
                   roma_err_m=rm.get("err_m"),
                   roma_n_inliers=rm.get("n_inliers"),
                   roma_exclude_reason=rm.get("exclude_reason"))
        g = geometry(pop, pid, rm, checks)
        if g is None:
            checks["n_rows_without_box"] += 1
            checks["geometry_sources"]["none"] = \
                checks["geometry_sources"].get("none", 0) + 1
            rec.update(box_source=None, roma_in_box=None)
            if rm.get("answered"):
                checks["roma_answered_without_box_n"] += 1
            pairs.append(rec)
            continue
        off, eff, tEN, src = g
        checks["n_rows_with_box"] += 1
        checks["geometry_sources"][src] = \
            checks["geometry_sources"].get(src, 0) + 1
        cell = F * eff
        true_c = (off[0] - crop["x0"]) / F
        true_r = (off[1] - crop["y0"]) / F
        inside_grid = (0 <= true_c <= ow - 1) and (0 <= true_r <= oh - 1)
        checks["check1_truth_inside_box_n" if inside_grid
               else "check1_truth_outside_box_n"] += 1
        ch = float(np.hypot((oh - 1) / 2.0 - true_r,
                            (ow - 1) / 2.0 - true_c)) * cell
        if s.get("center_hit_err_m") is not None:
            checks["check2_center_hit_n_compared"] += 1
            checks["check2_center_hit_max_abs_diff_m"] = max(
                checks["check2_center_hit_max_abs_diff_m"],
                abs(ch - s["center_hit_err_m"]))
        E_min = tEN[0] - true_c * cell
        E_max = tEN[0] + (ow - 1 - true_c) * cell
        N_min = tEN[1] - (oh - 1 - true_r) * cell
        N_max = tEN[1] + true_r * cell
        d_edge = float(min(true_c, ow - 1 - true_c,
                           true_r, oh - 1 - true_r)) * cell
        rec.update(box_source=src, off=[int(off[0]), int(off[1])],
                   cell_m=cell, true_c=round(true_c, 4), true_r=round(true_r, 4),
                   true_EN=[float(tEN[0]), float(tEN[1])],
                   box_E=[round(E_min, 2), round(E_max, 2)],
                   box_N=[round(N_min, 2), round(N_max, 2)],
                   box_span_m=[round((ow - 1) * cell, 2), round((oh - 1) * cell, 2)],
                   center_hit_recomputed_m=round(ch, 3),
                   truth_inside_box=bool(inside_grid),
                   d_edge_m=round(d_edge, 2))
        if rm.get("answered"):
            e_rm = float(np.hypot(rm["E"] - tEN[0], rm["N"] - tEN[1]))
            checks["check3_roma_err_recomputed_max_abs_diff_m"] = max(
                checks["check3_roma_err_recomputed_max_abs_diff_m"],
                abs(e_rm - rm["err_m"]))
            inside = (E_min <= rm["E"] <= E_max) and (N_min <= rm["N"] <= N_max)
            dx = max(E_min - rm["E"], rm["E"] - E_max, 0.0)
            dy = max(N_min - rm["N"], rm["N"] - N_max, 0.0)
            rec.update(roma_E=rm["E"], roma_N=rm["N"], roma_in_box=bool(inside),
                       roma_dist_outside_box_m=round(float(np.hypot(dx, dy)), 2))
        else:
            rec.update(roma_in_box=None, roma_dist_outside_box_m=None)
        pairs.append(rec)

    checks["eff_values"] = sorted(checks["eff_values"])
    checks["grid_shapes"] = sorted(checks["grid_shapes"])
    for k in list(checks):
        if isinstance(checks[k], float):
            checks[k] = round(checks[k], 4)

    def cell_stats(sub):
        ans = [p for p in sub if p["roma_answered"]]
        ins = [p for p in ans if p["roma_in_box"] is True]
        outs = [p for p in ans if p["roma_in_box"] is False]
        nobox = [p for p in ans if p["roma_in_box"] is None]
        nofit = [p for p in sub if not p["roma_answered"]]
        reasons = {}
        for p in nofit:
            k = str(p["roma_exclude_reason"])
            reasons[k] = reasons.get(k, 0) + 1
        return dict(
            n_pairs=len(sub),
            roma_answered=len(ans),
            roma_no_fit=len(nofit),
            roma_no_fit_reasons=dict(sorted(reasons.items())),
            roma_inside_box=len(ins),
            roma_outside_box=len(outs),
            roma_answered_no_box_derivable=len(nobox),
            roma_outside_pct_of_answered=(round(100.0 * len(outs) / len(ans), 1)
                                          if ans else None),
            roma_answer_rate_pct=round(100.0 * len(ans) / len(sub), 1),
            roma_inside_rate_pct=round(100.0 * len(ins) / len(sub), 1),
            roma_median_answered_m=med([p["roma_err_m"] for p in ans]),
            roma_median_inside_m=med([p["roma_err_m"] for p in ins]),
            roma_inside_min_m=mn([p["roma_err_m"] for p in ins]),
            roma_inside_max_m=mx([p["roma_err_m"] for p in ins]),
            roma_outside_median_m=med([p["roma_err_m"] for p in outs]),
            roma_outside_min_m=mn([p["roma_err_m"] for p in outs]),
            roma_outside_max_m=mx([p["roma_err_m"] for p in outs]),
            roma_outside_dist_beyond_box_median_m=med(
                [p["roma_dist_outside_box_m"] for p in outs]),
            roma_outside_dist_beyond_box_min_m=mn(
                [p["roma_dist_outside_box_m"] for p in outs]),
            roma_outside_dist_beyond_box_max_m=mx(
                [p["roma_dist_outside_box_m"] for p in outs]),
            roma_outside_err_below_inside_max_n=(
                sum(1 for p in outs
                    if p["roma_err_m"] < max(q["roma_err_m"] for q in ins))
                if ins else None),
            roma_outside_err_lt10m_n=sum(1 for p in outs if p["roma_err_m"] < 10),
            roma_outside_err_lt20m_n=sum(1 for p in outs if p["roma_err_m"] < 20),
            d_edge_min_m=mn([p.get("d_edge_m") for p in sub]),
            d_edge_median_m=med([p.get("d_edge_m") for p in sub]),
            d_edge_max_m=mx([p.get("d_edge_m") for p in sub]),
            box_span_min_m=mn([p["box_span_m"][0] for p in sub if p.get("box_span_m")]),
            box_span_max_m=mx([p["box_span_m"][0] for p in sub if p.get("box_span_m")]),
            sketch_answered=sum(1 for p in sub if p["sketch_answered"]),
            sketch_answered_gated=sum(1 for p in sub if p["sketch_answered_gated"]),
            sketch_median_all_m=med([p["sketch_err_m"] for p in sub]),
            sketch_median_answered_m=med([p["sketch_err_m"] for p in sub
                                          if p["sketch_answered"]]),
            sketch_shipped_median_all_m=med([p["sketch_shipped_err_m"] for p in sub]),
            center_hit_median_m=med([p["center_hit_err_m"] for p in sub]),
            center_hit_max_m=mx([p["center_hit_err_m"] for p in sub]),
        )

    def h2h(sub):
        both = [p for p in sub if p["sketch_answered"] and p["roma_answered"]]
        both_in = [p for p in both if p["roma_in_box"] is True]
        moved = [p for p in both if p["roma_in_box"] is False]

        def counts(s):
            rc = sum(1 for p in s if p["roma_err_m"] < p["sketch_err_m"] - TIE_M)
            sc = sum(1 for p in s if p["sketch_err_m"] < p["roma_err_m"] - TIE_M)
            return rc, sc, len(s) - rc - sc
        rc0, sc0, t0 = counts(both)
        rc1, sc1, t1 = counts(both_in)
        rcm, scm, tm = counts(moved)
        return dict(
            tie_rule_m=TIE_M,
            unrestricted_n_both_answered=len(both),
            unrestricted_roma_closer=rc0,
            unrestricted_sketch_closer=sc0,
            unrestricted_tie=t0,
            unrestricted_roma_median_m=med([p["roma_err_m"] for p in both]),
            unrestricted_sketch_median_m=med([p["sketch_err_m"] for p in both]),
            inbox_n_both_answered=len(both_in),
            inbox_roma_closer=rc1,
            inbox_sketch_closer=sc1,
            inbox_tie=t1,
            inbox_roma_median_m=med([p["roma_err_m"] for p in both_in]),
            inbox_sketch_median_m=med([p["sketch_err_m"] for p in both_in]),
            moved_out_n=len(moved),
            moved_roma_was_closer=rcm,
            moved_sketch_was_closer=scm,
            moved_tie=tm,
            moved_sketch_err_min_m=mn([p["sketch_err_m"] for p in moved]),
            moved_sketch_err_max_m=mx([p["sketch_err_m"] for p in moved]),
            sketch_only_after_recount=(
                sum(1 for p in sub if p["sketch_answered"]) - len(both_in)),
        )

    terrains = sorted(set(p["terrain"] for p in pairs))
    per_terrain = {t: cell_stats([p for p in pairs if p["terrain"] == t])
                   for t in terrains}
    per_terrain["all"] = cell_stats(pairs)
    per_h2h = {t: h2h([p for p in pairs if p["terrain"] == t]) for t in terrains}
    per_h2h["all"] = h2h(pairs)
    return dict(sketch_arm=cfg["sketch_label"], sketch_err_field=cfg["sketch_err"],
                terrains=terrains, checks=checks, per_terrain=per_terrain,
                head_to_head=per_h2h, pairs=pairs)


def fmt(v):
    return "n/a" if v is None else str(v)


def write_md(doc, path):
    L = []
    A = L.append
    A("# R9 in-window recount of the RUN 032 matrix (RoMa v2 vs sketch)")
    A("")
    A("Answers faculty comment R9 (`docs/AMA_REVIEW_2026-09-16.md`): a RoMa v2 "
      "answer that falls outside the sketch matcher's reachable position box is "
      "counted as unanswered, and the head-to-head and per-terrain tables are "
      "re-derived on that basis. No matcher was re-run. CPU only, no GPU. "
      "Writer: `r9_recount.py`; per-pair records: `R9_RECOUNT.json`.")
    A("")
    A("Facts and numbers only. No verdict is drawn here.")
    A("")
    A("## The box")
    A("")
    A("Same derivation as `../2026-09-15_equal_position_range/` (which did it on the "
      "older 120 px-jitter table). `rotation_probe.prep_pid` places the truth at "
      "`true_c = (off[0] - x0)/8`, `true_r = (off[1] - y0)/8` on the sketch's cost "
      "grid of shape `grid_shape`; the sketch returns one cell, so its reachable "
      "set is a lattice of `ow x oh` positions at a pitch of `8 * eff` m. The "
      "sketch estimate is a pure translation, so the displacement set is the same "
      "for the principal point the dense arm scores; in RoMa's coordinates the "
      "bounding box is")
    A("")
    A("```")
    A("E in [tE - true_c*cell, tE + (ow-1-true_c)*cell]")
    A("N in [tN - (oh-1-true_r)*cell, tN + true_r*cell]     cell = 8*eff, (tE,tN) = roma.true_EN")
    A("```")
    A("")
    A("The crop `x0, y0` is taken from each RoMa row. Both runners draw the crop "
      "with `random.Random(B._stable_pid_seed(pid))` and `jitter_px = "
      "SEARCH_MARGIN_M / eff` (uniform over the full half-width), so the crop is "
      "the same in both arms; check 2 below tests that rather than assuming it. "
      "The in/out test is on the axis-aligned box (`E_min <= E <= E_max AND "
      "N_min <= N <= N_max`), not a radius.")
    A("")
    A("**UAVScenes** geometry (`off`, `eff`, `x0y0_big`, `anchor_EN`) comes from "
      "`../2026-09-09_rtk_uniform_pool_pilot/prepack{,_ortho}/<pid>.npz`.")
    A("")
    A("**UAV-VisLoc**: only 5 of the 125 pids have a local prepack "
      "(`../2026-09-08_sketch_vs_roma_bench/visloc_bench_data/`). The VisLoc "
      "writer (`../2026-08-26_uav_visloc_adapt/visloc_build_site_hpc.py` lines "
      "558-559, 585, 588) defines `off = (int(round(anchor_x)), "
      "int(round(anchor_y)))`, `x0y0_big = (0, 0)` and `anchor_EN = (anchor_x*G, "
      "-anchor_y*G)`, so `off = (round(tE/eff), round(-tN/eff))` is recovered "
      "exactly from each RoMa row's own `true_EN` and `eff`. That reconstruction "
      "is checked against the 5 local prepacks (check 4) and against the sketch "
      "row's `center_hit_err_m` on every row that carries `true_EN` (check 2). "
      "Rows with no `true_EN` (RoMa `fit_failed`) have no box and need none: "
      "there is no answer to classify.")
    A("")
    for pop in ("uavscenes", "visloc"):
        d = doc[pop]
        c = d["checks"]
        A("## %s" % pop)
        A("")
        A("Sketch arm: %s (`%s`)." % (d["sketch_arm"], d["sketch_err_field"]))
        A("")
        A("### Sanity checks")
        A("")
        A("| check | result |")
        A("|---|---|")
        A("| rows / rows with a box / without | %d / %d / %d |"
          % (c["n_rows"], c["n_rows_with_box"], c["n_rows_without_box"]))
        A("| RoMa answered rows with no derivable box | %d |"
          % c["roma_answered_without_box_n"])
        A("| (1) truth inside the box | %d inside, %d outside |"
          % (c["check1_truth_inside_box_n"], c["check1_truth_outside_box_n"]))
        A("| (2) recomputed center-hit vs sketch row `center_hit_err_m` | "
          "max abs diff %.4f m over %d rows (stored to 0.1 m) |"
          % (c["check2_center_hit_max_abs_diff_m"],
             c["check2_center_hit_n_compared"]))
        A("| (3) RoMa `err_m` recomputed from stored `E,N,true_EN` | "
          "max abs diff %.4f m |" % c["check3_roma_err_recomputed_max_abs_diff_m"])
        if pop == "uavscenes":
            A("| `roma.true_EN` vs prepack `anchor_EN` | max abs diff %.4f m |"
              % c["trueEN_vs_anchorEN_max_abs_m"])
            A("| `roma.x0y0_big` vs prepack `x0y0_big` | max abs diff %.4f px |"
              % c["x0y0_vs_prepack_max_abs_px"])
        else:
            A("| (4) reconstructed `off` vs local prepack `off` | max abs diff "
              "%.4f px over %d local prepacks |"
              % (c["visloc_off_reconstructed_vs_prepack_max_abs_px"],
                 c["visloc_local_prepack_n"]))
            A("| (4) `roma.true_EN` vs local `pairs` `anchor_EN` | max abs diff "
              "%.4f m over %d local prepacks |"
              % (c["visloc_trueEN_vs_pairs_anchorEN_max_abs_m"],
                 c["visloc_local_prepack_n"]))
        A("| eff values | %s |" % c["eff_values"])
        A("| grid shapes | %s |" % c["grid_shapes"])
        A("| geometry sources | %s |" % json.dumps(c["geometry_sources"]))
        A("")
        A("### Counts and medians")
        A("")
        A("| %s | n | RoMa no-fit | RoMa answered | inside box | outside box | "
          "RoMa median, answered (m) | RoMa median, inside only (m) | sketch "
          "median, all pairs (m) |" % ("scene" if pop == "uavscenes" else "site"))
        A("|---|---|---|---|---|---|---|---|---|")
        for t in d["terrains"] + ["all"]:
            s = d["per_terrain"][t]
            A("| %s | %d | %d | %d | %d | %d | %s | %s | %s |"
              % (t, s["n_pairs"], s["roma_no_fit"], s["roma_answered"],
                 s["roma_inside_box"], s["roma_outside_box"],
                 fmt(s["roma_median_answered_m"]), fmt(s["roma_median_inside_m"]),
                 fmt(s["sketch_median_all_m"])))
        A("")
        A("Coverage accompanies every median: read each inside-only median with "
          "its `inside box` count.")
        A("")
        A("### RoMa answers outside the box")
        A("")
        A("| %s | outside | error median / min / max (m) | distance beyond the "
          "box edge median / min / max (m) | smallest possible error outside "
          "the box, `d_edge` min / median (m) |"
          % ("scene" if pop == "uavscenes" else "site"))
        A("|---|---|---|---|---|")
        for t in d["terrains"] + ["all"]:
            s = d["per_terrain"][t]
            A("| %s | %d | %s / %s / %s | %s / %s / %s | %s / %s |"
              % (t, s["roma_outside_box"],
                 fmt(s["roma_outside_median_m"]), fmt(s["roma_outside_min_m"]),
                 fmt(s["roma_outside_max_m"]),
                 fmt(s["roma_outside_dist_beyond_box_median_m"]),
                 fmt(s["roma_outside_dist_beyond_box_min_m"]),
                 fmt(s["roma_outside_dist_beyond_box_max_m"]),
                 fmt(s["d_edge_min_m"]), fmt(s["d_edge_median_m"])))
        A("")
        s = d["per_terrain"]["all"]
        A("Inside-box RoMa errors run %s to %s m over %d answers."
          % (fmt(s["roma_inside_min_m"]), fmt(s["roma_inside_max_m"]),
             s["roma_inside_box"]))
        A("")
        A("Unlike the older 120 px-jitter table, where the truth was never closer "
          "than 36 m to a box edge, the matrix run's uniform jitter over the full "
          "half-width lets the truth sit anywhere in the box: `d_edge` (the "
          "smallest error any outside position can have) is min %s m, median %s "
          "m, max %s m here, and the box centre sits `center_hit_err_m` median "
          "%s m, max %s m from truth. So an outside answer is not necessarily a "
          "large error: %s of %d outside answers have an error below the "
          "inside-set maximum, %d are below 10 m and %d below 20 m. Box span per "
          "axis: %s to %s m (grid shapes %s; `eff` per pair %s to %s m/px)."
          % (fmt(s["d_edge_min_m"]), fmt(s["d_edge_median_m"]),
             fmt(s["d_edge_max_m"]), fmt(s["center_hit_median_m"]),
             fmt(s["center_hit_max_m"]),
             fmt(s["roma_outside_err_below_inside_max_n"]), s["roma_outside_box"],
             s["roma_outside_err_lt10m_n"], s["roma_outside_err_lt20m_n"],
             fmt(s["box_span_min_m"]), fmt(s["box_span_max_m"]),
             ", ".join(c["grid_shapes"]), min(c["eff_values"]),
             max(c["eff_values"])))
        A("")
        A("### Head-to-head, pairs where both arms answer (tie = within %.1f m)"
          % TIE_M)
        A("")
        if pop == "visloc":
            A("The sketch arm here is the oracle-angle ceiling "
              "(`oracle_diag_err_m`, gt_in_estimator=true, citable=false). "
              "The counts in the previous table (answered / inside / outside) "
              "do not touch the sketch arm and stand on their own.")
            A("")
        A("| %s | unrestricted: n, RoMa closer / sketch closer / tie | "
          "RoMa-inside only: n, RoMa closer / sketch closer / tie | moved out "
          "(RoMa was closer / sketch was closer) | RoMa median, inside set (m) "
          "| sketch median, inside set (m) |"
          % ("scene" if pop == "uavscenes" else "site"))
        A("|---|---|---|---|---|---|")
        for t in d["terrains"] + ["all"]:
            h = d["head_to_head"][t]
            A("| %s | %d: %d / %d / %d | %d: %d / %d / %d | %d (%d / %d) | %s | %s |"
              % (t, h["unrestricted_n_both_answered"], h["unrestricted_roma_closer"],
                 h["unrestricted_sketch_closer"], h["unrestricted_tie"],
                 h["inbox_n_both_answered"], h["inbox_roma_closer"],
                 h["inbox_sketch_closer"], h["inbox_tie"],
                 h["moved_out_n"], h["moved_roma_was_closer"],
                 h["moved_sketch_was_closer"],
                 fmt(h["inbox_roma_median_m"]), fmt(h["inbox_sketch_median_m"])))
        A("")
        h = d["head_to_head"]["all"]
        A("On the %d moved pairs the sketch errors run %s to %s m; the sketch "
          "answered %d of %d pairs (%d after its peak-ratio gate, not used here)."
          % (h["moved_out_n"], fmt(h["moved_sketch_err_min_m"]),
             fmt(h["moved_sketch_err_max_m"]), s["sketch_answered"], s["n_pairs"],
             s["sketch_answered_gated"]))
        A("")
    A("## Citability")
    A("")
    A("No `run.json` exists for this reanalysis. Per the project record-keeping policy, "
      "nothing here is citable until that record exists. The VisLoc sketch arm "
      "is the oracle ceiling and is not citable in any case.")
    A("")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    doc = {
        "_what": "R9 in-window recount of the RUN 032 rerun matrix: RoMa v2 "
                 "answers outside the sketch matcher's reachable box counted "
                 "as unanswered",
        "_date": "2026-09-16",
        "_faculty_comment": "docs/AMA_REVIEW_2026-09-16.md R9",
        "_sources": {
            "roma": [POPS[p]["roma"] for p in POPS],
            "sketch": [POPS[p]["sketch"] for p in POPS],
            "matrix_dir": MAT,
            "uavscenes_geometry": POOL + "/prepack{,_ortho}/<pid>.npz",
            "visloc_geometry": "off = (round(tE/eff), round(-tN/eff)) from the roma "
                               "row's true_EN; visloc_build_site_hpc.py lines "
                               "558-559, 585, 588; 5 local prepacks under "
                               + VBD + " used as a direct check",
            "box_precedent": RES + "/2026-09-15_equal_position_range/equal_position_range.py",
        },
        "_tie_rule_m": TIE_M,
        "_no_matcher_was_rerun": True,
        "_gpu_used": False,
        "_citability": "no run.json; not citable until one exists",
    }
    for pop, cfg in POPS.items():
        doc[pop] = analyse(pop, cfg)
    with io.open(os.path.join(HERE, "R9_RECOUNT.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    write_md(doc, os.path.join(HERE, "R9_RECOUNT.md"))
    for pop in POPS:
        print("==", pop)
        print(json.dumps(doc[pop]["checks"], indent=1))
        for t in doc[pop]["terrains"] + ["all"]:
            s = doc[pop]["per_terrain"][t]
            h = doc[pop]["head_to_head"][t]
            print("%-11s n=%-3d nofit=%-3d ans=%-3d in=%-3d out=%-3d  "
                  "medAns=%-6s medIn=%-6s sk=%-6s | h2h in: %d R%d/S%d/T%d moved %d"
                  % (t, s["n_pairs"], s["roma_no_fit"], s["roma_answered"],
                     s["roma_inside_box"], s["roma_outside_box"],
                     s["roma_median_answered_m"], s["roma_median_inside_m"],
                     s["sketch_median_all_m"], h["inbox_n_both_answered"],
                     h["inbox_roma_closer"], h["inbox_sketch_closer"],
                     h["inbox_tie"], h["moved_out_n"]))
    print("wrote R9_RECOUNT.json and R9_RECOUNT.md")


if __name__ == "__main__":
    main()
