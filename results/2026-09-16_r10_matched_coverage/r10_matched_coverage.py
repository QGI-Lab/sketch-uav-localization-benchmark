"""Faculty comment R10 (docs/AMA_REVIEW_2026-09-16.md): how many wrong positions
does each confidence score admit when both methods admit the same share of
queries, with the acceptance threshold chosen on one set of pairs and applied
unchanged to different pairs?

Scores compared (higher = more confident, admit when score >= threshold):
  * sketch matcher cost ratio `peak_ratio`  (Canny; distilled and full drawers too)
  * RoMa v2 inlier count `roma.n_inliers`
A pair the matcher did not answer has no score and is NOT admitted at any
threshold (RoMa's unanswered rows carry n_inliers 0; they are still excluded).

Wrong = error above 30 m. Error fields: UAVScenes `shipped_err_m` (system
result); UAV-VisLoc `oracle_diag_err_m`, the labelled oracle-heading ceiling
(ground truth chose the heading, so NOT a system result; the peak_ratio in
those rows was recorded at that same oracle heading, gate_angle_deg ==
oracle_diag_angle_deg on all 125 rows). RoMa: `roma.err_m` on both.

Protocol, per score and per target coverage c in {10, 20, 30} percent:
  1. Leave-one-scene-out. For each scene S: on the pairs of the OTHER scenes,
     choose the threshold that admits the largest share of ALL those pairs
     that is <= c (closest from below; the threshold is the lowest admitted
     score, ties resolved by dropping below the tie). Apply it unchanged to
     the pairs of S. Pool the held-out decisions over all scenes.
  2. Cross-dataset. Choose the threshold on all 114 UAVScenes pairs the same
     way, apply it unchanged to all 125 UAV-VisLoc pairs.
  3. In-sample reference. Choose the threshold on a population and evaluate it
     on the same population, at c = 20 percent, so the gap to 1 and 2 is
     visible. These are NOT held-out numbers.
Reported: admitted count, share of the full population, wrong (>30 m) among
admitted, median error of admitted.

Reads only results/2026-09-15_rerun_matrix/*.jsonl (RUN 032). No matcher is
run. Writes R10_MATCHED_COVERAGE.json and R10_MATCHED_COVERAGE.md beside
this file.

    python results/2026-09-16_r10_matched_coverage/r10_matched_coverage.py
"""
import io
import json
import math
import os
import statistics as S

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "2026-09-15_rerun_matrix")
WRONG_M = 30.0
TARGETS = [0.10, 0.20, 0.30]
NEG_INF = float("-inf")

# score name -> (uavscenes file, uavscenes err field, visloc file, visloc err field)
SCORES = {
    "peak_ratio_canny": (
        "matrix_canny_uavscenes_shipped_cc1_gate.jsonl", "shipped_err_m",
        "matrix_canny_visloc_combined125_oracle_cc1_gate.jsonl", "oracle_diag_err_m"),
    "roma_n_inliers": (
        "matrix_roma_uavscenes.jsonl", "roma.err_m",
        "matrix_roma_visloc_combined125.jsonl", "roma.err_m"),
    "peak_ratio_distill": (
        "matrix_distill_uavscenes_shipped_cc1_gate.jsonl", "shipped_err_m",
        "matrix_distill_visloc_combined125_oracle_cc1_gate.jsonl", "oracle_diag_err_m"),
    "peak_ratio_learned": (
        "matrix_learned_uavscenes_shipped_cc1_gate.jsonl", "shipped_err_m",
        "matrix_learned_visloc_combined125_oracle_cc1_gate.jsonl", "oracle_diag_err_m"),
}
SCORE_LABEL = {
    "peak_ratio_canny": "sketch peak_ratio (Canny)",
    "roma_n_inliers": "RoMa v2 n_inliers",
    "peak_ratio_distill": "sketch peak_ratio (distilled drawer)",
    "peak_ratio_learned": "sketch peak_ratio (full drawer)",
}
SCENE_ORDER = {
    "uavscenes": ["amtown01", "amvalley01", "hkairport", "hkisland01"],
    "visloc": ["visloc02", "visloc03", "visloc09", "visloc10", "visloc11"],
}


def load(fname):
    p = os.path.join(SRC, fname)
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def scene_of(pid):
    return pid.split("_")[0]


def pairs(fname, err_field, score_key):
    """One record per pair: pid, scene, score (or -inf if unanswered), err
    (None if unanswered), plus the other error fields for the VisLoc note."""
    out = []
    for r in load(fname):
        rec = {"pid": r["pid"], "scene": scene_of(r["pid"])}
        if score_key == "roma_n_inliers":
            a = r.get("roma") or {}
            ans = bool(a.get("answered")) and a.get("err_m") is not None
            rec["score"] = float(a.get("n_inliers") or 0) if ans else NEG_INF
            rec["err"] = float(a["err_m"]) if ans else None
        else:
            ans = bool(r.get("answered")) and r.get("peak_ratio") is not None \
                and r.get(err_field) is not None
            rec["score"] = float(r["peak_ratio"]) if ans else NEG_INF
            rec["err"] = float(r[err_field]) if ans else None
            for k in ("shipped_err_m", "cost_err_m"):
                if k in r and r[k] is not None:
                    rec[k] = float(r[k])
        rec["answered"] = ans
        out.append(rec)
    return out


def choose_threshold(cal, target):
    """Largest admitted share <= target on the calibration pairs. Returns
    (threshold, admitted_count_on_cal). Threshold +inf admits nothing.
    A pair is admitted iff answered and score >= threshold."""
    n = len(cal)
    k_max = int(math.floor(target * n + 1e-9))
    scores = sorted((p["score"] for p in cal if p["answered"]), reverse=True)
    for k in range(min(k_max, len(scores)), 0, -1):
        t = scores[k - 1]
        admitted = sum(1 for s in scores if s >= t)
        if admitted <= k_max:
            return t, admitted
    return float("inf"), 0


def admit(ps, t):
    return [p for p in ps if p["answered"] and p["score"] >= t]


def summarise(adm, n_pop):
    errs = [p["err"] for p in adm]
    d = {
        "admitted": len(adm),
        "population": n_pop,
        "share_pct": round(100.0 * len(adm) / n_pop, 1),
        "wrong_gt30m": sum(1 for e in errs if e > WRONG_M),
        "median_err_m": round(S.median(errs), 1) if errs else None,
        "max_err_m": round(max(errs), 1) if errs else None,
    }
    d["wrong_pct_of_admitted"] = (round(100.0 * d["wrong_gt30m"] / len(adm), 1)
                                  if adm else None)
    return d


def loso(ps, pop, target):
    folds = []
    pooled = []
    for sc in SCENE_ORDER[pop]:
        held = [p for p in ps if p["scene"] == sc]
        cal = [p for p in ps if p["scene"] != sc]
        t, n_cal_adm = choose_threshold(cal, target)
        adm = admit(held, t)
        pooled.extend(adm)
        folds.append({
            "held_out_scene": sc,
            "calibration_pairs": len(cal),
            "calibration_scenes": [s for s in SCENE_ORDER[pop] if s != sc],
            "threshold": t if t != float("inf") else "inf",
            "calibration_admitted": n_cal_adm,
            "calibration_share_pct": round(100.0 * n_cal_adm / len(cal), 1),
            "held_out_pairs": len(held),
            "held_out": summarise(adm, len(held)),
            "held_out_admitted_pids": [p["pid"] for p in adm],
        })
    d = {"target_pct": int(round(100 * target)),
         "pooled_held_out": summarise(pooled, len(ps)),
         "folds": folds}
    if pop == "visloc" and any("shipped_err_m" in p for p in ps):
        d["pooled_same_admitted_set_other_headings_note"] = {
            k: secondary_errors(pooled, k) for k in ("shipped_err_m", "cost_err_m")}
    return d


def secondary_errors(adm, key):
    """Same admitted set, a different error column (VisLoc shipped / cost
    headings). The admission score was computed at the oracle heading, so this
    is a mismatched pairing and is reported only as a note."""
    e = [p[key] for p in adm if key in p]
    if not e:
        return None
    return {"n": len(e), "wrong_gt30m": sum(1 for v in e if v > WRONG_M),
            "median_err_m": round(S.median(e), 1)}


def main():
    out = {
        "question": "R10: wrong admissions per score at matched coverage, thresholds chosen on other pairs",
        "source_dir": os.path.basename(SRC),   # relative on purpose: no absolute paths in released output
        "wrong_threshold_m": WRONG_M,
        "targets_pct": [int(round(100 * t)) for t in TARGETS],
        "rule": "admit iff answered and score >= threshold; threshold = lowest admitted score on the calibration pairs giving the largest share <= target; unanswered pairs never admitted",
        "error_fields": {"uavscenes": "shipped_err_m (sketch) / roma.err_m",
                         "visloc": "oracle_diag_err_m (sketch; GT-selected heading, NOT a system result; peak_ratio recorded at that heading) / roma.err_m"},
        "scores": {},
    }
    data = {}
    for sk, (fu, eu, fv, ev) in SCORES.items():
        data[sk] = {"uavscenes": pairs(fu, eu, sk), "visloc": pairs(fv, ev, sk)}
        d = {"label": SCORE_LABEL[sk], "files": {"uavscenes": fu, "visloc": fv},
             "population": {}, "cross_dataset_uavscenes_to_visloc": {},
             "in_sample_20pct": {}}
        for pop in ("uavscenes", "visloc"):
            ps = data[sk][pop]
            d["population"][pop] = {
                "n_pairs": len(ps),
                "n_answered": sum(1 for p in ps if p["answered"]),
                "scenes": {sc: sum(1 for p in ps if p["scene"] == sc)
                           for sc in SCENE_ORDER[pop]},
                "loso": [loso(ps, pop, t) for t in TARGETS],
            }
            # in-sample reference at 20 %
            t, n_adm = choose_threshold(ps, 0.20)
            adm = admit(ps, t)
            d["in_sample_20pct"][pop] = {
                "threshold": t if t != float("inf") else "inf",
                "chosen_on": "all %d %s pairs (same pairs evaluated; NOT held out)" % (len(ps), pop),
                "result": summarise(adm, len(ps)),
            }
            if pop == "visloc" and sk != "roma_n_inliers":
                d["in_sample_20pct"][pop]["same_admitted_set_other_headings_note"] = {
                    k: secondary_errors(adm, k) for k in ("shipped_err_m", "cost_err_m")}
        # cross-dataset: choose on all UAVScenes, apply to all VisLoc
        for tgt in TARGETS:
            t, n_adm = choose_threshold(data[sk]["uavscenes"], tgt)
            adm = admit(data[sk]["visloc"], t)
            entry = {
                "target_pct": int(round(100 * tgt)),
                "threshold": t if t != float("inf") else "inf",
                "chosen_on": "all 114 UAVScenes pairs (admitted %d there, %.1f%%)" % (
                    n_adm, 100.0 * n_adm / 114),
                "evaluated_on": "all 125 UAV-VisLoc pairs, threshold unchanged",
                "result": summarise(adm, len(data[sk]["visloc"])),
                "admitted_pids": [p["pid"] for p in adm],
                "highest_score_in_visloc": max(p["score"] for p in data[sk]["visloc"] if p["answered"]),
            }
            if sk != "roma_n_inliers":
                entry["same_admitted_set_other_headings_note"] = {
                    k: secondary_errors(adm, k) for k in ("shipped_err_m", "cost_err_m")}
            d["cross_dataset_uavscenes_to_visloc"][str(int(round(100 * tgt)))] = entry
        out["scores"][sk] = d

    with io.open(os.path.join(HERE, "R10_MATCHED_COVERAGE.json"), "w",
                 encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    write_md(out)
    print(io.open(os.path.join(HERE, "R10_MATCHED_COVERAGE.md"), encoding="utf-8").read())


def fmt(r):
    if r["admitted"] == 0:
        return "0 | 0.0 | 0 | -- | --"
    return "%d | %.1f | %d | %s | %s" % (
        r["admitted"], r["share_pct"], r["wrong_gt30m"],
        "%.1f" % r["median_err_m"], "%.1f" % r["max_err_m"])


def fmt_t(t):
    if t == "inf":
        return "inf"
    return ("%d" % t) if float(t).is_integer() and t > 2 else ("%.4f" % t)


def write_md(out):
    L = []
    A = L.append
    A("# R10: wrong admissions at matched coverage, thresholds chosen on other pairs")
    A("")
    A("Source rows: RUN 032, `results/2026-09-15_rerun_matrix/` (no matcher re-run). "
      "Script: `r10_matched_coverage.py`; full numbers, per-fold thresholds and admitted pids: `R10_MATCHED_COVERAGE.json`.")
    A("")
    A("**Rule.** A pair is admitted when it was answered and its score is at or above the threshold. "
      "The threshold is the lowest admitted score on the calibration pairs that gives the largest admitted share "
      "at or below the target (closest from below; a tie that would overshoot is dropped). Unanswered pairs "
      "(sketch: no answer; RoMa: no fit, `n_inliers` 0) are never admitted. Wrong = error above %.0f m. "
      "Share is of the full population (114 UAVScenes pairs, 125 UAV-VisLoc pairs), not of the answered pairs." % WRONG_M)
    A("")
    A("**Error fields.** UAVScenes: sketch `shipped_err_m`, RoMa `roma.err_m`. UAV-VisLoc: sketch `oracle_diag_err_m`, "
      "the heading ground truth selected (a labelled ceiling, `gt_in_estimator: true`, not a system result; the "
      "`peak_ratio` in those rows was recorded at that same heading, `gate_angle_deg == oracle_diag_angle_deg` on all 125 rows); RoMa `roma.err_m`.")
    A("")
    A("**Scenes.** UAVScenes: amtown01 33, amvalley01 29, hkairport 36, hkisland01 16. UAV-VisLoc: visloc02, 03, 09, 10, 11 with 25 each.")
    A("")
    A("Columns: admitted | share of population % | wrong (>30 m) | median error of admitted, m | max error of admitted, m.")
    A("")
    # --- 1. LOSO tables
    A("## 1. Leave-one-scene-out (threshold chosen on the other scenes, applied unchanged to the held-out scene, pooled)")
    A("")
    for pop, popname in (("uavscenes", "UAVScenes, 114 pairs"), ("visloc", "UAV-VisLoc, 125 pairs")):
        A("### %s" % popname)
        A("")
        A("| score | target % | admitted | share % | wrong >30 m | median m | max m |")
        A("|---|---|---|---|---|---|---|")
        for sk in SCORES:
            d = out["scores"][sk]["population"][pop]
            for lo in d["loso"]:
                A("| %s | %d | %s |" % (SCORE_LABEL[sk], lo["target_pct"], fmt(lo["pooled_held_out"])))
        A("")
    # per-fold at 20 % for the two primary scores
    A("### Per-fold detail at the 20 % target, primary scores")
    A("")
    A("| population | score | held-out scene | threshold (chosen on the other scenes) | admitted on calibration | held-out admitted / pairs | held-out wrong >30 m | held-out median m |")
    A("|---|---|---|---|---|---|---|---|")
    for pop in ("uavscenes", "visloc"):
        for sk in ("peak_ratio_canny", "roma_n_inliers"):
            d = out["scores"][sk]["population"][pop]
            lo = [x for x in d["loso"] if x["target_pct"] == 20][0]
            for f in lo["folds"]:
                h = f["held_out"]
                A("| %s | %s | %s | %s | %d / %d (%.1f%%) | %d / %d | %d | %s |" % (
                    pop, SCORE_LABEL[sk], f["held_out_scene"], fmt_t(f["threshold"]),
                    f["calibration_admitted"], f["calibration_pairs"], f["calibration_share_pct"],
                    h["admitted"], f["held_out_pairs"], h["wrong_gt30m"],
                    "--" if h["median_err_m"] is None else "%.1f" % h["median_err_m"]))
    A("")
    # --- 2. cross-dataset
    A("## 2. Threshold chosen on all 114 UAVScenes pairs, applied unchanged to all 125 UAV-VisLoc pairs")
    A("")
    A("The threshold was chosen on UAVScenes only; no UAV-VisLoc pair took part in choosing it. Every UAV-VisLoc pair is an evaluation pair here.")
    A("")
    A("| score | target % | threshold | admitted on UAVScenes (chooser) | highest score among the 125 UAV-VisLoc pairs | UAV-VisLoc admitted | share % | wrong >30 m | median m | max m |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for sk in SCORES:
        for tp in ("10", "20", "30"):
            e = out["scores"][sk]["cross_dataset_uavscenes_to_visloc"][tp]
            A("| %s | %s | %s | %s | %s | %s |" % (SCORE_LABEL[sk], tp, fmt_t(e["threshold"]),
                                                   e["chosen_on"].split("(")[1].rstrip(")"),
                                                   fmt_t(e["highest_score_in_visloc"]),
                                                   fmt(e["result"])))
    A("")
    notes = []
    for sk in ("peak_ratio_canny", "peak_ratio_distill", "peak_ratio_learned"):
        lo = [x for x in out["scores"][sk]["population"]["visloc"]["loso"] if x["target_pct"] == 20][0]
        n = lo.get("pooled_same_admitted_set_other_headings_note") or {}
        parts = []
        for k in ("shipped_err_m", "cost_err_m"):
            v = n.get(k)
            if v:
                parts.append("%s: %d wrong of %d, median %.1f m" % (k, v["wrong_gt30m"], v["n"], v["median_err_m"]))
        if parts:
            notes.append("%s, leave-one-scene-out at 20 %%: %s" % (SCORE_LABEL[sk], "; ".join(parts)))
    if notes:
        A("Note on the UAV-VisLoc sketch rows of section 1: the same leave-one-scene-out admitted pairs scored with the "
          "shipped-heading and cost-selected-heading errors instead of the oracle-heading error. The admission score was "
          "computed at the oracle heading, so this pairs a score with an answer from a different heading; given only so "
          "the shipped-heading numbers are not hidden.")
        for n in notes:
            A("- " + n)
        A("")
    # --- 3. in-sample
    A("## 3. In-sample reference at the 20 % target (threshold chosen and evaluated on the same pairs; NOT held out)")
    A("")
    A("| population | score | threshold | admitted | share % | wrong >30 m | median m | max m |")
    A("|---|---|---|---|---|---|---|---|")
    for pop in ("uavscenes", "visloc"):
        for sk in SCORES:
            e = out["scores"][sk]["in_sample_20pct"][pop]
            A("| %s | %s | %s | %s |" % (pop, SCORE_LABEL[sk], fmt_t(e["threshold"]), fmt(e["result"])))
    A("")
    # --- 4. compact paper table
    A("## 4. Compact table (primary scores, 20 % target)")
    A("")
    A("| population | score | protocol | admitted / pairs | wrong >30 m | median m |")
    A("|---|---|---|---|---|---|")
    for pop, popname, n in (("uavscenes", "UAVScenes", 114), ("visloc", "UAV-VisLoc", 125)):
        for sk in ("peak_ratio_canny", "roma_n_inliers"):
            lo = [x for x in out["scores"][sk]["population"][pop]["loso"] if x["target_pct"] == 20][0]["pooled_held_out"]
            A("| %s | %s | leave-one-scene-out | %d / %d | %d | %s |" % (
                popname, SCORE_LABEL[sk], lo["admitted"], n, lo["wrong_gt30m"],
                "--" if lo["median_err_m"] is None else "%.1f" % lo["median_err_m"]))
            if pop == "visloc":
                cx = out["scores"][sk]["cross_dataset_uavscenes_to_visloc"]["20"]["result"]
                A("| %s | %s | threshold from UAVScenes | %d / %d | %d | %s |" % (
                    popname, SCORE_LABEL[sk], cx["admitted"], n, cx["wrong_gt30m"],
                    "--" if cx["median_err_m"] is None else "%.1f" % cx["median_err_m"]))
            ins = out["scores"][sk]["in_sample_20pct"][pop]["result"]
            A("| %s | %s | in-sample (reference) | %d / %d | %d | %s |" % (
                popname, SCORE_LABEL[sk], ins["admitted"], n, ins["wrong_gt30m"],
                "--" if ins["median_err_m"] is None else "%.1f" % ins["median_err_m"]))
    A("")
    A("UAV-VisLoc sketch errors are at the ground-truth-selected heading (ceiling, not a system result). "
      "Leave-one-scene-out: each scene's threshold was chosen on the other scenes of the same population and evaluated on that scene only. "
      "Threshold from UAVScenes: chosen on the 114 UAVScenes pairs, evaluated on the 125 UAV-VisLoc pairs. "
      "In-sample: chosen and evaluated on the same pairs.")
    A("")
    with io.open(os.path.join(HERE, "R10_MATCHED_COVERAGE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
