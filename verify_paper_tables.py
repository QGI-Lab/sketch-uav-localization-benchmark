# -*- coding: utf-8 -*-
"""Recompute the headline tables straight from the per-pair rows in this bundle.

    python verify_paper_tables.py

Standard library plus nothing else. No dataset download is needed: every number
below comes from the .jsonl rows under results/2026-09-15_rerun_matrix/.
Runs in under a second.
"""
import collections
import io
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
MAT = os.path.join(HERE, "results", "2026-09-15_rerun_matrix")

TERRAIN = [("amtown01", "town"), ("hkairport", "airport"),
           ("hkisland01", "island"), ("amvalley01", "valley")]
ORDER = ["town", "airport", "island", "valley"]


def rows(name):
    p = os.path.join(MAT, name)
    with io.open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def terrain_of(pid):
    for pre, t in TERRAIN:
        if pid.startswith(pre):
            return t
    raise KeyError(pid)


def line(s):
    print(s)


def table_terrain():
    line("Answer rate and error by terrain class, 114 UAVScenes pairs")
    line("  (dense = RoMa v2; sketch = untrained Canny drawer at the supplied heading)")
    roma = {r["pid"]: r for r in rows("matrix_roma_uavscenes.jsonl")}
    sk = {r["pid"]: r for r in rows("matrix_canny_uavscenes_shipped_cc1_gate.jsonl")}
    by = collections.defaultdict(list)
    for pid in sk:
        by[terrain_of(pid)].append(pid)
    line("  %-8s %4s %7s %12s %12s" % ("terrain", "n", "no fit", "dense med m", "sketch med m"))
    for t in ORDER:
        ps = by[t]
        nofit = sum(1 for p in ps if not roma[p]["roma"]["answered"])
        rm = [roma[p]["roma"]["err_m"] for p in ps if roma[p]["roma"]["answered"]]
        sm = [sk[p]["shipped_err_m"] for p in ps if sk[p].get("answered")]
        line("  %-8s %4d %7d %12.1f %12.1f" % (t, len(ps), nofit, st.median(rm), st.median(sm)))
    allr = [roma[p]["roma"]["err_m"] for p in sk if roma[p]["roma"]["answered"]]
    alls = [sk[p]["shipped_err_m"] for p in sk if sk[p].get("answered")]
    nofit = sum(1 for p in sk if not roma[p]["roma"]["answered"])
    line("  %-8s %4d %7d %12.1f %12.1f" % ("all", len(sk), nofit, st.median(allr), st.median(alls)))


def table_complementarity():
    line("")
    line("The pairs both matchers answer, UAVScenes")
    roma = {r["pid"]: r for r in rows("matrix_roma_uavscenes.jsonl")}
    sk = {r["pid"]: r for r in rows("matrix_canny_uavscenes_shipped_cc1_gate.jsonl")}
    both = [p for p in sk if sk[p].get("answered") and roma[p]["roma"]["answered"]]
    closer_sk = sum(1 for p in both if sk[p]["shipped_err_m"] < roma[p]["roma"]["err_m"])
    closer_rm = sum(1 for p in both if roma[p]["roma"]["err_m"] < sk[p]["shipped_err_m"])
    line("  both answer: %d of %d" % (len(both), len(sk)))
    line("  sketch closer %d, dense closer %d, tied %d"
         % (closer_sk, closer_rm, len(both) - closer_sk - closer_rm))


def table_ablation():
    line("")
    line("Drawing step on 125 UAV-VisLoc pairs, same matcher and search")
    line("  NOTE these rows carry gt_in_estimator: true. The heading is picked using the")
    line("  reference position, so this is a best case that bounds any heading source.")
    line("  It is a diagnostic ceiling, not a system result.")
    line("  %-22s %9s %10s %10s" % ("drawing step", "answered", "median m", "under 30 m"))
    for lab, f in [("Canny, untrained", "matrix_canny_visloc_combined125_oracle_cc1_gate.jsonl"),
                   ("distillation only", "matrix_distill_visloc_combined125_oracle_cc1_gate.jsonl"),
                   ("distillation + margin", "matrix_learned_visloc_combined125_oracle_cc1_gate.jsonl")]:
        rr = rows(f)
        e = [r["oracle_diag_err_m"] for r in rr if r.get("answered")]
        line("  %-22s %9s %10.1f %10d"
             % (lab, "%d/%d" % (len(e), len(rr)), st.median(e), sum(1 for x in e if x < 30)))


def table_headings():
    line("")
    line("UAV-VisLoc median error under three heading rules, Canny drawer, 125 pairs")
    rr = rows("matrix_canny_visloc_combined125_oracle_cc1_gate.jsonl")
    sh = [r["shipped_err_m"] for r in rr if r.get("shipped_err_m") is not None]
    co = [r["cost_err_m"] for r in rr if r.get("cost_err_m") is not None]
    orc = [r["oracle_diag_err_m"] for r in rr if r.get("oracle_diag_err_m") is not None]
    ce = [r["center_hit_err_m"] for r in rr if r.get("center_hit_err_m") is not None]
    line("  supplied heading                 %5.1f m   (n=%d)" % (st.median(sh), len(sh)))
    line("  heading chosen by matching cost   %5.1f m   (n=%d)" % (st.median(co), len(co)))
    line("  heading chosen using the truth    %5.1f m   (n=%d)  ORACLE, not a system result"
         % (st.median(orc), len(orc)))
    line("  search-window centre, no matching %5.1f m   (n=%d)" % (st.median(ce), len(ce)))


def table_kpsilence():
    line("")
    line("Sparse keypoint matcher on the 125 UAV-VisLoc pairs")
    x = rows("matrix_xfeat_visloc_combined125.jsonl")
    ans = sum(1 for r in x if r["xfeat_lg"]["answered"])
    reasons = collections.Counter(r["xfeat_lg"].get("exclude_reason") for r in x)
    line("  answered %d of %d" % (ans, len(x)))
    for k, v in reasons.most_common():
        line("    %-24s %d" % (k, v))


def table_answered():
    line("")
    line("Answer counts over the full population, unanswered pairs included")
    for lab, f, n in [("Canny, UAV-VisLoc", "matrix_canny_visloc_combined125_oracle_cc1_gate.jsonl", 125),
                      ("distilled, UAV-VisLoc", "matrix_distill_visloc_combined125_oracle_cc1_gate.jsonl", 125),
                      ("learned, UAV-VisLoc", "matrix_learned_visloc_combined125_oracle_cc1_gate.jsonl", 125),
                      ("Canny, UAVScenes", "matrix_canny_uavscenes_shipped_cc1_gate.jsonl", 114),
                      ("distilled, UAVScenes", "matrix_distill_uavscenes_shipped_cc1_gate.jsonl", 114),
                      ("learned, UAVScenes", "matrix_learned_uavscenes_shipped_cc1_gate.jsonl", 114),
                      ("RoMa v2, UAV-VisLoc", "matrix_roma_visloc_combined125.jsonl", 125),
                      ("RoMa v2, UAVScenes", "matrix_roma_uavscenes.jsonl", 114)]:
        rr = rows(f)
        if "roma" in rr[0]:
            a = sum(1 for r in rr if r["roma"]["answered"])
        else:
            a = sum(1 for r in rr if r.get("answered"))
        line("  %-24s %3d / %3d  (rows on file: %d)" % (lab, a, n, len(rr)))


def splits():
    line("")
    line("The fixed split of the 114 UAVScenes pairs")
    with io.open(os.path.join(HERE, "data", "uavscenes114_split.json"), encoding="utf-8") as f:
        s = json.load(f)
    tot = 0
    for k in ["train", "val", "test"]:
        line("  %-6s %3d" % (k, len(s[k])))
        tot += len(s[k])
    line("  total  %3d" % tot)
    heldout = len(s["val"]) + len(s["test"])
    line("  held out of training: %d" % heldout)


if __name__ == "__main__":
    table_terrain()
    table_complementarity()
    table_ablation()
    table_headings()
    table_kpsilence()
    table_answered()
    splits()
    line("")
    line("Every line above was recomputed from files in this bundle.")
