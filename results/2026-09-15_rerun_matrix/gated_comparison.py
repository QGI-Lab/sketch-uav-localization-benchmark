"""Does RoMa's inlier count work as a reliability signal? Faculty comment 8.

The draft says "RoMa exposes no comparable confidence". Comment 8 objects that
this is too broad: RoMa gives per-match confidence and inlier counts are
recorded, and if we never tested using them we should say so rather than claim
the signal does not exist.

This tests it. RoMa answers whenever a fit reaches 2 inliers and nothing gates
it, so its median includes fits with almost no support. Sweeping a minimum
inlier count turns "answer rate" against "accuracy of what is admitted", which
is exactly the trade-off the sketch matcher's peak-ratio gate is reported on.

Nothing here uses ground truth to CHOOSE the threshold; the sweep is reported
in full so the shape is visible and a threshold can be picked on held-out data
later. Errors are measured against truth, as everywhere else.

    python results/2026-09-15_rerun_matrix/gated_comparison.py
"""
import io
import json
import os
import statistics as S

HERE = os.path.dirname(os.path.abspath(__file__))
GATES = [2, 5, 10, 20, 50, 100]


def rows(name):
    """Accept an exact name or a glob. Cell filenames gained a _cc1/_cc0
    suffix once the connected-component filter became a variable, so a fixed
    name would silently miss the filtered runs. Prefers the FILTERED file when
    both exist, since that is the arms-comparable one (faculty comment 19)."""
    import glob
    # Look for a _cc1 sibling FIRST. Checking the exact name first was wrong:
    # the unfiltered files are still on disk, so the reference rows silently
    # read the pre-filter arms and reported 45/114 when the filtered cell had
    # answered all 114.
    stem = name[:-6] if name.endswith(".jsonl") else name
    cc1p = os.path.join(HERE, stem + "_cc1.jsonl")
    if os.path.exists(cc1p):
        return [json.loads(l) for l in io.open(cc1p, encoding="utf-8")
                if l.strip()]
    p = os.path.join(HERE, name)
    if os.path.exists(p):
        return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]
    hits = sorted(glob.glob(os.path.join(HERE, stem + "*.jsonl")))
    cc1 = [h for h in hits if h.endswith("_cc1.jsonl")]
    hits = cc1 or hits
    if not hits:
        return None
    if len(hits) > 1:
        print("  (note: %d files match %s, using %s)"
              % (len(hits), stem, os.path.basename(hits[0])))
    return [json.loads(l) for l in io.open(hits[0], encoding="utf-8")
            if l.strip()]


def roma_answers(rs):
    """(err_m, n_inliers) for every pair RoMa actually answered."""
    out = []
    for x in rs:
        a = x.get("roma") or {}
        if a.get("err_m") is not None:
            out.append((float(a["err_m"]), int(a.get("n_inliers") or 0)))
    return out


def sweep(name, label, n_total):
    rs = rows(name)
    if rs is None:
        print("  %s: FILE MISSING" % name)
        return
    ans = roma_answers(rs)
    load_err = sum(1 for x in rs if x.get("error"))
    print()
    print("%s  (%d rows, %d answered, %d load-errors)"
          % (label, len(rs), len(ans), load_err))
    if not ans:
        return
    print("  %-14s %10s %9s %9s %9s" % ("min inliers", "admitted", "median",
                                        "<=30 m", ">100 m"))
    for g in GATES:
        keep = [e for e, n in ans if n >= g]
        if not keep:
            print("  %-14d %10s %9s %9s %9s" % (g, "0", "-", "-", "-"))
            continue
        print("  %-14d %10s %9.1f %9d %9d"
              % (g, "%d/%d" % (len(keep), n_total), S.median(keep),
                 sum(1 for v in keep if v <= 30),
                 sum(1 for v in keep if v > 100)))


def sketch_reference(pop_files, n_total, label, key):
    print()
    print("%s, for reference (sketch arms answer every pair)" % label)
    print("  %-14s %10s %9s %9s" % ("arm", "answered", "median", "<=30 m"))
    for arm, fname in pop_files:
        rs = rows(fname)
        if rs is None:
            print("  %-14s %10s" % (arm, "MISSING"))
            continue
        ok = [x for x in rs if x.get("answered")]
        e = [x[key] for x in ok if x.get(key) is not None]
        if not e:
            continue
        print("  %-14s %10s %9.1f %9d"
              % (arm, "%d/%d" % (len(ok), n_total), S.median(e),
                 sum(1 for v in e if v <= 30)))


def main():
    print("=" * 66)
    print("RoMa v2 inlier count as a reliability signal (faculty comment 8)")
    print("=" * 66)

    sweep("matrix_roma_visloc_combined125.jsonl",
          "UAV-VisLoc, combined 125", 125)
    sketch_reference([("canny", "matrix_canny_visloc_combined125_oracle.jsonl"),
                      ("distill", "matrix_distill_visloc_combined125_oracle.jsonl"),
                      ("learned", "matrix_learned_visloc_combined125_oracle.jsonl")],
                     125, "UAV-VisLoc, shipped heading", "shipped_err_m")

    sweep("matrix_roma_uavscenes.jsonl", "UAVScenes, n=114", 114)
    sketch_reference([("canny", "matrix_canny_uavscenes_shipped.jsonl"),
                      ("distill", "matrix_distill_uavscenes_shipped.jsonl"),
                      ("learned", "matrix_learned_uavscenes_shipped.jsonl")],
                     114, "UAVScenes, shipped heading", "shipped_err_m")

    print()
    print("Read the sweep as a trade-off, not a tuned number: raising the "
          "minimum inlier count trades answers for accuracy. A threshold must "
          "be calibrated on held-out data before it is reported as a system "
          "setting.")


if __name__ == "__main__":
    main()
