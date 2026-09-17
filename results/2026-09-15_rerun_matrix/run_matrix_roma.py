"""The RoMa v2 cells of the full rerun matrix. Lead request, 2026-09-15.

Same protocol as run_matrix_sketch.py: uniform jitter over the box half-width.
RoMa's answer is NOT confined; see below for why that is the honest choice.

  BENCH_POP   uavscenes | visloc

NO ROTATION ARM. RoMa v2 consumes no rotation input at all: it is dense and
rotation-robust by construction, and its driver contains no angle, Mr or
rotation handling anywhere. "Oracle rotation" is therefore undefined for this
matcher, and its cell runs once per population. Nothing here touches ground
truth, so every row is citable.

HOW THE ANSWER IS CONFINED, and why this changed on 2026-09-15. RoMa fits a
rigid transform and evaluates R @ pp + t, which can land anywhere. RUN 031
censored the OUTPUT: a fit whose mapped principal point left the lattice was
rejected. Job 17865984 showed that failing in two ways at once, on UAV-VisLoc:
the draw cap was hit on 61 of 75 pairs with a median of 2 admitted hypotheses
out of 2000, and errors reached 234 m on a ~120 m window, meaning the
admissible box was anchored with the UAVScenes convention and sat in the wrong
place.

THE PREDICATE IS GONE, and is not replaced. Scored by distance to truth,
freedom to answer anywhere is a LIABILITY rather than an advantage: an answer
outside the lattice simply earns a large error. So there was never a fairness
problem to fix in that direction, and the asymmetry that does exist runs the
other way and is STATED instead of engineered away:

    the sketch matcher is given a 120 m position prior; RoMa is not.

Default `BENCH_ROMA_CROP=shared` hands RoMa exactly the crop the sketch arm
gets, footprint + lattice, and lets it answer wherever it likes.
`BENCH_ROMA_CROP=footprint` instead hands it only a footprint-sized crop
centred on the prior, which bounds the answer through the input rather than
through a rejection rule; that arm sees LESS map than the sketch matcher and
must say so wherever it is reported.

Everything else, the matcher, sampling, confidence floor, RANSAC, the stable
per-pid seed and the principal-point convention, is the RUN 031 driver's own
code, imported rather than copied.

Output: matrix_roma_<pop>.jsonl, append-only and resumable per pid.

    BENCH_POP=visloc python run_matrix_roma.py
"""
import io
import json
import os
import random
import sys
import time

import numpy as np

POP = os.environ.get("BENCH_POP", "visloc").lower()
assert POP in ("uavscenes", "visloc"), "bad BENCH_POP %r" % POP

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
BENCH = os.path.join(ROOT, "results", "2026-09-08_sketch_vs_roma_bench")
RTKD = os.path.join(ROOT, "results", "2026-09-14_rtk114_indomain_table")
WIND = os.path.join(ROOT, "results", "2026-09-15_windowed_roma")

for p in (WIND, BENCH, RTKD, os.path.join(ROOT, "code", "src")):
    sys.path.insert(0, p)
os.chdir(WIND)

import torch  # noqa: E402

import bench_sketch_vs_roma as B                 # noqa: E402
import run_roma_windowed_rtk114 as W             # noqa: E402

TD = B.TD


def _pop_tag():
    """Name the output after the population actually run, not the loader
    branch, so a substituted pid list cannot be mistaken for the default."""
    pids_file = os.environ.get("BENCH_PIDS")
    if not pids_file:
        return POP
    tag = os.path.splitext(os.path.basename(pids_file))[0]
    return tag[:-5] if tag.endswith("_pids") else tag


OUT = os.path.join(HERE, "matrix_roma_%s.jsonl" % _pop_tag())


def uniform_crop(pk):
    """crop_shared with the jitter widened to the full box half-width."""
    rng = random.Random(B._stable_pid_seed(pk.pid))
    jitter_px = int(round(TD.SEARCH_MARGIN_M / pk.eff))
    x0, y0, crop_hw = TD.local_map_geom(pk, jitter_px=jitter_px, rng=rng)
    return pk.ortho[y0:y0 + crop_hw[0], x0:x0 + crop_hw[1]], x0, y0


def repair_pool_paths(pool):
    """Drop a stray `legacy/` segment from pool paths that do not resolve.

    Job 17866969 lost all 36 hkairport and all 33 amtown01 pairs to

        FileNotFoundError: <DATA_ROOT>/results/
                           2026-09-09_rtk_uniform_pool_pilot/orthos/
                           amtown01_000000_rtku.npy

    while every real path on that tree is <DATA_ROOT>/results/... . The
    valley and island pairs survived because they are ortho-rectified and read
    their ortho from elsewhere; only the flat-render pairs go through
    `orthos_dir`, which is why the split was exactly 69 against 45. Note
    `preflight_files` reported all 114 present, so it checks a different
    location than `C.load_ortho` does for flat pairs.

    This only ever acts when the baked path is MISSING and the corrected one
    EXISTS, so it cannot silently redirect a working tree, and it reports what
    it changed. If neither resolves, the entry is left alone and the original
    error still names the original path.
    """
    fixed = 0
    for e in pool.values():
        for key in ("npz_dir", "prepack_dir", "orthos_dir", "orthos_ortho_dir"):
            p = e.get(key)
            if not p or os.path.isdir(p):
                continue
            alt = p.replace("/legacy/results/", "/results/")
            if alt != p and os.path.isdir(alt):
                e[key] = alt
                fixed += 1
    if fixed:
        print("pool: repaired %d directory paths that carried a stray "
              "'legacy/' segment" % fixed, flush=True)
    return fixed


def patch_load_ortho(CC):
    """Let load_ortho fall back to .png when the .npy is not there.

    `load_ortho` forces the .npy extension whenever a pool entry carries
    "orthos_dir", and its PNG branch is only reachable when that key is absent:

        if orthos_dir is not None:
            return np.load(os.path.join(orthos_dir, pair_id + ".npy"), ...)
        p = os.path.join(R10, "orthos", pair_id + ".png")   # unreachable

    But the pilot pool stores FLAT-render orthos as .png in `orthos/` and
    ortho-rectified ones as .npy in `orthos_ortho/`. Any flat pair whose entry
    carries "orthos_dir" is therefore looked up with the wrong extension, which
    is exactly the 36 hkairport and 33 amtown01 pairs that died in jobs
    17865984, 17866969 and 17867712 while the 45 ortho-rectified valley and
    island pairs loaded fine.

    This only ever acts when the .npy is MISSING and a .png of the same name
    exists beside it, so it cannot change how any currently-working pair loads.
    """
    import cv2
    _orig = CC.load_ortho

    def load_ortho_any(pair_id):
        d = (CC.pool().get(pair_id) or {}).get("orthos_dir")
        if d:
            npy = os.path.join(d, pair_id + ".npy")
            png = os.path.join(d, pair_id + ".png")
            if not os.path.exists(npy) and os.path.exists(png):
                bgr = cv2.imread(png, cv2.IMREAD_COLOR)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return _orig(pair_id)

    CC.load_ortho = load_ortho_any

def population_pids():
    if POP == "uavscenes":
        import rtk114_common as K
        # `use_local_pool_if_needed` inspects `C.POOL_FILE`, a module global.
        # Importing bench_sketch_vs_roma leaves it pointing at the VisLoc pool,
        # whose entries have no "prepack_dir" key, and job 17865725 died on
        # exactly that: KeyError: 'prepack_dir'. Point it back at the RTK pool
        # and drop the cache first, so this does not depend on import order.
        # configure the module PairPack actually loads through, not whichever
        # copy `import buildc_common` bound: two are on sys.path. See the
        # sketch driver's own note and job 17866969's traceback.
        CC = TD.C
        CC.POOL_FILE = os.path.join(K.ROUND6, "POOL_C51_RTK.json")
        CC._POOL = None
        print("pool file in force:", K.use_local_pool_if_needed(), flush=True)
        print("pool module:", CC.__file__, flush=True)
        repair_pool_paths(CC.pool())
        patch_load_ortho(CC)
        pids = K.pool_pids()
        if len(pids) != 114:
            sys.exit("FATAL: expected 114 UAVScenes pids, got %d" % len(pids))
        K.preflight_files(pids)
        return pids

    # Repoint a pid at the local copy ONLY IF that pid's file is really there.
    # Guarding on the directory is not enough: it exists on HPC and is merely
    # incomplete, which is how job 17865693 lost visloc11 a second time.
    pairs_dir = os.path.join(B.DATA_DIR, "pairs")
    moved = kept = 0
    for pid, e in B.C.pool().items():
        if not any(pid.startswith(s + "_") for s in B.SITES):
            continue
        if os.path.exists(os.path.join(pairs_dir, pid + ".npz")):
            e["npz_dir"] = pairs_dir
            e["prepack_dir"] = os.path.join(B.DATA_DIR, "prepack")
            e["orthos_dir"] = os.path.join(B.DATA_DIR, "orthos")
            moved += 1
        else:
            kept += 1
    print("pool: of %d entries for sites %s, %d have a local .npz and were "
          "repointed; %d keep the pool's own paths. (Whole-pool counts, not "
          "this run's population: sites outside %s are never touched.)"
          % (moved + kept, "/".join(B.SITES), moved, kept, "/".join(B.SITES)),
          flush=True)
    name = os.environ.get("BENCH_PIDS", "visloc_texture75_pids.json")
    return json.load(io.open(os.path.join(BENCH, name), encoding="utf-8"))


ROMA_CROP = os.environ.get("BENCH_ROMA_CROP", "shared").lower()
assert ROMA_CROP in ("footprint", "shared"), "bad BENCH_ROMA_CROP %r" % ROMA_CROP


def lattice_span_m():
    """The side of the box the sketch matcher's answers are confined to."""
    return 2.0 * TD.SEARCH_MARGIN_M


def footprint_crop(mO_raw, pk):
    """Hand RoMa a map crop the size of the drone frame, centred on the prior.

    Why this replaces the in_window predicate (lead's suggestion, 2026-09-15).
    The shared crop is footprint + lattice, and the predicate then threw away
    any fit whose mapped principal point left the lattice. Two things went
    wrong with that in job 17865984:

      starvation  rejection happens after sampling, so the sampler had to
                  stumble into a small admissible set by chance. The draw cap
                  was hit on 61 of 75 pairs, with a median of 2 admitted
                  hypotheses out of the 2000 wanted. That is not a search.
      anchoring   errors reached 234 m on a ~120 m window, so the admissible
                  box was in the wrong place: it was anchored pp..pp+window_px,
                  which is the UAVScenes convention, not this population's.

    Bounding the INPUT instead of censoring the OUTPUT fixes both. RoMa is
    shown only the footprint-sized region centred on the prior, so a wildly
    wrong answer has no supporting imagery to be drawn to, and nothing is
    rejected, so nothing starves.

    Geometry, measured on this population: the truth sits a median 37.0 m from
    the window centre, p90 63.4 m, worst 74.2 m. Against a 203x187 m footprint
    that leaves 66 percent of the frame overlapping at the median, 45 percent
    at p90 and 38 percent at worst. Dense matching works comfortably there.

    HONEST COST, and it belongs in the paper: RoMa now sees LESS map than the
    sketch matcher, which still gets footprint + lattice. Report the overlap
    figures above beside any RoMa failure so this is visible rather than
    buried. `BENCH_ROMA_CROP=shared` restores the old behaviour for comparison.
    """
    if ROMA_CROP == "shared":
        return mO_raw, 0, 0
    hA, wA = TD.drone_map_dims(pk)
    H, W_ = mO_raw.shape[:2]
    # the shared crop is footprint + lattice, so the prior-centred footprint
    # window starts at half the lattice span on each axis
    oy = max(0, (H - hA) // 2)
    ox = max(0, (W_ - wA) // 2)
    sub = mO_raw[oy:oy + hA, ox:ox + wA]
    return np.ascontiguousarray(sub), ox, oy


# The budget-equalised constrained RANSAC that lived here has been REMOVED.
# It existed to make the in_window predicate fair by counting admitted rather
# than attempted hypotheses. Job 17865984 showed the predicate itself was the
# problem, not its budget: 61 of 75 pairs hit the draw cap with a median of 2
# admitted hypotheses, and the admissible box was mis-anchored besides. Keeping
# a working implementation of a discarded idea in the file is a trap, so it is
# gone rather than commented out. Recover it from git if the predicate is ever
# revisited; the reasoning is in this module's docstring.



def main():
    print("cell: arm=roma pop=%s  (no rotation arm: RoMa takes no angle)" % POP,
          flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device,
          torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)
    if device != "cuda":
        sys.exit("FATAL: no CUDA device. RoMa needs 9.71 GB and this arm must "
                 "not run on CPU. Refusing to produce hours of numbers that "
                 "differ in nothing but runtime.")

    B.crop_shared = uniform_crop              # the protocol change

    # NOTHING IS REJECTED, in either crop mode. The in_window predicate is
    # gone: scored by distance to truth, freedom to answer anywhere is a
    # LIABILITY, not an advantage, because an answer outside the lattice simply
    # produces a large error. The real asymmetry runs the other way and is
    # stated rather than engineered away: the sketch matcher gets a 120 m
    # position prior and RoMa does not.
    _orig = W.fit_rigid_ransac

    def _unwindowed(src, dst, rng, thr=W.RANSAC_THR_PX, iters=2000,
                    window_px=None, pp=None):
        return _orig(src, dst, rng, thr=thr, iters=iters,
                     window_px=None, pp=None)

    W.fit_rigid_ransac = _unwindowed
    print("crop: %s. No answer constraint; RoMa may answer anywhere and is "
          "scored on distance to truth like every other arm."
          % ("footprint-sized, centred on the prior" if ROMA_CROP == "footprint"
             else "the shared footprint+lattice crop, same as the sketch arm"),
          flush=True)
    pids = population_pids()
    print("pids:", len(pids), flush=True)

    model = W.RoMaV2()
    model.H_hr = model.W_hr = 1280            # TRUE precise, no VRAM downgrade
    print("RoMa loaded, H_hr=W_hr=%d" % model.H_hr, flush=True)

    # Resume skips only pids with a REAL outcome. A crashed pid is retried,
    # because counting a crash as done makes the hole permanent across
    # re-submissions while the row count still looks right.
    done, retry = set(), 0
    if os.path.exists(OUT):
        keep = []
        for line in io.open(OUT, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error") is None and r.get("roma") is not None:
                done.add(r["pid"])
                keep.append(line)
            else:
                retry += 1
        if retry:
            with io.open(OUT, "w", encoding="utf-8") as f:
                f.writelines(keep)
        print("resuming: %d done, %d crashed rows dropped for retry"
              % (len(done), retry), flush=True)

    fh = io.open(OUT, "a", encoding="utf-8")
    t_start = time.perf_counter()
    for i, pid in enumerate(pids, 1):
        if pid in done:
            continue
        rec = dict(pid=pid, arm="roma", pop=POP, jitter="uniform",
                   region_m=120, convention="principal_point",
                   gt_in_estimator=False, citable=True)
        try:
            pk = TD.PairPack(pid)
            rec["eff"] = float(pk.eff)
            mO_raw, x0, y0 = uniform_crop(pk)
            sub, ox, oy = footprint_crop(mO_raw, pk)
            rec["roma_crop"] = ROMA_CROP
            rec["crop"] = dict(x0=int(x0 + ox), y0=int(y0 + oy),
                               h=int(sub.shape[0]), w=int(sub.shape[1]),
                               shared_h=int(mO_raw.shape[0]),
                               shared_w=int(mO_raw.shape[1]))
            rec["roma"] = W.run_roma_ppfix(model, pk, sub, x0 + ox, y0 + oy)
            # Whether RoMa's free answer lands inside the box the sketch
            # matcher is confined to is worth reporting, and is the honest form
            # of the "same position freedom" question: state how far the two
            # answer spaces differ in practice rather than censoring one.
            #
            # It is NOT derivable from err_m and center_hit_err_m, which are two
            # distances that only bound the answer's offset from the window
            # centre, never fix it. So record the geometry needed to compute it
            # exactly offline and do not guess it here: run_roma_ppfix already
            # stores the predicted E, N, and crop/eff/lattice_span_m below give
            # the window.
            rec["lattice_span_m"] = lattice_span_m()
        except Exception as exc:
            # Carry the path. Job 17865984 lost all 36 hkairport and all 33
            # amtown01 pairs to a bare FileNotFoundError while `preflight_files`
            # reported "npz+prepack+ortho all present" for the same 114 pids,
            # so preflight and C.load_ortho are checking different paths and
            # the repr alone cannot say which.
            rec["error"] = repr(exc)
            fn = getattr(exc, "filename", None)
            if fn:
                rec["error_path"] = fn
                rec["error"] = "%s: %s" % (rec["error"], fn)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        r = rec.get("roma", {})
        print(json.dumps(dict(n=i, pid=pid, n_conf=r.get("n_conf"),
                              n_inliers=r.get("n_inliers"),
                              answered=r.get("answered"),
                              err_m=r.get("err_m"),
                              why=r.get("exclude_reason"),
                              err=rec.get("error"),
                              el_min=round((time.perf_counter() - t_start) / 60.0, 1))),
              flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print("\nwrote " + os.path.basename(OUT), flush=True)


if __name__ == "__main__":
    main()
