"""The sketch-side cells of the full rerun matrix. Lead request, 2026-09-15.

Protocol, fixed for every cell so the cells are comparable:

  jitter    UNIFORM over the box half-width, jitter_px = SEARCH_MARGIN_M / eff,
            so the true position is spread across the box the matcher searches.
            The stock draw is +-CROP_JITTER_PX = 120 px against a 300 px
            half-width, which pins the truth to the central 40 percent and
            inflates the window-centre baseline. Measured, see
            results/2026-09-15_uniform_jitter/.
  region    the 120 m lattice, UNCHANGED. The "box or footprint whichever is
            bigger" rule is a floor on the CROP, already satisfied (crop
            323x307 m against a 203x187 m footprint); it is not an instruction
            to widen the lattice. Widening was measured separately and doubles
            the error, results/2026-09-15_box240/.
  rotation  see BENCH_ROT below.

Cells are selected by environment variable:

  BENCH_ARM   canny | distill | learned
  BENCH_POP   uavscenes | visloc
  BENCH_ROT   shipped | oracle

`shipped` evaluates the single angle the pair already carries (pk.Mr): -Phi1 on
UAV-VisLoc, the track-course render heading on UAVScenes.

`oracle` additionally sweeps 72 angles and reports BOTH the angle the matching
cost selects and the angle that MINIMISES ERROR AGAINST GROUND TRUTH.

    ####################################################################
    # THE ORACLE ANGLE PUTS GROUND TRUTH INSIDE THE ESTIMATOR.         #
    # Every oracle row carries gt_in_estimator=true and citable=false. #
    # It is a CEILING and a diagnostic: it bounds what any heading     #
    # source could achieve. It is not a system result and must never   #
    # be reported as one. Ground-truth firewall: the ground-truth firewall policy;  #
    # faculty comment 20.                                              #
    ####################################################################

BENCH_ROT=oracle is rejected for BENCH_POP=uavscenes. UAVScenes ships no Phi1
to correct, so there is no defective heading there to bound, and sweeping would
convert the one arm that works (5.7 m median) into a non-citable one.

Geometry, pooling, the native-centre rotation pivot and the two-sided learned
cost are REUSED from rotation_probe / rotation_corrected_c46_visloc_hpc rather
than reimplemented, so the 2026-09-11 pivot fix is inherited, not re-derived.

Output: matrix_<arm>_<pop>_<rot>.jsonl, one row per pid, append-only and
resumable per pid.

    BENCH_ARM=canny BENCH_POP=visloc BENCH_ROT=oracle python run_matrix_sketch.py
"""
import io
import json
import math
import os
import random
import sys
import time

import numpy as np

ARM = os.environ.get("BENCH_ARM", "canny").lower()
POP = os.environ.get("BENCH_POP", "visloc").lower()
ROT = os.environ.get("BENCH_ROT", "shipped").lower()

assert ARM in ("canny", "distill", "learned"), "bad BENCH_ARM %r" % ARM
assert POP in ("uavscenes", "visloc"), "bad BENCH_POP %r" % POP
assert ROT in ("shipped", "oracle"), "bad BENCH_ROT %r" % ROT
if ROT == "oracle" and POP == "uavscenes":
    sys.exit("REFUSED: UAVScenes ships no Phi1 to correct, and sweeping "
             "rotation there would put ground truth inside the one arm that "
             "works. Run BENCH_ROT=shipped on uavscenes.")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
BENCH = os.path.join(ROOT, "results", "2026-09-08_sketch_vs_roma_bench")
RTKD = os.path.join(ROOT, "results", "2026-09-14_rtk114_indomain_table")

for p in (BENCH, RTKD, os.path.join(ROOT, "code", "src")):
    sys.path.insert(0, p)
os.chdir(BENCH)

import torch  # noqa: E402

# CPU-compatibility shims: no-ops when a GPU is present, and on a CPU-only box
# they stop train_drawer's `torch.autocast("cuda", ...)` and `.pin_memory()`
# from raising "Cannot access accelerator device". Neither changes a value.
if not torch.cuda.is_available():
    import contextlib
    _oa = torch.autocast

    def _safe_autocast(device_type="cuda", *a, **k):
        if device_type == "cuda":
            return contextlib.nullcontext()
        return _oa(device_type, *a, **k)

    torch.autocast = _safe_autocast
    torch.Tensor.pin_memory = lambda self, *a, **k: self

import bench_sketch_vs_roma as B                     # noqa: E402
import buildc_common as C                            # noqa: E402
import rotation_probe as RP                          # noqa: E402
import rotation_probe_cost_argmin as RCA             # noqa: E402
import rotation_corrected_c46_visloc_hpc as M        # noqa: E402

TCF = B.TD.TRAIN_COARSE_F
SWEEP = list(range(-180, 180, 5))
FINE = 5

# Checkpoints. Candidates in the same style rtk114_common uses, because the
# local tree and the HPC tree do not agree on layout: on HPC these live under
# <DATA_ROOT>/results/2026-09-15_filtered_learned/, which is where the
# 2026-09-15 ablation job loaded them from (slurm_filtered_learned_17850121.out).
# Both are overridable, and the resolved path is printed and existence-checked
# before any GPU time is spent.
_FL = "results/2026-09-15_filtered_learned"
_CKPT_CANDIDATES = {
    "learned": [os.path.join(ROOT, _FL, "ckpt_full", "best.pt"),
                "<DATA_ROOT>/%s/ckpt_full/best.pt" % _FL,
                os.path.join(ROOT, "results",
                             "2026-09-12_buildc51_round6_l3off",
                             "checkpoints", "best.pt")],
    "distill": [os.path.join(ROOT, _FL, "ckpt_distill", "last.pt"),
                "<DATA_ROOT>/%s/ckpt_distill/last.pt" % _FL],
}


def _resolve_ckpt(arm):
    env = os.environ.get("BENCH_CKPT_%s" % arm.upper())
    if env:
        return env
    for p in _CKPT_CANDIDATES[arm]:
        if os.path.exists(p):
            return p
    return _CKPT_CANDIDATES[arm][0]     # report the first for the error message


def _pop_tag():
    """Name the output after the population actually run, not just the loader
    branch, so a cell run on a substituted pid list cannot be mistaken for the
    default one."""
    pids_file = os.environ.get("BENCH_PIDS")
    if not pids_file:
        return POP
    tag = os.path.splitext(os.path.basename(pids_file))[0]
    return tag[:-5] if tag.endswith("_pids") else tag


# The filter state is part of the identity of a cell, so it is part of the
# filename. Without it a filtered run would resume the unfiltered file and
# silently inherit its rows. Files written before 2026-09-15 18:20 carry no
# suffix and are the UNFILTERED-learned variants.
# `_gate` marks rows that carry peak_ratio. The earlier _cc1 files do not, and
# resume keys on a pid being present, so writing to the same name would skip
# every pair and produce no new data at all.
OUT = os.path.join(HERE, "matrix_%s_%s_%s_%s_gate.jsonl"
                   % (ARM, _pop_tag(), ROT,
                      "cc1" if os.environ.get("BENCH_CC_FILTER", "1") == "1"
                      else "cc0"))


def uniform_crop(pk):
    """crop_shared with the jitter widened to the full box half-width."""
    rng = random.Random(B._stable_pid_seed(pk.pid))
    jitter_px = int(round(B.TD.SEARCH_MARGIN_M / pk.eff))
    x0, y0, crop_hw = B.TD.local_map_geom(pk, jitter_px=jitter_px, rng=rng)
    return pk.ortho[y0:y0 + crop_hw[0], x0:x0 + crop_hw[1]], x0, y0


def maybe_local_pool():
    """Repoint a pid at the local copy ONLY IF that pid's file is really there.

    `B.make_local_pool` repoints every pid of `B.SITES` (visloc03/09/11) at
    `visloc_bench_data/`, unconditionally. Two jobs died on that:

      17865448  called it blind. All 25 visloc02 and all 25 visloc10 pairs
                answered; all 25 visloc11 pairs raised FileNotFoundError.
      17865693  guarded on `os.path.isdir(DATA_DIR)`, which is NOT enough:
                the directory DOES exist on HPC and is merely incomplete, so
                the guard passed, 2124 pids were repointed, and visloc11 died
                again at
                .../visloc_bench_data/pairs/visloc11_11_0212.npz

    Directory existence is the wrong question. The right one is per file, which
    is also what makes this correct on both trees at once without knowing which
    tree it is running on.
    """
    pairs_dir = os.path.join(B.DATA_DIR, "pairs")
    pool = B.C.pool()
    moved = kept = 0
    for pid, e in pool.items():
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


CC_FILTER = os.environ.get("BENCH_CC_FILTER", "1") == "1"


def apply_cc_filter():
    """Put the learned arms through the SAME long-structure filter as Canny.

    Faculty comment 19: the ablation confounds training with filtering. The
    Canny drawing applies a 75 px connected-component extent filter (15 m at
    0.20 m/px) inside compute_sketch_v08_fulllab, while the learned readout
    does not, because prob_to_edges stopped calling cc_filter in BUILD-C39. So
    a difference between the arms cannot be attributed to training.

    Patching at `bmr_edges`, the single point where the learned probability
    field becomes discrete, is the same one-variable intervention RUN 029 used.
    Canny is untouched: it already filters internally, so with this on all
    three arms are filtered identically.

    BENCH_CC_FILTER=0 restores the unfiltered learned readout, which is what
    the previously reported distillation-only and full-training numbers used.
    """
    if not CC_FILTER:
        print("cc_filter: OFF (learned arms unfiltered, Canny still filtered "
              "internally -- arms NOT comparable)", flush=True)
        return
    if ARM == "canny":
        print("cc_filter: canny filters internally, nothing to patch",
              flush=True)
        return
    _unfiltered = B.TD.bmr_edges

    def _filtered(p, budget, mask=None):
        e = _unfiltered(p, budget, mask)
        return B.C.cc_filter(e, B.C.MIN_EXT)

    B.TD.bmr_edges = _filtered
    print("cc_filter: ON for arm %r, matching Canny's own %d px extent filter"
          % (ARM, B.C.MIN_EXT), flush=True)

def population_pids():
    if POP == "uavscenes":
        import rtk114_common as K
        # TWO buildc_common MODULES ARE LOADED. `import buildc_common as C`
        # above binds whichever copy sys.path reaches first; the traceback in
        # job 17866969 shows PairPack loading through
        # 2026-09-01_buildc50_visloc_final/buildc_common.py while
        # rtk114_common uses the round-6 copy. Setting C.POOL_FILE configured
        # one module while the data loaded through the other, which is why the
        # earlier "fix" removed the KeyError and changed nothing else.
        #
        # `B.TD.C` is the module PairPack actually resolves load_pair and
        # load_ortho through, so that is the one to configure. Reaching it via
        # the loader rather than via an import makes this correct regardless of
        # sys.path order.
        CC = B.TD.C
        CC.POOL_FILE = os.path.join(K.ROUND6, "POOL_C51_RTK.json")
        CC._POOL = None
        print("pool file in force:", K.use_local_pool_if_needed(), flush=True)
        print("pool module:", CC.__file__, flush=True)
        repair_pool_paths(CC.pool())
        patch_load_ortho(CC)
        pids = K.pool_pids()
        if len(pids) != 114:
            sys.exit("FATAL: expected 114 UAVScenes pids, got %d" % len(pids))
        return pids
    maybe_local_pool()
    # BENCH_PIDS lets a smaller pid list be substituted, which is how this is
    # smoke-tested on the local tree: only visloc03/09/11 are synced here, and
    # visloc_texture75_pids.json is sites 02/10/11.
    name = os.environ.get("BENCH_PIDS", "visloc_texture75_pids.json")
    return json.load(io.open(os.path.join(BENCH, name), encoding="utf-8"))


def cost_grid_at_angle(ctx, angle_deg):
    """The cost grid itself, which the per-angle cost functions discard.

    `B.peak_ratio_gate` needs the whole 2D surface, not its minimum, and both
    `rotation_probe_cost_argmin.cost_and_err_at_angle` and
    `rotation_corrected_c46_visloc_hpc.c46_cost_err_at_angle` return only
    (min_cost, err). Their bodies are mirrored here so the gate is read off the
    same surface the reported answer came from, not a recomputation that might
    differ.

    Called ONCE per pair, at the angle actually reported, never inside the
    72-angle sweep, so it adds one correlation per pair rather than 72.
    """
    import cv2
    Mr = cv2.getRotationMatrix2D(
        (ctx["wA_native"] / 2.0, ctx["hA_native"] / 2.0), angle_deg, 1.0)
    if ARM == "canny":
        Wrot = cv2.warpAffine(ctx["drone_edges"], Mr, (ctx["wA"], ctx["hA"]),
                              flags=cv2.INTER_NEAREST)
        Wc = RCA.RC.coarse_pool(Wrot.astype(np.float32), TCF)
        return RCA.RC.valid_correlate(Wc, ctx["Pc"])
    Wrot = cv2.warpAffine(ctx["pAc_np"], Mr, (ctx["wA"], ctx["hA"]),
                          flags=cv2.INTER_LINEAR)
    Wc = M.pool_pad(Wrot, TCF, val=0.0)
    Pdrone_rot = cv2.warpAffine(ctx["P_drone_full"], Mr, (ctx["wA"], ctx["hA"]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
    Pc_drone = M.pool_pad(Pdrone_rot, TCF, val=1.0)
    cost_A = M.valid_correlate_torch(Wc, ctx["Pc_map"])
    cost_B = M.valid_correlate_windowed_torch(Pc_drone, ctx["Mc"])
    return 0.5 * (cost_A + cost_B)


def centre_err_m(ctx, grid_hw):
    """Error of returning the middle of the cost grid, this pair's baseline."""
    oh, ow = grid_hw
    return float(np.hypot((oh - 1) / 2.0 - ctx["true_r"],
                          (ow - 1) / 2.0 - ctx["true_c"])) * TCF * ctx["eff"]


def build_ctx(pid, pk, model):
    """Returns (ctx, per-angle fn, grid shape). The canvas is fixed across
    angles, so the cost grid shape is too, and one derivation serves the
    whole sweep."""
    if ARM == "canny":
        ctx = RP.prep_pid(pid)
        fn = RCA.cost_and_err_at_angle
        Pc = ctx["Pc"]
    else:
        mO_raw, x0, y0 = uniform_crop(pk)
        ctx = M.prep_c46_ctx(model, pk, mO_raw, x0, y0)
        if ctx["starved_drone"] or ctx["starved_map"]:
            return ctx, None, None
        fn = M.c46_cost_err_at_angle
        Pc = ctx["Pc_map"]
    oh = Pc.shape[0] - math.ceil(ctx["hA"] / float(TCF)) + 1
    ow = Pc.shape[1] - math.ceil(ctx["wA"] / float(TCF)) + 1
    return ctx, fn, (oh, ow)


def main():
    print("cell: arm=%s pop=%s rot=%s" % (ARM, POP, ROT), flush=True)
    print("device:", "cuda" if torch.cuda.is_available() else "cpu", flush=True)
    B.crop_shared = uniform_crop              # the protocol change
    apply_cc_filter()
    pids = population_pids()
    print("pids:", len(pids), flush=True)

    model = None
    if ARM != "canny":
        ckpt = _resolve_ckpt(ARM)
        if not os.path.exists(ckpt):
            sys.exit("FATAL: checkpoint not found for arm %r:\n  %s\nSet "
                     "BENCH_CKPT_%s to the right path before spending GPU time."
                     % (ARM, ckpt, ARM.upper()))
        print("checkpoint:", ckpt, flush=True)
        model = C.TinyDrawer()
        if torch.cuda.is_available():
            model = model.cuda()
        ck = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        model.eval()

    # Resume skips only pids with a REAL outcome. A crashed pid is left to be
    # retried, because treating a FileNotFoundError as "done" would make the
    # failure permanent across re-submissions: probe 17865448's 25 visloc11
    # crashes would never have been reattempted.
    done, retry = set(), 0
    if os.path.exists(OUT):
        keep = []
        for line in io.open(OUT, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("answered") or r.get("exclude_reason") == "starved":
                done.add(r["pid"])
                keep.append(line)
            else:
                retry += 1
        if retry:
            # drop the crashed rows so the file stays one row per pid
            with io.open(OUT, "w", encoding="utf-8") as f:
                f.writelines(keep)
        print("resuming: %d done, %d crashed rows dropped for retry"
              % (len(done), retry), flush=True)

    fh = io.open(OUT, "a", encoding="utf-8")
    for i, pid in enumerate(pids, 1):
        if pid in done:
            continue
        t0 = time.perf_counter()
        try:
            pk = B.TD.PairPack(pid)
            ctx, fn, grid_hw = build_ctx(pid, pk, model)
            if fn is None:
                fh.write(json.dumps(dict(
                    pid=pid, arm=ARM, pop=POP, rot=ROT, answered=False,
                    exclude_reason="starved")) + "\n")
                fh.flush()
                print("%3d/%d %-24s STARVED" % (i, len(pids), pid), flush=True)
                continue

            shipped_cost, shipped_err = fn(ctx, ctx["orig_angle"])
            rec = dict(
                pid=pid, arm=ARM, pop=POP, rot=ROT, answered=True,
                jitter="uniform", region_m=120, cc_filter=CC_FILTER,
                shipped_angle_deg=round(float(ctx["orig_angle"]), 2),
                shipped_err_m=round(shipped_err, 1),
                center_hit_err_m=round(centre_err_m(ctx, grid_hw), 1),
                grid_shape=[int(grid_hw[0]), int(grid_hw[1])],
                eff=float(ctx["eff"]),
                gt_in_estimator=False, citable=True)

            if ROT == "oracle":
                swept = [(a,) + fn(ctx, a) for a in SWEEP]
                coarse = min(swept, key=lambda t: t[1])
                fine = [(a,) + fn(ctx, a)
                        for a in range(coarse[0] - FINE, coarse[0] + FINE + 1)]
                by_cost = min(fine, key=lambda t: t[1])
                by_err = min(swept, key=lambda t: t[2])
                rec.update(
                    cost_angle_deg=float(by_cost[0]),
                    cost_err_m=round(by_cost[2], 1),
                    sweep_cost_std=round(
                        float(np.std([c for _, c, _ in swept])), 6),
                    # ---- ground truth selected the angle below ----
                    oracle_diag_angle_deg=float(by_err[0]),
                    oracle_diag_err_m=round(by_err[2], 1),
                    oracle_diag_note=("GT-SELECTED CEILING, NOT A SYSTEM "
                                      "RESULT: bounds what any heading source "
                                      "could achieve"),
                    gt_in_estimator=True, citable=False)

            # The confidence gate, read off the surface at the REPORTED angle.
            # Recorded, never acted on: no accuracy figure here is filtered by
            # it. peak_ratio is null if the grid is too small for a runner-up
            # peak, which is a real outcome rather than a value to invent.
            reported_angle = (float(rec["oracle_diag_angle_deg"])
                              if ROT == "oracle" else ctx["orig_angle"])
            try:
                ratio, _, _ = B.peak_ratio_gate(cost_grid_at_angle(
                    ctx, reported_angle))
            except Exception as exc:
                ratio = None
                rec["peak_ratio_error"] = repr(exc)
            rec["gate_angle_deg"] = round(float(reported_angle), 2)
            rec["peak_ratio"] = round(float(ratio), 4) if ratio is not None else None
            rec["answered_gated"] = bool(ratio is not None
                                         and ratio >= B.OP_PEAK_RATIO)
            rec["gate_threshold"] = B.OP_PEAK_RATIO

            rec["t_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            extra = ""
            if ROT == "oracle":
                extra = "  cost=%6.1f oracle=%6.1f" % (rec["cost_err_m"],
                                                       rec["oracle_diag_err_m"])
            print("%3d/%d %-24s shipped=%6.1f centre=%6.1f%s  %.1fs"
                  % (i, len(pids), pid, shipped_err, rec["center_hit_err_m"],
                     extra, rec["t_ms"] / 1000.0), flush=True)
        except Exception as exc:
            # repr() alone drops the path on FileNotFoundError, which is what
            # made job 17865448's visloc11 failures take a code read to find.
            why = repr(exc)
            fn = getattr(exc, "filename", None)
            if fn:
                why = "%s: %s" % (why, fn)
            fh.write(json.dumps(dict(pid=pid, arm=ARM, pop=POP, rot=ROT,
                                     answered=False,
                                     exclude_reason=why)) + "\n")
            fh.flush()
            print("%3d/%d %-24s FAIL %s" % (i, len(pids), pid, why), flush=True)
    fh.close()
    print("\nwrote " + os.path.basename(OUT), flush=True)


if __name__ == "__main__":
    main()
