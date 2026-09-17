"""Directional test of the sketch-geometry theory (report/Theory/theory-main.tex).

THE PREDICTION UNDER TEST
-------------------------
Near the ideal translation t*, the capped chamfer cost is

    C(t* + d) = sum_i alpha_i |n_i^T d| + O(|d|^2),   G = sum_i alpha_i n_i n_i^T

so the direction v_min (eigenvector of G's smallest eigenvalue) is the one
along which the cost rises least. The directional prediction: the matcher's
position error should run preferentially ALONG v_min. This script tests that
directly. results/2026-09-15_boundary_geometry/ tested only the magnitude
lambda_min against error magnitude and found nothing (Spearman +0.024), with
lambda_min saturated near its 0.5 ceiling for 95 of 114 sketches.

WHAT IS RECOMPUTED, AND HOW IT IS CHECKED
-----------------------------------------
The sealed matrix rows (matrix_canny_uavscenes_shipped_cc1_gate.jsonl) store
only |error|. The error VECTOR is recomputed by mirroring, not editing, the
exact path run_matrix_sketch.py used for the Canny/UAVScenes/shipped cell:

    B.crop_shared := uniform_crop  (jitter_px = SEARCH_MARGIN_M / eff, seeded
                                    random.Random(B._stable_pid_seed(pid)))
    ctx = rotation_probe.prep_pid(pid)
    Wrot = warpAffine(drone_edges, getRotationMatrix2D(native centre,
                                                       orig_angle, 1.0))
    cost = run_classical.valid_correlate(coarse_pool(Wrot), ctx["Pc"])
    (r_est, c_est) = argmin cost;  dr_m, dc_m against (true_r, true_c)

Per pid, the recomputed |error| (rounded to 0.1 m) and the recomputed
peak_ratio (rounded to 4 dp) are compared with the sealed row. Both must
match before anything downstream is trusted; the match rate is reported.

FRAMES
------
Everything directional lives in the cost-grid frame: x = column (east on the
map crop), y = row (down). The error vector is e = (dc_m, dr_m). G is computed
on the UNROTATED drone sketch in its own pixel frame (structure-tensor code
imported from boundary_geometry.py, unchanged) and carried into the grid frame
with G_grid = R G R^T, R the linear part of the warp matrix. As a check the
same G is also computed directly on the warped sketch Wrot; the two weak
directions are compared per pair.

E1  error direction vs weak direction.
    theta = angle between e and v_min, folded to [0, 90]. Null: theta uniform
    on [0, 90], mean 45. Statistics: mean, median, fraction below 45, mean
    cos(2 theta) (0 under the null, +1 for perfect alignment) with bootstrap
    CI, a permutation test that re-pairs error directions with the weak
    directions of OTHER pairs (this null keeps both marginals, so it is immune
    to grid quantisation of e and to any clustering of v_min), and the
    G-quadratic form q = e^T G e / |e|^2 (0.5 under an isotropic error
    direction; below 0.5 if the error prefers the weak direction; this weights
    anisotropic sketches more, automatically).
    Four weak-direction variants: v_min of G (rotated analytically), v_min of
    G computed on Wrot, the L1-cone minimiser argmin_phi sum_i alpha_i
    |n_i . d(phi)| (the direction the theorem's first-order term actually
    minimises; equals v_min only for symmetric normal distributions), and
    v_min of the INLIER-restricted G_eff (E2, below).
    Subsets: all; |e| > 5 m; 5 < |e| <= 30 m (in-basin, meaningful); |e| > 30
    m (wrong basin); lambda_min below its median; |e| > 5 m and lambda_min
    below median; per scene.

E1b cost-surface geometry vs G (mechanism check, needs no error at all).
    The theorem is about the cost surface. Around the argmin cell, directional
    slopes of the cost grid are measured in 12 axial directions (bilinear
    sampling at radius 1 and 2 cells) and compared with the theory's predicted
    profile s(phi) = sum_i alpha_i |n_i . d(phi)|: per pair the angle between
    the empirical and predicted weak directions, and the Pearson correlation
    of the two 12-point profiles. Also done at the TRUE cell as a diagnostic
    (ground truth is used there for the diagnostic only, never in an
    estimate; the rows are labelled gt_diag).

E2  local geometry, to recover dynamic range.
    (a) G_eff: G over only the drone points that land within the 3 m cap of
        a map boundary at the argmin placement, i.e. the points that actually
        carry first-order cost. GT-free.
    (b) lambda_min of the MAP edges inside the placed canvas window, computed
        for every cell of the cost grid via integral images; read at the
        argmin (GT-free) and at the true cell (gt_diag).
    (c) 3x3 tiles of the drone sketch: min over tiles of lambda_min.
    For each: spread (quantiles, IQR) and Spearman with |e| with a bootstrap
    CI, plus per-scene Spearman.

E3  only if E1 or E2 shows signal (a CI that excludes the null): a
    leave-one-scene-out matched-coverage comparison of rho alone against rho
    combined with the local geometry score.

Ground truth: enters only the scoring (error vector) and the two gt_diag
diagnostics, never any estimate.

    CUDA_VISIBLE_DEVICES=-1 python results/2026-09-15_directional_geometry/directional_geometry.py
"""
import io
import json
import math
import os
import random
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(PROJ, "results")
RTK114 = os.path.join(RES, "2026-09-14_rtk114_indomain_table")
BENCH = os.path.join(RES, "2026-09-08_sketch_vs_roma_bench")
BGEO = os.path.join(RES, "2026-09-15_boundary_geometry")
MATRIX = os.path.join(RES, "2026-09-15_rerun_matrix",
                      "matrix_canny_uavscenes_shipped_cc1_gate.jsonl")

import torch                             # noqa: E402
if not torch.cuda.is_available():
    import contextlib
    _oa = torch.autocast

    def _safe_autocast(device_type="cuda", *a, **k):
        if device_type == "cuda":
            return contextlib.nullcontext()
        return _oa(device_type, *a, **k)
    torch.autocast = _safe_autocast
    torch.Tensor.pin_memory = lambda self, *a, **k: self

sys.path.insert(0, RTK114)
import rtk114_common as K                # noqa: E402  round-6 C/TD first, then B
C, TD, B = K.C, K.TD, K.B
sys.path.insert(0, BENCH)
import run_classical as RC               # noqa: E402
import rotation_probe as RP              # noqa: E402
sys.path.insert(0, BGEO)
import boundary_geometry as BG           # noqa: E402  structure tensor, reused

import cv2                               # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

TCF = TD.TRAIN_COARSE_F
EPS = 1e-12
SEED = 20260915
BOOT = 10000
NPERM = 20000
PHI_GRID = np.deg2rad(np.arange(0, 180, 1.0))       # axial directions, 1 deg
SLOPE_DIRS = np.deg2rad(np.arange(0, 180, 15.0))    # 12 axial directions
SLOPE_RADII = (1.0, 2.0)                            # cells (1.6 m, 3.2 m)


# ----------------------------------------------------------------------
# the exact crop the reported cell used (mirrored from run_matrix_sketch.py)
# ----------------------------------------------------------------------
LAST_CROP = {}


def uniform_crop(pk):
    rng = random.Random(B._stable_pid_seed(pk.pid))
    jitter_px = int(round(TD.SEARCH_MARGIN_M / pk.eff))
    x0, y0, crop_hw = TD.local_map_geom(pk, jitter_px=jitter_px, rng=rng)
    LAST_CROP[pk.pid] = (int(x0), int(y0), tuple(int(v) for v in crop_hw))
    return pk.ortho[y0:y0 + crop_hw[0], x0:x0 + crop_hw[1]], x0, y0


def repair_pool_paths(pool):
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
    return fixed


def patch_load_ortho(CC):
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


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------
def normal_angles(edges, sigma_d=BG.SIGMA_D, sigma_t=BG.SIGMA_T):
    """Per edge pixel: (ys, xs, theta) with theta the structure-tensor normal
    angle in [0, pi) in (x=col, y=row) coordinates. Same fields as
    boundary_geometry.gram_from_edges, exposed per pixel."""
    e = np.asarray(edges).astype(bool)
    jxx, jxy, jyy = BG.structure_tensor_fields(e, sigma_d, sigma_t)
    ys, xs = np.nonzero(e)
    a, b, c = jxx[ys, xs], jxy[ys, xs], jyy[ys, xs]
    keep = (a + c) > EPS
    ys, xs = ys[keep], xs[keep]
    th = 0.5 * np.arctan2(2.0 * b[keep], a[keep] - c[keep]).astype(np.float64)
    th = np.mod(th, np.pi)
    return ys, xs, th


def gram_from_angles(th, w=None):
    if th.size == 0:
        return None
    if w is None:
        mc, ms = float(np.cos(2 * th).mean()), float(np.sin(2 * th).mean())
    else:
        w = np.asarray(w, float)
        s = w.sum()
        if s <= 0:
            return None
        mc = float((w * np.cos(2 * th)).sum() / s)
        ms = float((w * np.sin(2 * th)).sum() / s)
    return 0.5 * np.array([[1.0 + mc, ms], [ms, 1.0 - mc]])


def eig2(g):
    """(lambda_min, lambda_max, v_min, v_max) of a symmetric 2x2, v as (x,y)."""
    ev, V = np.linalg.eigh(g)
    ev = np.clip(ev, 0.0, None)
    return float(ev[0]), float(ev[1]), V[:, 0].copy(), V[:, 1].copy()


def axial_angle_deg(v):
    """Direction of an axis in [0, 180) degrees, (x, y) frame."""
    return float(np.mod(np.degrees(np.arctan2(v[1], v[0])), 180.0))


def fold90(a_deg, b_deg):
    """Angle between two axes, in [0, 90]."""
    d = abs((a_deg - b_deg + 90.0) % 180.0 - 90.0)
    return float(d)


def l1_profile(th, w=None):
    """s(phi) = sum_i alpha_i |cos(theta_i - phi)| on PHI_GRID (the theorem's
    first-order term for a unit step in direction phi)."""
    h, _ = np.histogram(np.mod(th, np.pi), bins=180, range=(0.0, np.pi),
                        weights=w)
    h = h / max(h.sum(), EPS)
    centres = (np.arange(180) + 0.5) * np.pi / 180.0
    M = np.abs(np.cos(centres[None, :] - PHI_GRID[:, None]))
    return M @ h


def rot_linear(angle_deg):
    Mr = cv2.getRotationMatrix2D((0.0, 0.0), angle_deg, 1.0)
    return np.asarray(Mr[:, :2], dtype=np.float64)


def cost_grid_shipped(ctx):
    """Mirror of rotation_probe_cost_argmin.cost_and_err_at_angle at
    ctx['orig_angle'], returning the whole grid and the warp."""
    Mr_test = cv2.getRotationMatrix2D(
        (ctx["wA_native"] / 2.0, ctx["hA_native"] / 2.0), ctx["orig_angle"], 1.0)
    Wrot = cv2.warpAffine(ctx["drone_edges"], Mr_test, (ctx["wA"], ctx["hA"]),
                          flags=cv2.INTER_NEAREST)
    Wc = RC.coarse_pool(Wrot.astype(np.float32), TCF)
    cost = RC.valid_correlate(Wc, ctx["Pc"])
    return cost, Wrot, np.asarray(Mr_test, dtype=np.float64)


def directional_slopes(cost, r0, c0):
    """Empirical slope of the cost per cell in each axial direction of
    SLOPE_DIRS around (r0, c0), averaged over +-radius for each radius.
    nan where a sample would leave the grid."""
    oh, ow = cost.shape
    out = np.full(len(SLOPE_DIRS), np.nan)
    c_val = float(cost[r0, c0])
    for k, phi in enumerate(SLOPE_DIRS):
        vals = []
        for rad in SLOPE_RADII:
            for sgn in (1.0, -1.0):
                rr = r0 + sgn * rad * math.sin(phi)
                cc = c0 + sgn * rad * math.cos(phi)
                if rr < 0 or rr > oh - 1 or cc < 0 or cc > ow - 1:
                    vals = None
                    break
                v = map_coordinates(cost, [[rr], [cc]], order=1, mode="nearest")[0]
                vals.append((float(v) - c_val) / rad)
            if vals is None:
                break
        if vals:
            out[k] = float(np.mean(vals))
    return out


def profile_stats(slopes, th_profile):
    """Compare the empirical 12-direction slope profile with the theory's
    L1 profile evaluated at the same directions."""
    if np.isnan(slopes).any():
        return dict(valid=False)
    idx = np.round(np.degrees(SLOPE_DIRS)).astype(int)
    pred = th_profile[idx]
    emp_weak = float(np.degrees(SLOPE_DIRS[int(np.argmin(slopes))]))
    pred_weak = float(np.degrees(SLOPE_DIRS[int(np.argmin(pred))]))
    r = BG.pearson(slopes, pred)
    return dict(valid=True, emp_weak_deg=emp_weak, pred_weak_deg=pred_weak,
                angle_deg=fold90(emp_weak, pred_weak), pearson=r,
                emp_min=float(slopes.min()), emp_max=float(slopes.max()),
                emp_aniso=float(slopes.min() / max(slopes.max(), EPS)),
                pred_aniso=float(pred.min() / max(pred.max(), EPS)))


def map_lambda_field(map_edges_crop, hA, wA, oh, ow):
    """lambda_min of the MAP edges inside the canvas window at every cost
    cell, via integral images. Window for cell (r, c) is
    [TCF*r : TCF*r + hA, TCF*c : TCF*c + wA] in crop pixels."""
    ys, xs, th = normal_angles(map_edges_crop)
    H, W = map_edges_crop.shape
    Hp, Wp = max(H, TCF * (oh - 1) + hA), max(W, TCF * (ow - 1) + wA)
    f = np.zeros((3, Hp, Wp), np.float64)
    f[0, ys, xs] = 1.0
    f[1, ys, xs] = np.cos(2 * th)
    f[2, ys, xs] = np.sin(2 * th)
    S = np.zeros((3, Hp + 1, Wp + 1), np.float64)
    S[:, 1:, 1:] = f.cumsum(1).cumsum(2)
    r = TCF * np.arange(oh)[:, None]
    c = TCF * np.arange(ow)[None, :]
    box = (S[:, r + hA, c + wA] - S[:, r, c + wA] - S[:, r + hA, c] + S[:, r, c])
    n = box[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        mc, ms = box[1] / n, box[2] / n
    lam = 0.5 * (1.0 - np.sqrt(mc ** 2 + ms ** 2))
    lam[n < 1] = np.nan
    return lam, n


def tile_lambda(edges, k=3):
    H, W = edges.shape
    out = []
    for i in range(k):
        for j in range(k):
            t = edges[i * H // k:(i + 1) * H // k, j * W // k:(j + 1) * W // k]
            if t.sum() < 50:
                continue
            g = BG.gram_from_edges(t)
            if g is not None:
                out.append(g["lambda_min"])
    return out


# ----------------------------------------------------------------------
# per pair
# ----------------------------------------------------------------------
def analyse_pair(pid, mrow):
    ctx = RP.prep_pid(pid)                          # uses patched crop_shared
    cost, Wrot, Mr_test = cost_grid_shipped(ctx)
    oh, ow = cost.shape
    r_est, c_est = np.unravel_index(np.argmin(cost), cost.shape)
    r_est, c_est = int(r_est), int(c_est)
    dr_m = (r_est - ctx["true_r"]) * TCF * ctx["eff"]
    dc_m = (c_est - ctx["true_c"]) * TCF * ctx["eff"]
    err_m = float(np.hypot(dr_m, dc_m))
    ratio, _, _ = B.peak_ratio_gate(cost)
    rec = dict(pid=pid, scene=pid.split("_")[0],
               err_m=round(err_m, 1), dc_m=float(dc_m), dr_m=float(dr_m),
               err_dir_deg=axial_angle_deg((dc_m, dr_m)),
               peak_ratio=(round(float(ratio), 4) if ratio is not None else None),
               shipped_err_m=mrow["shipped_err_m"],
               shipped_peak_ratio=mrow["peak_ratio"],
               grid_shape=[oh, ow], r_est=r_est, c_est=c_est,
               true_r=float(ctx["true_r"]), true_c=float(ctx["true_c"]),
               angle_deg=float(ctx["orig_angle"]), eff=float(ctx["eff"]))
    rec["err_match"] = abs(rec["err_m"] - mrow["shipped_err_m"]) < 0.051
    rec["ratio_match"] = (ratio is not None and mrow["peak_ratio"] is not None
                          and abs(rec["peak_ratio"] - mrow["peak_ratio"]) < 5e-4)

    # ---- G on the unrotated sketch, carried into the grid frame ----
    drone = ctx["drone_edges"].astype(bool)
    ys, xs, th = normal_angles(drone)
    G_d = gram_from_angles(th)
    R = rot_linear(ctx["orig_angle"])
    G = R @ G_d @ R.T
    lam_min, lam_max, v_min, v_max = eig2(G)
    # normal angles in the grid frame: theta' = theta - angle (see rot_linear)
    th_g = np.mod(th - math.radians(ctx["orig_angle"]), np.pi)
    prof = l1_profile(th_g)
    l1_weak = float(np.degrees(PHI_GRID[int(np.argmin(prof))]))
    rec.update(m=int(th.size), lambda_min=lam_min, lambda_max=lam_max,
               G=G.tolist(), vmin_deg=axial_angle_deg(v_min),
               l1_weak_deg=l1_weak,
               l1_aniso=float(prof.min() / max(prof.max(), EPS)))

    # ---- check: G computed directly on the warped sketch ----
    ysw, xsw, thw = normal_angles(Wrot.astype(bool))
    G_w = gram_from_angles(thw)
    lam_min_w, _, v_min_w, _ = eig2(G_w)
    rec.update(lambda_min_wrot=lam_min_w, vmin_wrot_deg=axial_angle_deg(v_min_w))
    rec["vmin_vs_wrot_deg"] = fold90(rec["vmin_deg"], rec["vmin_wrot_deg"])

    # ---- E2a: inlier-restricted G_eff at the argmin placement ----
    # P_map (unpooled capped DT) is not in ctx; rebuild it the way prep_pid
    # did, from the same crop.
    pk_prepack = C.load_prepack(pid)
    map_edges_full = np.asarray(pk_prepack["v08_map_edges"]).astype(bool)
    x0, y0, crop_hw = LAST_CROP[pid]       # the crop prep_pid just drew
    y1, x1 = min(y0 + crop_hw[0], map_edges_full.shape[0]), min(x0 + crop_hw[1], map_edges_full.shape[1])
    map_edges_crop = map_edges_full[y0:y1, x0:x1]
    P_map = C.capped_P(map_edges_crop)
    Hc, Wc_ = P_map.shape

    def inlier_G(r_cell, c_cell):
        yy = ysw + TCF * r_cell
        xx = xsw + TCF * c_cell
        ok = (yy >= 0) & (yy < Hc) & (xx >= 0) & (xx < Wc_)
        p = np.ones(ysw.shape, np.float32)
        p[ok] = P_map[yy[ok], xx[ok]]
        inl = p < 1.0                      # inside the 3 m cap
        w = np.clip(1.0 - p, 0.0, None)     # closer to a boundary, more weight
        out = dict(n_in=int(inl.sum()), frac_in=float(inl.mean()) if inl.size else 0.0)
        g = gram_from_angles(thw[inl])
        if g is not None and inl.sum() >= 30:
            a, b_, v, _ = eig2(g)
            out.update(lambda_min=a, vmin_deg=axial_angle_deg(v))
            profw = l1_profile(thw[inl])
            out["l1_weak_deg"] = float(np.degrees(PHI_GRID[int(np.argmin(profw))]))
        gw = gram_from_angles(thw, w)
        if gw is not None:
            a, b_, v, _ = eig2(gw)
            out.update(lambda_min_w=a, vmin_w_deg=axial_angle_deg(v))
        return out
    rec["eff_est"] = inlier_G(r_est, c_est)
    rt, ct = int(round(ctx["true_r"])), int(round(ctx["true_c"]))
    rt, ct = min(max(rt, 0), oh - 1), min(max(ct, 0), ow - 1)
    rec["eff_true_gt_diag"] = inlier_G(rt, ct)

    # ---- E2b: map-side lambda field ----
    lam_map, n_map = map_lambda_field(map_edges_crop, ctx["hA"], ctx["wA"], oh, ow)
    rec["map_lambda_est"] = float(lam_map[r_est, c_est]) if np.isfinite(lam_map[r_est, c_est]) else None
    rec["map_lambda_true_gt_diag"] = float(lam_map[rt, ct]) if np.isfinite(lam_map[rt, ct]) else None
    fin = lam_map[np.isfinite(lam_map)]
    rec["map_lambda_field"] = dict(median=float(np.median(fin)) if fin.size else None,
                                   iqr=float(np.percentile(fin, 75) - np.percentile(fin, 25)) if fin.size else None)
    # ---- E2c: drone tiles ----
    tl = tile_lambda(drone)
    rec["tile_lambda_min"] = float(min(tl)) if tl else None
    rec["tile_n"] = len(tl)

    # ---- E1b: cost-surface slopes vs the predicted profile ----
    rec["surf_est"] = profile_stats(directional_slopes(cost, r_est, c_est), prof)
    rec["surf_true_gt_diag"] = profile_stats(directional_slopes(cost, rt, ct), prof)
    rec["cost_min"] = float(cost[r_est, c_est])

    # ---- E1: angles between the error and each weak direction ----
    ed = rec["err_dir_deg"]
    rec["theta_vmin"] = fold90(ed, rec["vmin_deg"])
    rec["theta_vmin_wrot"] = fold90(ed, rec["vmin_wrot_deg"])
    rec["theta_l1"] = fold90(ed, rec["l1_weak_deg"])
    ee = rec["eff_est"]
    rec["theta_eff"] = fold90(ed, ee["vmin_deg"]) if "vmin_deg" in ee else None
    rec["theta_eff_w"] = fold90(ed, ee["vmin_w_deg"]) if "vmin_w_deg" in ee else None
    e = np.array([dc_m, dr_m]) / max(err_m, EPS)
    rec["q_G"] = float(e @ G @ e)          # 0.5 under isotropy
    return rec, cost, Wrot, G


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------
def cos2(theta_deg):
    return np.cos(np.deg2rad(2.0 * np.asarray(theta_deg, float)))


def align_stats(theta, label, rng):
    theta = np.asarray(theta, float)
    n = len(theta)
    if n < 3:
        return dict(label=label, n=int(n))
    c = cos2(theta)
    m = float(c.mean())
    boot = np.array([c[rng.integers(0, n, n)].mean() for _ in range(2000)])
    z = m * math.sqrt(2.0 * n)            # var(cos 2theta) = 1/2 under uniform
    p_uniform = 0.5 * math.erfc(z / math.sqrt(2.0))   # one-sided, alignment
    return dict(label=label, n=int(n), mean_theta=float(theta.mean()),
                median_theta=float(np.median(theta)),
                frac_below_45=float((theta < 45).mean()),
                mean_cos2=m, cos2_ci95=[float(np.percentile(boot, 2.5)),
                                         float(np.percentile(boot, 97.5))],
                z_uniform=z, p_uniform_onesided=p_uniform,
                hist_15deg=np.histogram(theta, bins=6, range=(0, 90))[0].tolist())


def perm_test(err_dir, weak_dir, rng, nperm=NPERM):
    """Re-pair error directions with other pairs' weak directions."""
    err_dir = np.asarray(err_dir, float)
    weak_dir = np.asarray(weak_dir, float)
    n = len(err_dir)
    if n < 3:
        return dict(n=int(n))
    d = (err_dir[:, None] - weak_dir[None, :] + 90.0) % 180.0 - 90.0
    Mc = np.cos(np.deg2rad(2.0 * d))      # n x n matrix of cos 2theta
    obs = float(np.trace(Mc) / n)
    null = np.empty(nperm)
    idx = np.arange(n)
    for k in range(nperm):
        p = rng.permutation(idx)
        null[k] = Mc[idx, p].mean()
    p_val = float((null >= obs).mean())
    return dict(n=int(n), observed_mean_cos2=obs, null_mean=float(null.mean()),
                null_sd=float(null.std()), p_perm_onesided=p_val,
                p_perm_twosided=float((np.abs(null - null.mean())
                                       >= abs(obs - null.mean())).mean()))


def subset_masks(recs):
    err = np.array([r["err_m"] for r in recs])
    lam = np.array([r["lambda_min"] for r in recs])
    med = float(np.median(lam))
    scenes = np.array([r["scene"] for r in recs])
    masks = [("all", np.ones(len(recs), bool)),
             ("err > 5 m", err > 5),
             ("5 < err <= 30 m (in-basin)", (err > 5) & (err <= 30)),
             ("err > 30 m (wrong basin)", err > 30),
             ("lambda_min < median (%.3f)" % med, lam < med),
             ("err > 5 m & lambda_min < median", (err > 5) & (lam < med)),
             ("err > 5 m & lambda_min < 0.40", (err > 5) & (lam < 0.40))]
    for s in sorted(set(scenes)):
        masks.append(("scene %s, err > 5 m" % s, (scenes == s) & (err > 5)))
    return masks


def run_e1(recs, out, rng):
    print()
    print("=" * 78)
    print("E1  error direction against the weak direction")
    print("=" * 78)
    variants = [("v_min(G), analytic rotation", "theta_vmin", "vmin_deg"),
                ("v_min(G) on warped sketch", "theta_vmin_wrot", "vmin_wrot_deg"),
                ("L1-cone minimiser", "theta_l1", "l1_weak_deg"),
                ("v_min(G_eff), inliers at argmin", "theta_eff", None),
                ("v_min(G_eff), (1-P)-weighted", "theta_eff_w", None)]
    out["e1"] = []
    masks = subset_masks(recs)
    for vname, key, dkey in variants:
        print("\n  variant: %s" % vname)
        print("  %-38s %4s %7s %7s %6s %8s %18s %8s %8s"
              % ("subset", "n", "mean", "median", "<45", "cos2", "95% CI",
                 "p_unif", "p_perm"))
        for sname, msk in masks:
            sel = [r for r, m in zip(recs, msk) if m and r.get(key) is not None]
            th = [r[key] for r in sel]
            st = align_stats(th, sname, rng)
            if dkey is not None:
                wd = [r[dkey] for r in sel]
            else:
                wd = [r["eff_est"]["vmin_deg" if key == "theta_eff" else "vmin_w_deg"]
                      for r in sel]
            pt = perm_test([r["err_dir_deg"] for r in sel], wd, rng)
            st.update(variant=vname, subset=sname, perm=pt)
            out["e1"].append(st)
            if st["n"] >= 3:
                print("  %-38s %4d %7.1f %7.1f %6.2f %+8.3f [%+.3f, %+.3f] %8.3f %8.3f"
                      % (sname, st["n"], st["mean_theta"], st["median_theta"],
                         st["frac_below_45"], st["mean_cos2"], st["cos2_ci95"][0],
                         st["cos2_ci95"][1], st["p_uniform_onesided"],
                         pt.get("p_perm_onesided", float("nan"))))
            else:
                print("  %-38s %4d   (too few)" % (sname, st["n"]))
    # q_G
    print("\n  G-quadratic form of the error direction, q = e^T G e / |e|^2 "
          "(0.5 under isotropy, lower = along weak direction)")
    out["e1_qG"] = []
    for sname, msk in masks[:7]:
        q = np.array([r["q_G"] for r, m in zip(recs, msk) if m])
        if len(q) < 3:
            continue
        boot = np.array([q[rng.integers(0, len(q), len(q))].mean() for _ in range(2000)])
        d = dict(subset=sname, n=int(len(q)), mean_q=float(q.mean()),
                 ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                 median_q=float(np.median(q)))
        out["e1_qG"].append(d)
        print("  %-38s n=%3d  mean q = %.3f  [%.3f, %.3f]  median %.3f"
              % (sname, d["n"], d["mean_q"], d["ci95"][0], d["ci95"][1], d["median_q"]))
    # marginals, to say whether the analytic uniform null is even appropriate
    ed = np.array([r["err_dir_deg"] for r in recs])
    vd = np.array([r["vmin_deg"] for r in recs])
    out["e1_marginals"] = dict(
        err_dir_hist_30deg=np.histogram(ed, bins=6, range=(0, 180))[0].tolist(),
        vmin_hist_30deg=np.histogram(vd, bins=6, range=(0, 180))[0].tolist(),
        err_dir_axis_locked_frac=float((np.minimum(ed % 90, 90 - ed % 90) < 5).mean()),
        vmin_vs_wrot_median_deg=float(np.median([r["vmin_vs_wrot_deg"] for r in recs])),
        vmin_vs_wrot_p90_deg=float(np.percentile([r["vmin_vs_wrot_deg"] for r in recs], 90)),
        vmin_vs_l1_median_deg=float(np.median([fold90(r["vmin_deg"], r["l1_weak_deg"]) for r in recs])))
    print("\n  marginals: error-direction histogram (30 deg bins, 0..180): %s"
          % out["e1_marginals"]["err_dir_hist_30deg"])
    print("             v_min histogram (30 deg bins):                  %s"
          % out["e1_marginals"]["vmin_hist_30deg"])
    print("             error directions within 5 deg of a grid axis:   %.2f"
          % out["e1_marginals"]["err_dir_axis_locked_frac"])
    print("             v_min analytic vs on-warp: median %.1f deg, p90 %.1f deg"
          % (out["e1_marginals"]["vmin_vs_wrot_median_deg"],
             out["e1_marginals"]["vmin_vs_wrot_p90_deg"]))
    print("             v_min vs L1-cone minimiser: median %.1f deg"
          % out["e1_marginals"]["vmin_vs_l1_median_deg"])


def run_e1b(recs, out, rng):
    print()
    print("=" * 78)
    print("E1b  cost-surface slopes around the minimum against the predicted profile")
    print("=" * 78)
    out["e1b"] = {}
    for key, label in (("surf_est", "at the argmin (GT-free)"),
                       ("surf_true_gt_diag", "at the TRUE cell (gt_diag)")):
        sel = [r for r in recs if r[key].get("valid")]
        ang = np.array([r[key]["angle_deg"] for r in sel])
        pr = np.array([r[key]["pearson"] for r in sel])
        pr = pr[np.isfinite(pr)]
        ea = np.array([r[key]["emp_aniso"] for r in sel])
        pa = np.array([r[key]["pred_aniso"] for r in sel])
        st = align_stats(ang, label, rng)
        pt = perm_test([r[key]["emp_weak_deg"] for r in sel],
                       [r[key]["pred_weak_deg"] for r in sel], rng)
        d = dict(label=label, n_valid=len(sel), n_total=len(recs),
                 weak_dir_angle=st, perm=pt,
                 profile_pearson_median=float(np.median(pr)) if pr.size else None,
                 profile_pearson_frac_pos=float((pr > 0).mean()) if pr.size else None,
                 profile_pearson_frac_above_0p5=float((pr > 0.5).mean()) if pr.size else None,
                 spearman_aniso=BG.spearman(ea, pa) if len(sel) > 3 else None,
                 emp_aniso_median=float(np.median(ea)) if ea.size else None,
                 pred_aniso_median=float(np.median(pa)) if pa.size else None)
        # split by whether the match was right (err <= 5 m), where t* ~ argmin
        for band, msk in (("err <= 5 m", lambda r: r["err_m"] <= 5),
                          ("err > 5 m", lambda r: r["err_m"] > 5)):
            s2 = [r for r in sel if msk(r)]
            if len(s2) >= 3:
                a2 = [r[key]["angle_deg"] for r in s2]
                p2 = np.array([r[key]["pearson"] for r in s2])
                d["band_" + band] = dict(align=align_stats(a2, band, rng),
                                         pearson_median=float(np.nanmedian(p2)))
        out["e1b"][key] = d
        print("  %s: %d/%d pairs with a full 2-cell neighbourhood" % (label, len(sel), len(recs)))
        if st.get("n", 0) >= 3:
            print("    empirical weak dir vs predicted weak dir: mean %.1f deg, median %.1f, "
                  "<45: %.2f, cos2 %+.3f [%+.3f, %+.3f], p_unif %.3f, p_perm %.3f"
                  % (st["mean_theta"], st["median_theta"], st["frac_below_45"],
                     st["mean_cos2"], st["cos2_ci95"][0], st["cos2_ci95"][1],
                     st["p_uniform_onesided"], pt["p_perm_onesided"]))
            print("    profile Pearson (12 directions): median %.3f, frac > 0: %.2f, frac > 0.5: %.2f"
                  % (d["profile_pearson_median"], d["profile_pearson_frac_pos"],
                     d["profile_pearson_frac_above_0p5"]))
            print("    anisotropy min/max slope: empirical median %.3f, predicted median %.3f, "
                  "Spearman %s"
                  % (d["emp_aniso_median"], d["pred_aniso_median"],
                     "%+.3f" % d["spearman_aniso"] if d["spearman_aniso"] is not None else "n/a"))
            for band in ("err <= 5 m", "err > 5 m"):
                b = d.get("band_" + band)
                if b:
                    print("    %-12s n=%3d  weak-dir cos2 %+.3f [%+.3f, %+.3f]  profile Pearson median %.3f"
                          % (band, b["align"]["n"], b["align"]["mean_cos2"],
                             b["align"]["cos2_ci95"][0], b["align"]["cos2_ci95"][1],
                             b["pearson_median"]))


def run_e2(recs, out, rng):
    print()
    print("=" * 78)
    print("E2  local geometry: spread and correlation with |error|")
    print("=" * 78)
    err = np.array([r["err_m"] for r in recs])
    scenes = np.array([r["scene"] for r in recs])
    cands = [("lambda_min, whole sketch (reference)", [r["lambda_min"] for r in recs]),
             ("lambda_min(G_eff), inliers at argmin", [r["eff_est"].get("lambda_min") for r in recs]),
             ("lambda_min(G_eff), (1-P)-weighted", [r["eff_est"].get("lambda_min_w") for r in recs]),
             ("inlier fraction at argmin", [r["eff_est"].get("frac_in") for r in recs]),
             ("map-side lambda_min in placed window (argmin)", [r["map_lambda_est"] for r in recs]),
             ("map-side lambda_min at TRUE window (gt_diag)", [r["map_lambda_true_gt_diag"] for r in recs]),
             ("min over 3x3 drone tiles", [r["tile_lambda_min"] for r in recs]),
             ("peak_ratio rho (reference)", [r["peak_ratio"] for r in recs])]
    out["e2"] = []
    print("  %-46s %4s %7s %7s %7s %7s %8s %18s  %s"
          % ("score", "n", "q10", "median", "q90", "IQR", "Spear", "95% CI", "per-scene Spearman"))
    for name, vals in cands:
        v = np.array([np.nan if x is None else x for x in vals], float)
        ok = np.isfinite(v)
        vv, ee = v[ok], err[ok]
        if len(vv) < 5:
            continue
        rho = BG.spearman(vv, ee)
        lo, hi = BG.boot_ci(vv, ee, n=BOOT)
        ps = {}
        for s in sorted(set(scenes)):
            m = ok & (scenes == s)
            if m.sum() >= 5:
                ps[s] = BG.spearman(v[m], err[m])
        # failure detection: AUROC for err > 30 m, low score = failure
        fail = ee > 30
        auroc = None
        if fail.any() and (~fail).any():
            rk = BG.rankdata(-vv)        # low score ranks high
            auroc = float((rk[fail].sum() - fail.sum() * (fail.sum() + 1) / 2.0)
                          / (fail.sum() * (~fail).sum()))
        d = dict(score=name, n=int(len(vv)), q10=float(np.percentile(vv, 10)),
                 median=float(np.median(vv)), q90=float(np.percentile(vv, 90)),
                 iqr=float(np.percentile(vv, 75) - np.percentile(vv, 25)),
                 spearman_err=rho, ci95=[lo, hi], per_scene=ps,
                 auroc_low_score_predicts_fail_over_30m=auroc)
        out["e2"].append(d)
        print("  %-46s %4d %7.3f %7.3f %7.3f %7.3f %+8.3f [%+.3f, %+.3f]  %s   AUROC(fail) %s"
              % (name, d["n"], d["q10"], d["median"], d["q90"], d["iqr"], rho, lo, hi,
                 " ".join("%s %+.2f" % (k[:6], x) for k, x in ps.items()),
                 "%.3f" % auroc if auroc is not None else "n/a"))


def loso_matched(recs, score_key_fn, levels=(0.10, 0.20, 0.30, 0.50)):
    """rho alone vs 'both above a common quantile', thresholds fitted on the
    other three scenes. Same construction as boundary_geometry.test_b_loso."""
    scenes = sorted({r["scene"] for r in recs})
    pr = np.array([r["peak_ratio"] for r in recs], float)
    s2 = np.array([score_key_fn(r) for r in recs], float)
    err = np.array([r["err_m"] for r in recs])
    sc = np.array([r["scene"] for r in recs])
    ok = np.isfinite(s2) & np.isfinite(pr)
    n = int(ok.sum())
    qgrid = np.arange(0.0, 0.999, 0.002)
    rows = []
    for c in levels:
        for rule in ("rho only", "rho AND score"):
            keep = np.zeros(len(recs), bool)
            for s in scenes:
                te = ok & (sc == s)
                tr = ok & (sc != s)
                if rule == "rho only":
                    t = np.quantile(pr[tr], 1.0 - c)
                    keep[te] = pr[te] >= t
                else:
                    best, bq = None, None
                    for q in qgrid:
                        tp, tl = np.quantile(pr[tr], q), np.quantile(s2[tr], q)
                        cov = float(((pr[tr] >= tp) & (s2[tr] >= tl)).mean())
                        d = abs(cov - c)
                        if best is None or d < best:
                            best, bq = d, q
                    tp, tl = np.quantile(pr[tr], bq), np.quantile(s2[tr], bq)
                    keep[te] = (pr[te] >= tp) & (s2[te] >= tl)
            e = err[keep]
            rows.append(dict(coverage=c, rule=rule, admitted=int(keep.sum()), n=n,
                             median_err_m=float(np.median(e)) if len(e) else None,
                             n_over_30m=int((e > 30).sum())))
    return rows


def run_e3(recs, out):
    """Gate: run only if E2 gave a local-geometry score whose Spearman CI with
    error excludes zero on the negative side (low lambda, high error), or E1
    gave an alignment CI excluding zero on the positive side."""
    e1_signal = [s for s in out["e1"] if s.get("n", 0) >= 10
                 and s["cos2_ci95"][0] > 0 and s["perm"].get("p_perm_onesided", 1) < 0.05]
    e2_signal = [d for d in out["e2"] if "reference" not in d["score"]
                 and "gt_diag" not in d["score"] and d["ci95"][1] < 0]
    out["e3"] = dict(triggered=bool(e1_signal or e2_signal),
                     e1_signal=[(s["variant"], s["subset"]) for s in e1_signal],
                     e2_signal=[d["score"] for d in e2_signal], rules=[])
    print()
    print("=" * 78)
    print("E3  rejection rule  (triggered: %s)" % out["e3"]["triggered"])
    print("=" * 78)
    if e1_signal:
        print("  E1 signal in:", out["e3"]["e1_signal"])
    if e2_signal:
        print("  E2 signal in:", out["e3"]["e2_signal"])
    if not out["e3"]["triggered"]:
        print("  no CI excluded the null in E1 or E2; E3 not run, by the pre-registered gate")
        return
    fns = {"lambda_min(G_eff), inliers at argmin": lambda r: r["eff_est"].get("lambda_min", np.nan),
           "lambda_min(G_eff), (1-P)-weighted": lambda r: r["eff_est"].get("lambda_min_w", np.nan),
           "inlier fraction at argmin": lambda r: r["eff_est"].get("frac_in", np.nan),
           "map-side lambda_min in placed window (argmin)": lambda r: (r["map_lambda_est"] if r["map_lambda_est"] is not None else np.nan),
           "min over 3x3 drone tiles": lambda r: (r["tile_lambda_min"] if r["tile_lambda_min"] is not None else np.nan)}
    names = [d["score"] for d in e2_signal] or list(fns)
    for name in names:
        if name not in fns:
            continue
        rows = loso_matched(recs, fns[name])
        out["e3"]["rules"].append(dict(score=name, loso=rows))
        print("  score: %s  (LOSO, 4 scenes)" % name)
        print("  %-8s %-16s %10s %10s %8s" % ("cover", "rule", "admitted", "median m", ">30 m"))
        for r in rows:
            print("  %-8s %-16s %10s %10s %8s"
                  % ("%d%%" % round(100 * r["coverage"]), r["rule"],
                     "%d/%d" % (r["admitted"], r["n"]),
                     "%.1f" % r["median_err_m"] if r["median_err_m"] is not None else "n/a",
                     "%d/%d" % (r["n_over_30m"], r["admitted"])))


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def make_figures(recs, examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    err = np.array([r["err_m"] for r in recs])
    th = np.array([r["theta_vmin"] for r in recs])
    lam = np.array([r["lambda_min"] for r in recs])
    # ---- fig 1: theta distribution and error components in the G eigenframe
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for msk, lab, col in ((np.ones(len(recs), bool), "all (n=%d)" % len(recs), "#4c72b0"),
                          (err > 5, "err > 5 m (n=%d)" % int((err > 5).sum()), "#dd8452")):
        ax[0].hist(th[msk], bins=6, range=(0, 90), histtype="step", lw=2, label=lab, color=col)
    ax[0].axhline(len(recs) / 6.0, ls="--", color="grey", lw=1, label="uniform, all")
    ax[0].set_xlabel("angle between error vector and v_min (deg, folded)")
    ax[0].set_ylabel("pairs")
    ax[0].set_title("E1: theta distribution")
    ax[0].legend(fontsize=8)
    # components
    along, across = [], []
    for r in recs:
        G = np.array(r["G"])
        _, _, vmin, vmax = eig2(G)
        e = np.array([r["dc_m"], r["dr_m"]])
        along.append(abs(e @ vmin))
        across.append(abs(e @ vmax))
    along, across = np.array(along), np.array(across)
    sc = ax[1].scatter(across, along, c=lam, cmap="viridis", s=22, vmin=0, vmax=0.5)
    lim = max(along.max(), across.max()) * 1.05
    ax[1].plot([0, lim], [0, lim], "k--", lw=1)
    ax[1].set_xlim(0, lim)
    ax[1].set_ylim(0, lim)
    ax[1].set_xlabel("|error| across v_min, along v_max (m)")
    ax[1].set_ylabel("|error| along v_min (m)")
    ax[1].set_title("E1: error components, colour = lambda_min")
    plt.colorbar(sc, ax=ax[1], label="lambda_min(G)")
    # theta vs lambda_min
    ax[2].scatter(lam, th, c=np.log10(np.maximum(err, 0.3)), cmap="magma", s=22)
    ax[2].axhline(45, ls="--", color="grey", lw=1)
    ax[2].set_xlabel("lambda_min(G)")
    ax[2].set_ylabel("theta (deg)")
    ax[2].set_title("E1: theta vs anisotropy, colour = log10 err")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_e1_alignment.png"), dpi=130)
    plt.close(fig)

    # ---- fig 2: example pairs
    n = len(examples)
    fig, ax = plt.subplots(n, 2, figsize=(10, 4.6 * n))
    if n == 1:
        ax = ax[None, :]
    for i, (rec, cost, Wrot, G) in enumerate(examples):
        a = ax[i, 0]
        a.imshow(Wrot, cmap="gray_r")
        h, w = Wrot.shape
        lmin, lmax, vmin, vmax = eig2(G)
        ang = math.degrees(math.atan2(vmax[1], vmax[0]))
        s = 0.35 * min(h, w)
        a.add_patch(Ellipse((w / 2, h / 2), 2 * s * math.sqrt(lmax / 0.5),
                            2 * s * math.sqrt(lmin / 0.5), angle=ang,
                            fill=False, color="tab:red", lw=2))
        L = 0.4 * min(h, w)
        a.plot([w / 2 - L * vmin[0], w / 2 + L * vmin[0]],
               [h / 2 - L * vmin[1], h / 2 + L * vmin[1]], color="tab:blue", lw=2,
               label="v_min (weak)")
        a.set_title("%s\nlambda_min=%.3f  err=%.1f m  theta=%.0f deg"
                    % (rec["pid"], rec["lambda_min"], rec["err_m"], rec["theta_vmin"]),
                    fontsize=9)
        a.legend(fontsize=8, loc="lower right")
        a.set_xticks([])
        a.set_yticks([])
        b = ax[i, 1]
        b.imshow(cost, cmap="viridis")
        b.plot(rec["true_c"], rec["true_r"], "w+", ms=12, mew=2, label="true")
        b.plot(rec["c_est"], rec["r_est"], "rx", ms=10, mew=2, label="argmin")
        b.annotate("", xy=(rec["c_est"], rec["r_est"]), xytext=(rec["true_c"], rec["true_r"]),
                   arrowprops=dict(arrowstyle="->", color="w", lw=1.5))
        L = 12
        b.plot([rec["c_est"] - L * vmin[0], rec["c_est"] + L * vmin[0]],
               [rec["r_est"] - L * vmin[1], rec["r_est"] + L * vmin[1]],
               color="tab:blue", lw=2, label="v_min at argmin")
        b.set_title("cost grid (1.6 m cells), rho=%.3f" % (rec["peak_ratio"] or float("nan")),
                    fontsize=9)
        b.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_examples.png"), dpi=110)
    plt.close(fig)

    # ---- fig 3: E2 spread and E1b
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    le = np.array([r["eff_est"].get("lambda_min", np.nan) for r in recs])
    lm = np.array([np.nan if r["map_lambda_est"] is None else r["map_lambda_est"] for r in recs])
    ax[0].hist([lam, le[np.isfinite(le)], lm[np.isfinite(lm)]], bins=20, range=(0, 0.5),
               label=["whole sketch", "G_eff inliers @argmin", "map window @argmin"],
               histtype="step", lw=2)
    ax[0].set_xlabel("lambda_min")
    ax[0].set_title("E2: dynamic range of local vs whole-sketch lambda_min")
    ax[0].legend(fontsize=8)
    ok = np.isfinite(le)
    ax[1].scatter(le[ok], err[ok], s=18)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("lambda_min(G_eff) at argmin")
    ax[1].set_ylabel("|error| (m)")
    ax[1].set_title("E2: G_eff vs error")
    sv = [r for r in recs if r["surf_est"].get("valid")]
    if sv:
        ax[2].hist([r["surf_est"]["angle_deg"] for r in sv], bins=6, range=(0, 90),
                   histtype="step", lw=2, color="#55a868")
        ax[2].axhline(len(sv) / 6.0, ls="--", color="grey", lw=1)
    ax[2].set_xlabel("angle: empirical weak slope dir vs predicted (deg)")
    ax[2].set_title("E1b: cost-surface weak direction vs theory (n=%d)" % len(sv))
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_e2_e1b.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    K.assert_modules()
    B.crop_shared = uniform_crop          # the protocol the sealed cell used
    CC = B.TD.C
    print("pool file in force:", K.use_local_pool_if_needed(), flush=True)
    repair_pool_paths(CC.pool())
    patch_load_ortho(CC)
    pids = K.pool_pids()
    assert len(pids) == 114, len(pids)
    if os.environ.get("BENCH_N"):
        pids = pids[:int(os.environ["BENCH_N"])]
    matrix = {}
    for line in io.open(MATRIX, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            matrix[r["pid"]] = r
    missing = [p for p in pids if p not in matrix]
    assert not missing, missing[:5]

    rng = np.random.default_rng(SEED)
    # Per-pair records are cached as they complete, because a native
    # segmentation fault (cv2/torch on this box) killed two full runs partway;
    # the shell loop in RUN.sh reruns until the cache holds all 114 pids.
    cache = os.path.join(HERE, "pairs_cache.jsonl")
    cached = {}
    if os.path.exists(cache):
        for line in io.open(cache, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                cached[r["pid"]] = r
    print("cache: %d pairs already done" % len(cached), flush=True)
    recs, examples = [], []
    fails = []
    cf = io.open(cache, "a", encoding="utf-8")
    for i, pid in enumerate(pids, 1):
        t1 = time.time()
        if pid in cached:
            recs.append(cached[pid])
            continue
        try:
            rec, cost, Wrot, G = analyse_pair(pid, matrix[pid])
        except Exception as exc:
            fails.append((pid, repr(exc)))
            print("%3d/%d %-36s FAIL %r" % (i, len(pids), pid, exc), flush=True)
            continue
        recs.append(rec)
        cf.write(json.dumps(rec, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
        cf.write("\n")
        cf.flush()
        print("%3d/%d %-36s err %6.1f (sealed %6.1f) %s  rho %.4f (sealed %.4f) %s  "
              "lam %.3f  theta %4.1f  %.1fs"
              % (i, len(pids), pid, rec["err_m"], rec["shipped_err_m"],
                 "OK " if rec["err_match"] else "BAD",
                 rec["peak_ratio"] or float("nan"), rec["shipped_peak_ratio"] or float("nan"),
                 "OK " if rec["ratio_match"] else "BAD",
                 rec["lambda_min"], rec["theta_vmin"], time.time() - t1), flush=True)
    cf.close()
    if os.environ.get("BENCH_N"):
        pass
    elif len(recs) != len(pids):
        sys.exit("INCOMPLETE: %d of %d pairs in cache; rerun to resume" % (len(recs), len(pids)))

    n_ok = sum(r["err_match"] for r in recs)
    n_rok = sum(r["ratio_match"] for r in recs)
    print()
    print("REPRODUCTION  %d/%d pairs analysed, %d failed" % (len(recs), len(pids), len(fails)))
    print("  |error| reproduces the sealed shipped_err_m to 0.1 m: %d/%d" % (n_ok, len(recs)))
    print("  peak_ratio reproduces the sealed value to 4 dp:      %d/%d" % (n_rok, len(recs)))
    dr = [abs(r["peak_ratio"] - r["shipped_peak_ratio"]) for r in recs
          if r["peak_ratio"] is not None and r["shipped_peak_ratio"] is not None]
    print("  max |peak_ratio - sealed| = %.2e" % (max(dr) if dr else float("nan")))
    bad = [r for r in recs if not r["err_match"]]
    for r in bad[:10]:
        print("    mismatch %-36s recomputed %.1f sealed %.1f" % (r["pid"], r["err_m"], r["shipped_err_m"]))
    out = dict(generated="2026-09-15", script=os.path.basename(__file__),
               source_matrix=os.path.relpath(MATRIX, PROJ).replace("\\", "/"),
               n_pairs=len(recs), n_failed=len(fails), fails=fails,
               reproduction=dict(err_match=n_ok, ratio_match=n_rok, n=len(recs)),
               method=dict(normals="structure tensor (boundary_geometry.py, sigma_d=%g, sigma_t=%g)"
                           % (BG.SIGMA_D, BG.SIGMA_T),
                           frame="cost grid: x=column, y=row; G_grid = R G R^T",
                           slope_dirs_deg=np.degrees(SLOPE_DIRS).tolist(),
                           slope_radii_cells=list(SLOPE_RADII), cell_m=TCF * 0.2))

    err = np.array([r["err_m"] for r in recs])
    print("  error bands: err<=5 m %d, 5<err<=30 m %d, err>30 m %d"
          % (int((err <= 5).sum()), int(((err > 5) & (err <= 30)).sum()), int((err > 30).sum())))

    run_e1(recs, out, rng)
    run_e1b(recs, out, rng)
    run_e2(recs, out, rng)
    run_e3(recs, out)

    out["pairs"] = recs
    p = os.path.join(HERE, "directional_geometry.json")
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    print("\nwrote %s" % p)
    try:
        # examples for the figure: low-lambda pairs with a large error first,
        # then high-lambda ones; recomputed here so a resumed run has them
        cand = sorted([r for r in recs if r["err_m"] > 5], key=lambda r: r["lambda_min"])
        ex_pids = [r["pid"] for r in cand[:4]] + [r["pid"] for r in cand[-2:]]
        for pid in ex_pids:
            rec, cost, Wrot, G = analyse_pair(pid, matrix[pid])
            examples.append((rec, cost, Wrot, G))
        make_figures(recs, examples)
        print("wrote fig_e1_alignment.png, fig_examples.png, fig_e2_e1b.png")
    except Exception as exc:
        print("FIGURES FAILED: %r" % (exc,))
    print("total %.1f min" % ((time.time() - t0) / 60.0))


if __name__ == "__main__":
    main()
