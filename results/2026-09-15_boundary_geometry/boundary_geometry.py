"""lambda_min(G): a boundary-geometry degeneracy certificate for the drone sketch.

WHAT THIS COMPUTES
------------------
report/Theory/theory-main.tex proves that near an ideal translation t* the capped
chamfer cost of a sketch against a map behaves as

    C(t* + d) = sum_i alpha_i |n_i^T d| + O(|d|^2),    G = sum_i alpha_i n_i n_i^T

with n_i the unit normal to the boundary at drone sketch sample q_i and alpha_i
positive weights summing to one. G is 2x2 positive semi-definite. lambda_min(G)
is a degeneracy certificate: near zero means the boundaries are mostly parallel
and one translation direction is weakly constrained, near 0.5 means orientations
are diverse and both directions are constrained.

No earlier run here computed lambda_min(G) before this script. Two predictions are
tested against it here, on the 114-pair RTK-uniform UAVScenes population:

  Test A (contribution 2)  lambda_min(G) tracks localisation error.
  Test B (contribution 1)  lambda_min(G) combined with the basin ratio rho
                           (peak_ratio) rejects unreliable estimates better than
                           rho alone, at matched coverage.

METHOD
------
Normals come from the STRUCTURE TENSOR of the edge map, not from the gradient of
its distance transform. The distance transform is at a minimum on the edge set
itself, so its gradient is degenerate exactly at the points the theory samples
(the sketch points q_i are the edge pixels). Recovering a normal from it needs
the gradient read off a dilated ring beside the boundary, which introduces an
offset scale and breaks down wherever two boundaries run close together. The
structure tensor is evaluated at the sample point itself and needs no such
choice, so it is used throughout.

Per edge pixel:
    E        binary edge map (v08_drone_edges), smoothed by a Gaussian sigma_d
    g        its gradient (Sobel), which for a thin boundary points ACROSS it
    T        Gaussian-windowed (sigma_t) outer product of g with itself
    n        principal eigenvector of T, i.e. the estimated unit normal

G is then the EQUAL-WEIGHT mean of n n^T over the m edge pixels. Equal weights
are the paper's alpha_i normalised to sum to one under the assumption that every
sampled boundary point carries the same weight, which is what the shipped
matcher does: run_classical.py correlates the rotated binary edge mask itself, so
every edge pixel enters the cost with the same coefficient.

Because every n_i is a unit vector and the weights sum to one, trace(G) = 1
exactly. That is asserted per pair. It has a consequence worth stating: lambda_max
= 1 - lambda_min and ratio = lambda_min / (1 - lambda_min), so lambda_max and the
ratio carry no information beyond lambda_min, and 0.5 is a hard ceiling on
lambda_min reached only at perfect isotropy.

ASSUMPTIONS AND THEIR KNOWN FAILURE MODES
-----------------------------------------
1. Every edge pixel is a valid boundary sample. The theory assumes each q_i lies
   on a smooth boundary away from endpoints, junctions and competing nearby
   boundaries. Real sketches violate this at corners, junctions and texture
   clumps, where T is near isotropic and its principal eigenvector is close to
   arbitrary while still contributing a full rank-one term. Clutter can therefore
   raise lambda_min. A robustness variant accumulates the trace-normalised tensor
   T_i / trace(T_i) instead of n_i n_i^T; that degrades to I/2 at an isotropic
   point rather than to a random direction. Both are reported. Per-pair coherence
   statistics say how much of each sketch is in this regime.
2. Full resolution, unmasked. run_classical.py reads v08_drone_edges raw, with
   neither drone_mask nor suppress_drone applied, so the unmasked full-resolution
   edge set is the matcher-faithful one. The matcher then rotates it by Mr and
   mean-pools it by TRAIN_COARSE_F = 8. Rotation maps G to R G R^T and leaves the
   eigenvalues unchanged, so working in the unrotated drone frame is exact.
   Pooling is not modelled here; lambda_min is a property of the sketch, computed
   at the resolution the sketch is drawn at.
3. The map side is not used at all, by construction. G depends only on the drone
   sketch, which is what makes it available before any match is attempted.
4. G says nothing about global uniqueness. A repeated boundary pattern can give
   several equally sharp minima with a well conditioned G at each. That is what
   the basin ratio rho measures, and it is why Test B pairs the two.

SANITY CHECKS
-------------
Four synthetic images, reported in full rather than tuned:
  parallel lines       expected lambda_min near 0
  perpendicular cross  expected lambda_min near 0.5
  square               expected lambda_min near 0.5
  circle               expected lambda_min near 0.5, corner free, exactly
                       isotropic, so it isolates the upper end without the
                       junction contamination a square carries

GROUND TRUTH
------------
No ground truth enters any computed statistic. Errors (shipped_err_m) are read
from the already-sealed matrix rows and used only for scoring, and every
threshold reported as an operating point is chosen leave-one-scene-out.

    CUDA_VISIBLE_DEVICES=-1 python results/2026-09-15_boundary_geometry/boundary_geometry.py

Writes lambda_geometry.json (per pair) and prints the full report.
"""
import io
import json
import os
import sys

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.normpath(os.path.join(HERE, "..", ".."))
RTK114 = os.path.join(PROJ, "results", "2026-09-14_rtk114_indomain_table")
MATRIX = os.path.join(PROJ, "results", "2026-09-15_rerun_matrix",
                      "matrix_canny_uavscenes_shipped_cc1_gate.jsonl")

import cv2  # noqa: E402

SIGMA_D = 1.0          # gradient pre-smoothing, px (0.20 m/px pairs)
SIGMA_T = 2.0          # tensor integration window, px
SIGMA_SWEEP = [1.0, 2.0, 4.0, 8.0]
COH_LOW = 0.5          # coherence below this counts as "near isotropic"
EPS = 1e-12
BOOT = 10000
SEED = 20260915


# ----------------------------------------------------------------------
# the certificate
# ----------------------------------------------------------------------
def structure_tensor_fields(edges, sigma_d=SIGMA_D, sigma_t=SIGMA_T):
    """Jxx, Jxy, Jyy of the windowed structure tensor, over the whole image."""
    e = np.asarray(edges).astype(np.float32)
    es = cv2.GaussianBlur(e, (0, 0), sigma_d, borderType=cv2.BORDER_REPLICATE)
    gx = cv2.Sobel(es, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(es, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma_t, borderType=cv2.BORDER_REPLICATE)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma_t, borderType=cv2.BORDER_REPLICATE)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma_t, borderType=cv2.BORDER_REPLICATE)
    return jxx, jxy, jyy


def gram_from_edges(edges, sigma_d=SIGMA_D, sigma_t=SIGMA_T):
    """G and its companions for one binary edge map.

    Returns a dict. `G` is the literal equal-weight mean of n_i n_i^T over the
    edge pixels; `G_tn` is the robustness variant that accumulates the
    trace-normalised tensor instead.
    """
    e = np.asarray(edges).astype(bool)
    jxx, jxy, jyy = structure_tensor_fields(e, sigma_d, sigma_t)
    ys, xs = np.nonzero(e)
    a, b, c = jxx[ys, xs], jxy[ys, xs], jyy[ys, xs]
    tr = a + c
    keep = tr > EPS                      # pixels with no local gradient at all
    n_dead = int((~keep).sum())
    a, b, c, tr = a[keep], b[keep], c[keep], tr[keep]
    m = int(a.size)
    if m == 0:
        return None

    # principal eigenvector of [[a,b],[b,c]] as a doubled angle. For a unit
    # vector n at angle th, n n^T = 0.5 * [[1+cos2th, sin2th],[sin2th, 1-cos2th]],
    # so the mean of n n^T is fixed by the means of cos2th and sin2th.
    two_th = np.arctan2(2.0 * b, a - c).astype(np.float64)
    cos2, sin2 = np.cos(two_th), np.sin(two_th)
    mc, ms = float(cos2.mean()), float(sin2.mean())
    g = 0.5 * np.array([[1.0 + mc, ms], [ms, 1.0 - mc]], dtype=np.float64)

    # trace-normalised variant: T/tr has trace 1 and equals I/2 when isotropic
    a64, b64, c64, tr64 = (v.astype(np.float64) for v in (a, b, c, tr))
    g_tn = np.array([[float((a64 / tr64).mean()), float((b64 / tr64).mean())],
                     [float((b64 / tr64).mean()), float((c64 / tr64).mean())]],
                    dtype=np.float64)

    # coherence (l1-l2)/(l1+l2) of the structure tensor, per edge pixel
    disc = np.sqrt(np.maximum((a - c) ** 2 + 4.0 * b * b, 0.0))
    coh = disc / np.maximum(tr, EPS)

    # eigenvalues of a PSD matrix can come back at -1e-17 on a degenerate
    # sketch; clamp that float noise at zero rather than report a negative
    # eigenvalue for a positive semi-definite matrix.
    ev = np.clip(np.linalg.eigvalsh(g), 0.0, None)
    ev_tn = np.clip(np.linalg.eigvalsh(g_tn), 0.0, None)
    return dict(
        m=m, n_dead=n_dead,
        G=g.tolist(),
        lambda_min=float(ev[0]), lambda_max=float(ev[1]),
        ratio=float(ev[0] / max(ev[1], EPS)),
        trace=float(np.trace(g)),
        lambda_min_tracenorm=float(ev_tn[0]),
        coh_mean=float(coh.mean()),
        coh_frac_low=float((coh < COH_LOW).mean()),
    )


# ----------------------------------------------------------------------
# sanity synthetics
# ----------------------------------------------------------------------
def synth_parallel(n=256, spacing=20):
    im = np.zeros((n, n), np.uint8)
    for x in range(spacing, n - spacing + 1, spacing):
        im[20:n - 20, x] = 1
    return im.astype(bool)


def synth_cross(n=256):
    im = np.zeros((n, n), np.uint8)
    im[n // 2, 20:n - 20] = 1
    im[20:n - 20, n // 2] = 1
    return im.astype(bool)


def synth_square(n=256, side=120):
    im = np.zeros((n, n), np.uint8)
    a, b = (n - side) // 2, (n + side) // 2
    im[a, a:b] = 1
    im[b, a:b] = 1
    im[a:b, a] = 1
    im[a:b + 1, b] = 1
    return im.astype(bool)


def synth_circle(n=256, r=80):
    im = np.zeros((n, n), np.uint8)
    cv2.circle(im, (n // 2, n // 2), r, 1, 1)
    return im.astype(bool)


def run_sanity():
    cases = [("parallel lines (expect ~0)", synth_parallel()),
             ("perpendicular cross (expect ~0.5)", synth_cross()),
             ("square (expect ~0.5)", synth_square()),
             ("circle (expect ~0.5)", synth_circle())]
    out = []
    print("SANITY CHECKS  (sigma_d=%.1f, sigma_t=%.1f)" % (SIGMA_D, SIGMA_T))
    print("  %-34s %8s %10s %10s %9s %9s %9s"
          % ("case", "m", "lam_min", "lam_max", "ratio", "trace", "lam_min_tn"))
    for name, im in cases:
        r = gram_from_edges(im)
        assert abs(r["trace"] - 1.0) < 1e-9, (name, r["trace"])
        print("  %-34s %8d %10.4f %10.4f %9.4f %9.6f %9.4f"
              % (name, r["m"], r["lambda_min"], r["lambda_max"], r["ratio"],
                 r["trace"], r["lambda_min_tracenorm"]))
        out.append(dict(case=name, **{k: r[k] for k in
                                      ("m", "lambda_min", "lambda_max", "ratio",
                                       "trace", "lambda_min_tracenorm",
                                       "coh_mean", "coh_frac_low")}))
    return out


# ----------------------------------------------------------------------
# statistics, written out rather than imported so the bootstrap and the
# partial correlation use exactly the same ranking
# ----------------------------------------------------------------------
def rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    # average ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def partial_spearman(x, y, z):
    """Spearman between x and y with z held, by residualising the ranks."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rz1 = np.column_stack([rz, np.ones_like(rz)])
    bx = np.linalg.lstsq(rz1, rx, rcond=None)[0]
    by = np.linalg.lstsq(rz1, ry, rcond=None)[0]
    return pearson(rx - rz1 @ bx, ry - rz1 @ by)


def boot_ci(x, y, fn=spearman, n=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = len(x)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, k, k)
        vals[i] = fn(x[idx], y[idx])
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def med(v):
    return float(np.median(v)) if len(v) else float("nan")


# ----------------------------------------------------------------------
def load_rows():
    rows = [json.loads(l) for l in io.open(MATRIX, encoding="utf-8") if l.strip()]
    return {r["pid"]: r for r in rows}


def compute_all():
    sys.path.insert(0, RTK114)
    import rtk114_common as K
    K.assert_modules()
    print("pool file in force:", K.use_local_pool_if_needed(), flush=True)
    pids = K.pool_pids()
    matrix = load_rows()
    missing = [p for p in pids if p not in matrix]
    if missing:
        sys.exit("FATAL: %d pool pids absent from the matrix file: %s"
                 % (len(missing), missing[:5]))

    recs = []
    for i, pid in enumerate(pids):
        pre = dict(K.C.load_prepack(pid))
        edges = np.asarray(pre["v08_drone_edges"]).astype(bool)
        r = gram_from_edges(edges)
        if r is None:
            sys.exit("FATAL: empty drone sketch for %s" % pid)
        assert abs(r["trace"] - 1.0) < 1e-9, (pid, r["trace"])
        e = K.C.pool()[pid]
        mrow = matrix[pid]
        rec = dict(pid=pid, scene=pid.split("_")[0], terrain=e.get("terrain"),
                   flight=e.get("flight"), render=e.get("render"),
                   shape=[int(edges.shape[0]), int(edges.shape[1])],
                   edge_density=float(edges.mean()),
                   shipped_err_m=mrow.get("shipped_err_m"),
                   center_hit_err_m=mrow.get("center_hit_err_m"),
                   peak_ratio=mrow.get("peak_ratio"),
                   answered=bool(mrow.get("answered")),
                   **{k: r[k] for k in ("m", "n_dead", "lambda_min", "lambda_max",
                                        "ratio", "trace", "lambda_min_tracenorm",
                                        "coh_mean", "coh_frac_low")})
        for s in SIGMA_SWEEP:
            if s == SIGMA_T:
                rec["lambda_min_sigma_%g" % s] = r["lambda_min"]
            else:
                rs = gram_from_edges(edges, SIGMA_D, s)
                rec["lambda_min_sigma_%g" % s] = rs["lambda_min"]
        recs.append(rec)
        if (i + 1) % 20 == 0:
            print("  %d/%d" % (i + 1, len(pids)), flush=True)
    return recs


# ----------------------------------------------------------------------
# Test A
# ----------------------------------------------------------------------
def test_a(recs):
    lam = np.array([r["lambda_min"] for r in recs])
    err = np.array([r["shipped_err_m"] for r in recs])
    m = np.array([r["m"] for r in recs], float)
    lam_tn = np.array([r["lambda_min_tracenorm"] for r in recs])
    out = {}

    print()
    print("=" * 78)
    print("TEST A  lambda_min(G) against localisation error, n=%d" % len(recs))
    print("=" * 78)
    rho = spearman(lam, err)
    lo, hi = boot_ci(lam, err)
    out["spearman_lambda_err"] = rho
    out["spearman_lambda_err_ci95"] = [lo, hi]
    print("  Spearman(lambda_min, shipped_err_m) = %+.3f   95%% bootstrap [%+.3f, %+.3f]"
          % (rho, lo, hi))
    print("    (pair-level bootstrap, 4 scenes, so scene clustering makes this "
          "interval optimistic)")

    r_me = spearman(m, err)
    r_lm = spearman(lam, m)
    out["spearman_m_err"] = r_me
    out["spearman_lambda_m"] = r_lm
    print("  Spearman(m, shipped_err_m)          = %+.3f" % r_me)
    print("  Spearman(lambda_min, m)             = %+.3f" % r_lm)
    pr = partial_spearman(lam, err, m)
    lo2, hi2 = boot_ci(np.column_stack([lam, m]), err,
                       fn=lambda A, y: partial_spearman(A[:, 0], y, A[:, 1]))
    out["partial_spearman_lambda_err_given_m"] = pr
    out["partial_spearman_ci95"] = [lo2, hi2]
    print("  partial Spearman(lambda_min, err | m) = %+.3f  95%% [%+.3f, %+.3f]"
          % (pr, lo2, hi2))

    print()
    print("  within m-terciles (density held):")
    om = np.argsort(m)
    out["m_tercile"] = []
    for name, idx in zip(("low m", "mid m", "high m"), np.array_split(om, 3)):
        rr = spearman(lam[idx], err[idx])
        print("    %-7s n=%3d  m %6d..%6d  Spearman(lambda_min, err) = %+.3f"
              % (name, len(idx), int(m[idx].min()), int(m[idx].max()), rr))
        out["m_tercile"].append(dict(band=name, n=int(len(idx)), spearman=rr))

    print()
    print("  within scene:")
    out["per_scene"] = []
    for sc in sorted({r["scene"] for r in recs}):
        idx = np.array([i for i, r in enumerate(recs) if r["scene"] == sc])
        rr = spearman(lam[idx], err[idx])
        rr_m = spearman(m[idx], err[idx])
        print("    %-11s n=%3d  Spearman(lambda_min, err) = %+.3f   "
              "median lambda_min = %.3f   median err = %6.1f m   "
              "Spearman(m, err) = %+.3f"
              % (sc, len(idx), rr, med(lam[idx]), med(err[idx]), rr_m))
        out["per_scene"].append(dict(scene=sc, n=int(len(idx)), spearman=rr,
                                     spearman_m_err=rr_m,
                                     median_lambda_min=med(lam[idx]),
                                     median_err_m=med(err[idx])))

    print()
    print("  lambda_min terciles:")
    ol = np.argsort(lam)
    out["lambda_tercile"] = []
    for name, idx in zip(("lowest", "middle", "highest"), np.array_split(ol, 3)):
        print("    %-8s n=%3d  lambda_min %.3f..%.3f  median err = %6.1f m  "
              ">30 m: %2d/%d"
              % (name, len(idx), lam[idx].min(), lam[idx].max(),
                 med(err[idx]), int((err[idx] > 30).sum()), len(idx)))
        out["lambda_tercile"].append(
            dict(band=name, n=int(len(idx)), lo=float(lam[idx].min()),
                 hi=float(lam[idx].max()), median_err_m=med(err[idx]),
                 n_over_30m=int((err[idx] > 30).sum())))

    print()
    print("  robustness:")
    rr = spearman(lam_tn, err)
    print("    trace-normalised variant  Spearman = %+.3f "
          "(vs %+.3f for the hard eigenvector)" % (rr, rho))
    out["spearman_tracenorm_err"] = rr
    out["sigma_sweep"] = []
    for s in SIGMA_SWEEP:
        v = np.array([r["lambda_min_sigma_%g" % s] for r in recs])
        rr = spearman(v, err)
        print("    sigma_t = %-4g  median lambda_min = %.3f  Spearman = %+.3f"
              % (s, med(v), rr))
        out["sigma_sweep"].append(dict(sigma_t=s, median_lambda_min=med(v),
                                       spearman=rr))
    coh = np.array([r["coh_frac_low"] for r in recs])
    print("    near-isotropic edge-pixel fraction (coherence < %.1f): "
          "median %.3f, range %.3f..%.3f" % (COH_LOW, med(coh), coh.min(), coh.max()))
    print("    Spearman(coh_frac_low, err) = %+.3f" % spearman(coh, err))
    out["spearman_cohfraclow_err"] = spearman(coh, err)
    # mechanism note, not a finding: clutter raises lambda_min (isotropic
    # junction pixels still contribute a full rank-one term) and also goes with
    # higher error, two effects of opposite sign that can pin the lambda-error
    # association near zero. Holding the clutter fraction says whether that is
    # what is happening.
    out["spearman_lambda_cohfraclow"] = spearman(lam, coh)
    out["partial_spearman_lambda_err_given_coh"] = partial_spearman(lam, err, coh)
    print("    Spearman(lambda_min, coh_frac_low) = %+.3f, "
          "partial Spearman(lambda_min, err | coh_frac_low) = %+.3f"
          % (out["spearman_lambda_cohfraclow"],
             out["partial_spearman_lambda_err_given_coh"]))
    # saturation: how much dynamic range lambda_min actually has here
    out["saturation"] = dict(
        iqr=float(np.percentile(lam, 75) - np.percentile(lam, 25)),
        n_above_0p40=int((lam > 0.40).sum()), n_below_0p30=int((lam < 0.30).sum()),
        ceiling=0.5)
    print("    dynamic range: IQR %.3f on a [0, 0.5] scale, %d of %d pairs above "
          "0.40, %d below 0.30"
          % (out["saturation"]["iqr"], out["saturation"]["n_above_0p40"],
             len(recs), out["saturation"]["n_below_0p30"]))
    low = sorted(recs, key=lambda r: r["lambda_min"])[:5]
    print("    the five lowest-lambda pairs (pid, lambda_min, m, err_m), against "
          "a population median m of %d:" % int(np.median(m)))
    for r in low:
        print("      %-36s %.3f %7d %7.1f"
              % (r["pid"], r["lambda_min"], r["m"], r["shipped_err_m"]))
    out["lowest_five"] = [dict(pid=r["pid"], lambda_min=r["lambda_min"], m=r["m"],
                               shipped_err_m=r["shipped_err_m"]) for r in low]
    out["peak_ratio_err_spearman"] = spearman(
        [r["peak_ratio"] for r in recs], err)
    out["spearman_lambda_peakratio"] = spearman(
        lam, [r["peak_ratio"] for r in recs])
    print("    for reference, Spearman(peak_ratio, err) = %+.3f, "
          "Spearman(lambda_min, peak_ratio) = %+.3f"
          % (out["peak_ratio_err_spearman"], out["spearman_lambda_peakratio"]))
    return out


# ----------------------------------------------------------------------
# Test B
# ----------------------------------------------------------------------
def pct_rank(v):
    """Percentile rank in [0,1], high = better on both signals here."""
    return (rankdata(v) - 1.0) / max(len(v) - 1, 1)


def admitted_stats(err):
    return dict(n=int(len(err)), median_err_m=med(err),
                n_over_30m=int((np.asarray(err) > 30).sum()))


def test_b_insample(recs, levels):
    lam = np.array([r["lambda_min"] for r in recs])
    pr = np.array([r["peak_ratio"] for r in recs])
    err = np.array([r["shipped_err_m"] for r in recs])
    n = len(recs)
    scores = {"peak_ratio only": pct_rank(pr),
              "lambda_min only": pct_rank(lam),
              "both (min of ranks)": np.minimum(pct_rank(pr), pct_rank(lam))}
    print()
    print("=" * 78)
    print("TEST B (i)  IN-SAMPLE matched coverage, thresholds chosen on all %d "
          "pairs" % n)
    print("            an upper bound, NOT an operating point")
    print("=" * 78)
    print("  %-22s %8s %10s %10s %10s"
          % ("rule", "cover", "admitted", "median m", ">30 m"))
    out = []
    for c in levels:
        k = max(1, int(round(c * n)))
        print("  --- target coverage %d%% (%d of %d) ---" % (round(c * 100), k, n))
        for name, s in scores.items():
            idx = np.argsort(-s, kind="mergesort")[:k]
            st = admitted_stats(err[idx])
            print("  %-22s %8s %10d %10.1f %10s"
                  % (name, "%d%%" % round(c * 100), st["n"], st["median_err_m"],
                     "%d/%d" % (st["n_over_30m"], st["n"])))
            out.append(dict(coverage=c, rule=name, **st))
    return out


def test_b_loso(recs, levels):
    """Leave-one-scene-out operating points.

    For each held-out scene the threshold is fitted on the other three scenes to
    hit the target coverage there, then applied unchanged to the held-out scene.
    The combined rule uses one shared quantile q for both signals, scanned on the
    training folds for the q whose joint admission rate is closest to the target,
    which is exactly "admit only if both exceed their thresholds" with the
    thresholds placed at a common quantile.
    """
    scenes = sorted({r["scene"] for r in recs})
    lam = np.array([r["lambda_min"] for r in recs])
    pr = np.array([r["peak_ratio"] for r in recs])
    err = np.array([r["shipped_err_m"] for r in recs])
    sc = np.array([r["scene"] for r in recs])
    n = len(recs)
    qgrid = np.arange(0.0, 0.999, 0.002)

    print()
    print("=" * 78)
    print("TEST B (ii)  LEAVE-ONE-SCENE-OUT operating points, %d scenes" % len(scenes))
    print("=" * 78)
    out = []
    for c in levels:
        print("  --- target coverage %d%% ---" % round(c * 100))
        print("  %-22s %10s %10s %10s   %s"
              % ("rule", "admitted", "median m", ">30 m", "per fold admitted"))
        for rule in ("peak_ratio only", "lambda_min only", "both (min of ranks)"):
            keep = np.zeros(n, bool)
            perfold = []
            for s in scenes:
                te = sc == s
                tr = ~te
                if rule == "peak_ratio only":
                    t = np.quantile(pr[tr], 1.0 - c)
                    k = pr[te] >= t
                elif rule == "lambda_min only":
                    t = np.quantile(lam[tr], 1.0 - c)
                    k = lam[te] >= t
                else:
                    best, bq = None, None
                    for q in qgrid:
                        tp, tl = np.quantile(pr[tr], q), np.quantile(lam[tr], q)
                        cov = float(((pr[tr] >= tp) & (lam[tr] >= tl)).mean())
                        d = abs(cov - c)
                        if best is None or d < best:
                            best, bq = d, q
                    tp, tl = np.quantile(pr[tr], bq), np.quantile(lam[tr], bq)
                    k = (pr[te] >= tp) & (lam[te] >= tl)
                keep[te] = k
                perfold.append("%s %d/%d" % (s[:9], int(k.sum()), int(te.sum())))
            st = admitted_stats(err[keep])
            print("  %-22s %10s %10.1f %10s   %s"
                  % (rule, "%d/%d (%d%%)" % (st["n"], n, round(100.0 * st["n"] / n)),
                     st["median_err_m"], "%d/%d" % (st["n_over_30m"], st["n"]),
                     "  ".join(perfold)))
            out.append(dict(coverage=c, rule=rule, per_fold=perfold, **st))
    return out


def main():
    sanity = run_sanity()
    recs = compute_all()

    lam = np.array([r["lambda_min"] for r in recs])
    m = np.array([r["m"] for r in recs])
    print()
    print("POPULATION  n=%d, all answered=%s" % (len(recs), all(r["answered"] for r in recs)))
    print("  lambda_min  min %.4f  q25 %.4f  median %.4f  q75 %.4f  max %.4f"
          % (lam.min(), np.percentile(lam, 25), np.median(lam),
             np.percentile(lam, 75), lam.max()))
    print("  m           min %d  median %d  max %d" % (m.min(), int(np.median(m)), m.max()))

    a = test_a(recs)
    levels = [0.10, 0.20, 0.30, 0.50]
    b1 = test_b_insample(recs, levels)
    b2 = test_b_loso(recs, levels)

    payload = dict(
        generated="2026-09-15",
        script=os.path.basename(__file__),
        source_matrix=os.path.relpath(MATRIX, PROJ).replace("\\", "/"),
        sketch_field="v08_drone_edges",
        method=dict(normals="structure tensor principal eigenvector",
                    sigma_d=SIGMA_D, sigma_t=SIGMA_T,
                    weights="equal (alpha_i = 1/m)",
                    note="trace(G)=1 by construction, so lambda_max=1-lambda_min"),
        sanity=sanity, pairs=recs, test_a=a,
        test_b_insample=b1, test_b_loso=b2)
    p = os.path.join(HERE, "lambda_geometry.json")
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print()
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
