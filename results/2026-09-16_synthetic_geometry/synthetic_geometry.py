"""Synthetic study of the capped-Chamfer local-geometry proposition (review R2).

CPU only, no torch, no GPU. Seeded. One run regenerates every number in
SYNTHETIC_GEOMETRY.json and the figure.

WHAT IS TESTED
--------------
The proposition (the theory section of the paper): with sketch boundary points
q_i carrying unit normals n_i and weights alpha_i summing to one, and C(t) the
weighted mean capped distance from the shifted points to the map boundaries,

    C(t_star + delta) = l(delta) + O(||delta||^2),   l(delta) = sum_i a_i |n_i.delta|
    G = sum_i a_i n_i n_i^T.

E1 first-order accuracy, E2 lambda_min and the strict minimum, E3 error
direction under sketch corruption, E4 a repeated pattern with full-rank G and
several equal-cost minima.

COST DEFINITION
---------------
Mirrors the deployed matcher. capped_P convention (buildc_common.py:280-288):
P = min(DT(map edges), CAP_M) / CAP_M with DT = cv2.distanceTransform(
~edges, cv2.DIST_L2, 3) * GSD. cost(t) = sum(W * P_shifted) / sum(W), low =
good. W here is a binary sketch, so alpha_i = 1/m uniform and sums to one.

coarse_pool, valid_correlate and peak_ratio_gate below are VERBATIM COPIES of
results/2026-09-08_sketch_vs_roma_bench/run_classical.py:39-67 and
results/2026-09-08_sketch_vs_roma_bench/bench_sketch_vs_roma.py:162-199.
They are copied, not imported, because importing run_classical pulls torch
(15 s) and a CUDA context into a task that must stay on the CPU. The copies
are byte-identical apart from docstring trimming; see SYNTHETIC_GEOMETRY.md.

RESOLUTION POLICY
-----------------
E1/E2/E3 run at NATIVE resolution (1 px = 0.20 m). The deployed pooling factor
8 makes one cell 1.6 m and would erase the sub-metre regime the proposition is
about. E4 runs through the deployed pooled path (pool 8, valid_correlate,
peak_ratio_gate exclusion_radius=3) because rho and the gate are deployed
quantities. A native-vs-pooled argmin cross-check is reported (X1).

THREE COST ARMS in E1, a ladder that separates the mathematics from the pixels:
  ideal    continuous sketch samples exactly on the boundary + analytic distance
  raster   rasterised sketch pixels + analytic distance (adds sketch rasterisation)
  deployed rasterised sketch pixels + cv2 DT mask 3, bilinear sampled (the pipeline)
  precise  as deployed but cv2.DIST_MASK_PRECISE (isolates the mask-3 approximation)
"""
import json
import math
import zlib
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # belt and braces: no GPU

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- constants
SEED = 20260916
IMG = 512                 # map canvas, px
WIN = 240                 # drone sketch window, px (centred)
GSD = 0.20                # m per px, the frozen pair convention
CAP_M = 3.0               # distance cap, m  (= 15 px)
COARSE_F = 8              # deployed pooling factor (train_drawer TRAIN_COARSE_F)
EXCL = 3                  # deployed peak_ratio_gate exclusion radius, cells
EPS = 1e-12

# E1 sampling
STEPS_PX = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0]
N_DIRS_E1 = 36            # directions over the full circle, 10 deg apart
REL_INCLUDE_FRAC = 0.20   # a direction enters the relative-error stat only if
                          # l(phi) >= 0.20 * max_phi l(phi).  FIXED A PRIORI.
REL_TOL = 0.10            # the "under 10 percent" band of the task

# E2
SLOPE_STEP_PX = 2.0       # step at which the measured slope is read, px
N_DIRS_SLOPE = 180

# E3  -- every constant below is fixed A PRIORI, before any result was seen
E3_TRIALS = 40
E3_SEARCH_PX = 20         # integer search half-width
E3_MIN_ERR_PX = 2.0       # trials with |error| below this are dropped from the
                          # angle statistics: the direction of a sub-2-px error
                          # is not defined on a 1-px argmin lattice
E3_PERM = 2000
E3_BOOT = 2000
E3_LEVELS = {             # deletion fraction, jitter sigma px, spurious strokes
    "light":  dict(p_del=0.20, jit=0.5, n_spur=1),
    "medium": dict(p_del=0.40, jit=1.5, n_spur=3),
    "heavy":  dict(p_del=0.60, jit=3.0, n_spur=6),
    # the three levels above were fixed first; the first run displaced the
    # argmin by less than one pixel in almost every trial, so no error
    # direction could be measured.  The two levels below were ADDED AFTER
    # that run, for the sole purpose of producing measurable errors.  Their
    # n_used counts are reported next to the first three.
    "severe": dict(p_del=0.75, jit=5.0, n_spur=12),
    "extreme": dict(p_del=0.90, jit=8.0, n_spur=20),
}
E3_ORDER = ["light", "medium", "heavy", "severe", "extreme"]

# E4
E4_PERIOD_PX = 64         # grid line spacing, px (= 12.8 m, = 8 coarse cells)


# ------------------------------------------- deployed helpers (VERBATIM COPIES)
def coarse_pool(x, factor):
    """run_classical.py:39-49, verbatim."""
    h, w = x.shape
    pad_h, pad_w = (-h) % factor, (-w) % factor
    xp = np.pad(x, ((0, pad_h), (0, pad_w)), mode="constant")
    hh, ww = xp.shape[0] // factor, xp.shape[1] // factor
    return xp.reshape(hh, factor, ww, factor).mean(axis=(1, 3))


def valid_correlate(Wc, Pc):
    """run_classical.py:52-67, verbatim."""
    Wch, Wcw = Wc.shape
    Mch, Mcw = Pc.shape
    oh, ow = Mch - Wch + 1, Mcw - Wcw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("degenerate geometry: oh=%d ow=%d" % (oh, ow))
    wsum = max(float(Wc.sum()), 1e-6)
    cost = np.empty((oh, ow), dtype=np.float32)
    for r in range(oh):
        for c in range(ow):
            cost[r, c] = float((Wc * Pc[r:r + Wch, c:c + Wcw]).sum()) / wsum
    return cost


def peak_ratio_gate(cost, exclusion_radius=3):
    """bench_sketch_vs_roma.py:162-199, verbatim apart from the returned
    runner-up location, which this study needs to verify the runner-up really
    is the period-shifted peak."""
    r_best, c_best = np.unravel_index(np.argmin(cost), cost.shape)
    best_val = float(cost[r_best, c_best])
    oh, ow = cost.shape
    rlo, rhi = max(0, r_best - exclusion_radius), min(oh, r_best + exclusion_radius + 1)
    clo, chi = max(0, c_best - exclusion_radius), min(ow, c_best + exclusion_radius + 1)
    valid = np.ones_like(cost, dtype=bool)
    valid[rlo:rhi, clo:chi] = False
    if not valid.any():
        return None, r_best, c_best, None
    masked = np.where(valid, cost, np.inf)
    r2, c2 = np.unravel_index(np.argmin(masked), masked.shape)
    cost_second = float(cost[r2, c2])
    ratio = float(cost_second) / max(best_val, 1e-9)
    return ratio, (r_best, c_best), (r2, c2), cost_second


def capped_P(edges, mask=cv2.DIST_MASK_3):
    """buildc_common.py:280-288 convention, with the DT mask exposed."""
    DTm = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, mask) * GSD
    return (np.minimum(DTm, CAP_M) / CAP_M).astype(np.float32)


# ------------------------------------------------------------------ geometry
class Seg(object):
    """Straight boundary. Normal is constant, perpendicular to the segment."""

    def __init__(self, p0, p1):
        self.p0 = np.asarray(p0, float)
        self.p1 = np.asarray(p1, float)
        d = self.p1 - self.p0
        self.L = float(np.hypot(*d))
        self.u = d / self.L
        self.n = np.array([-self.u[1], self.u[0]])   # unit normal, (x, y)

    def dist(self, pts):
        """Exact point-to-segment distance, pts (N,2) in (x, y) px."""
        w = pts - self.p0[None, :]
        t = np.clip(w @ self.u, 0.0, self.L)
        proj = self.p0[None, :] + t[:, None] * self.u[None, :]
        return np.hypot(*(pts - proj).T)

    def normals(self, pts):
        return np.repeat(self.n[None, :], len(pts), axis=0)

    def sample(self, spacing=1.0):
        k = max(int(round(self.L / spacing)), 2)
        t = (np.arange(k) + 0.5) * (self.L / k)
        return self.p0[None, :] + t[:, None] * self.u[None, :]

    def draw(self, img):
        cv2.line(img, tuple(np.round(self.p0).astype(int)),
                 tuple(np.round(self.p1).astype(int)), 1, thickness=1,
                 lineType=cv2.LINE_8)


class Circ(object):
    """Circular boundary. Normal is radial."""

    def __init__(self, c, R):
        self.c = np.asarray(c, float)
        self.R = float(R)
        self.L = 2 * math.pi * self.R

    def dist(self, pts):
        return np.abs(np.hypot(*(pts - self.c[None, :]).T) - self.R)

    def normals(self, pts):
        v = pts - self.c[None, :]
        r = np.hypot(*v.T)[:, None]
        return v / np.maximum(r, EPS)

    def sample(self, spacing=1.0):
        k = max(int(round(self.L / spacing)), 8)
        a = (np.arange(k) + 0.5) * (2 * math.pi / k)
        return self.c[None, :] + self.R * np.stack([np.cos(a), np.sin(a)], 1)

    def draw(self, img):
        cv2.circle(img, tuple(np.round(self.c).astype(int)), int(round(self.R)),
                   1, thickness=1, lineType=cv2.LINE_8)


def line_through(cx, cy, deg, half_len):
    a = math.radians(deg)
    d = np.array([math.cos(a), math.sin(a)])
    c = np.array([cx, cy], float)
    return Seg(c - half_len * d, c + half_len * d)


C0 = IMG / 2.0
HALF = IMG                      # long enough to cross the whole canvas


def make_patterns():
    """Every pattern is a list of primitives in MAP coordinates."""
    P = {}
    P["line_single"] = [line_through(C0, C0, 20, HALF)]
    P["cross_90"] = [line_through(C0, C0, 20, HALF), line_through(C0, C0, 110, HALF)]
    P["cross_30"] = [line_through(C0, C0, 20, HALF), line_through(C0, C0, 50, HALF)]
    s = 90.0
    P["square"] = [Seg((C0 - s, C0 - s), (C0 + s, C0 - s)),
                   Seg((C0 + s, C0 - s), (C0 + s, C0 + s)),
                   Seg((C0 + s, C0 + s), (C0 - s, C0 + s)),
                   Seg((C0 - s, C0 + s), (C0 - s, C0 - s))]
    P["circle"] = [Circ((C0, C0), 80.0)]
    # road: two parallel kerbs 12 px apart at 8 deg.  field edge: a shorter
    # boundary at 75 deg.  Unequal boundary lengths give unequal weights.
    a = math.radians(8.0)
    off = 6.0 * np.array([-math.sin(a), math.cos(a)])
    P["road_field"] = [
        Seg(np.array([C0, C0]) - HALF * np.array([math.cos(a), math.sin(a)]) + off,
            np.array([C0, C0]) + HALF * np.array([math.cos(a), math.sin(a)]) + off),
        Seg(np.array([C0, C0]) - HALF * np.array([math.cos(a), math.sin(a)]) - off,
            np.array([C0, C0]) + HALF * np.array([math.cos(a), math.sin(a)]) - off),
        line_through(C0 + 40, C0 + 40, 75, 100.0)]
    # a deliberately uneven right-angle cross for E3 (anisotropic G)
    P["cross_90_uneven"] = [line_through(C0, C0, 20, HALF),
                            line_through(C0, C0, 110, 60.0)]
    # E4 repeated pattern
    g = []
    for k in range(-4, 5):
        g.append(Seg((C0 + k * E4_PERIOD_PX, 0), (C0 + k * E4_PERIOD_PX, IMG - 1)))
        g.append(Seg((0, C0 + k * E4_PERIOD_PX), (IMG - 1, C0 + k * E4_PERIOD_PX)))
    P["grid"] = g
    return P


def raster(prims, shape=(IMG, IMG)):
    img = np.zeros(shape, np.uint8)
    for p in prims:
        p.draw(img)
    return img.astype(bool)


def window_slice():
    lo = (IMG - WIN) // 2
    return slice(lo, lo + WIN), slice(lo, lo + WIN), lo


def sketch_points(prims):
    """Rasterised sketch: the map on-pixels inside the centred window, each
    with the normal of its owning primitive.  A pixel whose second-nearest
    primitive is within 2 px is flagged as a junction, where the proposition's
    smoothness assumption does not hold."""
    edges = raster(prims)
    ys, xs, lo = window_slice()
    m = np.zeros_like(edges)
    m[ys, xs] = True
    yy, xx = np.nonzero(edges & m)
    pts = np.stack([xx.astype(float), yy.astype(float)], 1)   # (x, y)
    D = np.stack([p.dist(pts) for p in prims], 1)             # (N, K)
    own = np.argmin(D, 1)
    srt = np.sort(D, 1)
    junction = (srt[:, 1] < 2.0) if D.shape[1] > 1 else np.zeros(len(pts), bool)
    n = np.zeros((len(pts), 2))
    for k, p in enumerate(prims):
        sel = own == k
        if sel.any():
            n[sel] = p.normals(pts[sel])
    return pts, n, junction, lo


def ideal_points(prims, spacing=1.0):
    """Continuous samples lying exactly on the boundary, inside the window."""
    lo = (IMG - WIN) // 2
    hi = lo + WIN - 1
    allp, alln = [], []
    for p in prims:
        s = p.sample(spacing)
        keep = (s[:, 0] >= lo) & (s[:, 0] <= hi) & (s[:, 1] >= lo) & (s[:, 1] <= hi)
        if keep.any():
            allp.append(s[keep])
            alln.append(p.normals(s[keep]))
    return np.concatenate(allp, 0), np.concatenate(alln, 0)


def analytic_dist(prims, pts):
    return np.min(np.stack([p.dist(pts) for p in prims], 1), 1)


# ------------------------------------------------------------ G and profiles
def gram(n):
    """G = mean_i n_i n_i^T with uniform alpha = 1/m.  Trace is 1 for unit n."""
    return (n[:, :, None] * n[:, None, :]).mean(0)


def eig2(G):
    ev, V = np.linalg.eigh(G)
    ev = np.clip(ev, 0.0, None)
    return float(ev[0]), float(ev[1]), V[:, 0].copy(), V[:, 1].copy()


def axial_deg(v):
    return float(np.mod(np.degrees(math.atan2(v[1], v[0])), 180.0))


def fold90(a, b):
    return float(abs((a - b + 90.0) % 180.0 - 90.0))


PHI = np.radians(np.arange(0, 180, 0.25))


def l_profile(n):
    """l(phi) = mean_i |n_i . d(phi)| for unit d(phi).  Dimensionless."""
    d = np.stack([np.cos(PHI), np.sin(PHI)], 1)          # (P, 2)
    return np.abs(n @ d.T).mean(0)                        # (P,)


def l_argmin_dirs(n, tol=1e-3):
    """Every direction (deg, axial) within `tol` of the minimum of l(phi),
    clustered.  A tie means the weakest direction is not unique."""
    prof = l_profile(n)
    mn = prof.min()
    idx = np.nonzero(prof <= mn * (1.0 + tol) + 1e-12)[0]
    dirs, cur = [], [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a <= 2:
            cur.append(b)
        else:
            dirs.append(cur)
            cur = [b]
    dirs.append(cur)
    # a cluster touching both ends of [0,180) is one axial direction
    if len(dirs) > 1 and dirs[0][0] == 0 and dirs[-1][-1] == len(prof) - 1:
        dirs[0] = dirs[-1] + dirs[0]
        dirs.pop()
    return [round(float(np.degrees(PHI[c[len(c) // 2]])), 2) for c in dirs]


def l_of(n, phi_rad):
    d = np.array([math.cos(phi_rad), math.sin(phi_rad)])
    return float(np.abs(n @ d).mean())


# ------------------------------------------------------------------- costing
def cost_analytic(prims, pts, dx, dy):
    """Capped mean distance in METRES for the shifted point set."""
    d = analytic_dist(prims, pts + np.array([dx, dy])[None, :]) * GSD
    return float(np.minimum(d, CAP_M).mean())


def cost_field(P, pts, dx, dy):
    """Deployed field, bilinear sampled, returned in METRES (P is normalised
    by the cap, so multiply back by CAP_M)."""
    c = pts[:, 0] + dx
    r = pts[:, 1] + dy
    v = map_coordinates(P, [r, c], order=1, mode="nearest")
    return float(v.mean()) * CAP_M


# =============================================================== E1 + E2
def run_e1_e2(patterns, log):
    res = {}
    dirs = np.radians(np.arange(0, 360, 360.0 / N_DIRS_E1))
    for name in ["line_single", "cross_90", "cross_30", "square", "circle",
                 "road_field"]:
        prims = patterns[name]
        edges = raster(prims)
        P_dep = capped_P(edges, cv2.DIST_MASK_3)
        P_pre = capped_P(edges, cv2.DIST_MASK_PRECISE)
        pts_r, n_r, junc, lo = sketch_points(prims)
        pts_i, n_i = ideal_points(prims)

        G_r, G_i = gram(n_r), gram(n_i)
        assert abs(np.trace(G_r) - 1.0) < 1e-9, "trace(G) != 1 (raster) %s" % name
        assert abs(np.trace(G_i) - 1.0) < 1e-9, "trace(G) != 1 (ideal) %s" % name
        lam_min, lam_max, vmin, vmax = eig2(G_r)
        lmi, lma, vmi_i, _ = eig2(G_i)
        prof = l_profile(n_r)
        prof_i = l_profile(n_i)
        l_min = float(prof.min())
        l_max = float(prof.max())
        l_min_ideal = float(prof_i.min())
        l_weak_deg = float(np.degrees(PHI[int(np.argmin(prof))]))
        l_weak_dirs = l_argmin_dirs(n_i)
        vmin_deg = axial_deg(vmin)

        # zero-shift floors
        c0_dep = cost_field(P_dep, pts_r, 0.0, 0.0)
        c0_ideal = cost_analytic(prims, pts_i, 0.0, 0.0)
        c0_raster = cost_analytic(prims, pts_r, 0.0, 0.0)
        assert c0_dep < 1e-9, "deployed cost(0) not zero for %s: %g" % (name, c0_dep)

        arms = {}
        for arm in ["ideal", "raster", "deployed", "precise"]:
            pts = pts_i if arm == "ideal" else pts_r
            nn = n_i if arm == "ideal" else n_r
            rows = []
            for s in STEPS_PX:
                s_m = s * GSD
                true, pred = [], []
                for phi in dirs:
                    dx, dy = s * math.cos(phi), s * math.sin(phi)
                    if arm in ("ideal", "raster"):
                        c = cost_analytic(prims, pts, dx, dy)
                    elif arm == "deployed":
                        c = cost_field(P_dep, pts, dx, dy)
                    else:
                        c = cost_field(P_pre, pts, dx, dy)
                    true.append(c)
                    pred.append(s_m * l_of(nn, phi))
                true = np.array(true)
                pred = np.array(pred)
                resid = np.abs(true - pred)
                inc = pred >= REL_INCLUDE_FRAC * pred.max()
                rel = resid[inc] / np.maximum(pred[inc], EPS)
                rows.append(dict(
                    step_px=s, step_m=round(s_m, 4),
                    resid_over_step_mean=float((resid / s_m).mean()),
                    resid_over_step_max=float((resid / s_m).max()),
                    rel_median=float(np.median(rel)), rel_max=float(rel.max()),
                    n_dirs_included=int(inc.sum()),
                    C_true_mean_m=float(true.mean()), C_pred_mean_m=float(pred.mean())))
            # contiguous band of steps whose WORST included direction is under 10 %
            ok = [r["rel_max"] <= REL_TOL for r in rows]
            band = None
            if any(ok):
                k = int(np.argmin([r["rel_max"] for r in rows]))
                a = k
                while a - 1 >= 0 and ok[a - 1]:
                    a -= 1
                b = k
                while b + 1 < len(ok) and ok[b + 1]:
                    b += 1
                band = [rows[a]["step_m"], rows[b]["step_m"]]
            mono = None          # largest step with rel_max <= tol for ALL steps up to it
            for k, o in enumerate(ok):
                if not o:
                    break
                mono = rows[k]["step_m"]
            arms[arm] = dict(rows=rows, band_10pct_m=band, monotone_10pct_m=mono)
            log("  E1 %-11s %-8s band<=10%%: %s   monotone: %s" %
                (name, arm, band, mono))

        # ---- E2 measured slopes, dimensionless (metres of cost per metre)
        sl = {}
        for arm, fn in (("ideal", lambda dx, dy: cost_analytic(prims, pts_i, dx, dy)),
                        ("deployed", lambda dx, dy: cost_field(P_dep, pts_r, dx, dy))):
            phis = np.radians(np.arange(0, 180, 180.0 / N_DIRS_SLOPE))
            vals = []
            for phi in phis:
                a = fn(SLOPE_STEP_PX * math.cos(phi), SLOPE_STEP_PX * math.sin(phi))
                b = fn(-SLOPE_STEP_PX * math.cos(phi), -SLOPE_STEP_PX * math.sin(phi))
                vals.append(0.5 * (a + b) / (SLOPE_STEP_PX * GSD))
            vals = np.array(vals)
            sl[arm] = dict(min=float(vals.min()), max=float(vals.max()),
                           weak_deg=float(np.degrees(phis[int(np.argmin(vals))])),
                           profile=[round(float(v), 5) for v in vals])
        # sliding probe: cost along the measured weakest direction out to 5 m
        wphi = math.radians(sl["ideal"]["weak_deg"])
        slide = {}
        for r_m in (1.0, 2.0, 5.0):
            r_px = r_m / GSD
            slide["%.0fm" % r_m] = round(max(
                cost_analytic(prims, pts_i, r_px * math.cos(wphi), r_px * math.sin(wphi)),
                cost_analytic(prims, pts_i, -r_px * math.cos(wphi), -r_px * math.sin(wphi))), 5)

        res[name] = dict(
            m_raster=int(len(pts_r)), m_ideal=int(len(pts_i)),
            junction_frac=float(junc.mean()),
            G=[[float(v) for v in row] for row in G_r],
            lambda_min=lam_min, lambda_max=lam_max,
            lambda_min_ideal=lmi, lambda_max_ideal=lma,
            l_min=l_min, l_max=l_max, l_min_ideal=l_min_ideal,
            l_weak_dirs_ideal_deg=l_weak_dirs,
            vmin_deg=vmin_deg, l_weak_deg=l_weak_deg,
            vmin_vs_lweak_deg=fold90(vmin_deg, l_weak_deg),
            cost0_ideal_m=c0_ideal, cost0_raster_m=c0_raster, cost0_deployed_m=c0_dep,
            slopes=sl, slide_max_cost_m=slide, arms=arms)
        log("  E2 %-11s lam=[%.4f %.4f] l_min=%.4f meas_slope_min(ideal)=%.4f "
            "(dep)=%.4f  vmin=%.1fdeg l_weak_dirs=%s d=%.1f  junc=%.3f  "
            "slide(<=5m)=%s" %
            (name, lam_min, lam_max, l_min, sl["ideal"]["min"], sl["deployed"]["min"],
             vmin_deg, l_weak_dirs, fold90(vmin_deg, l_weak_deg), junc.mean(),
             slide))
    return res


# ===================================================================== E3
def corrupt(pts, rng, p_del, jit, n_spur):
    keep = rng.random(len(pts)) >= p_del
    q = pts[keep].copy()
    if jit > 0:
        q = q + rng.normal(0.0, jit, size=q.shape)
    q = np.round(q)
    extra = []
    lo = (IMG - WIN) // 2
    for _ in range(n_spur):
        c = rng.uniform(lo + 20, lo + WIN - 20, size=2)
        a = rng.uniform(0, math.pi)
        L = rng.uniform(20, 70)
        d = np.array([math.cos(a), math.sin(a)])
        t = np.arange(-L / 2, L / 2, 1.0)
        extra.append(np.round(c[None, :] + t[:, None] * d[None, :]))
    if extra:
        q = np.concatenate([q] + extra, 0)
    q[:, 0] = np.clip(q[:, 0], 0, IMG - 1)
    q[:, 1] = np.clip(q[:, 1], 0, IMG - 1)
    return q


def argmin_offset(P, pts, half=E3_SEARCH_PX):
    """Integer argmin of the deployed cost over a +-half px window, then a
    quadratic 3-point refinement in each axis."""
    r = pts[:, 1].astype(np.int64)
    c = pts[:, 0].astype(np.int64)
    n = 2 * half + 1
    grid = np.empty((n, n), np.float64)
    for i, dy in enumerate(range(-half, half + 1)):
        rr = np.clip(r + dy, 0, IMG - 1)
        for j, dx in enumerate(range(-half, half + 1)):
            cc = np.clip(c + dx, 0, IMG - 1)
            grid[i, j] = P[rr, cc].mean()
    i0, j0 = np.unravel_index(np.argmin(grid), grid.shape)
    dy = float(i0 - half)
    dx = float(j0 - half)
    if 0 < i0 < n - 1:
        a, b, cc_ = grid[i0 - 1, j0], grid[i0, j0], grid[i0 + 1, j0]
        den = (a - 2 * b + cc_)
        if den > EPS:
            dy += float(np.clip(0.5 * (a - cc_) / den, -0.5, 0.5))
    if 0 < j0 < n - 1:
        a, b, cc_ = grid[i0, j0 - 1], grid[i0, j0], grid[i0, j0 + 1]
        den = (a - 2 * b + cc_)
        if den > EPS:
            dx += float(np.clip(0.5 * (a - cc_) / den, -0.5, 0.5))
    return dx, dy, grid


def run_e3(patterns, log):
    rng_master = np.random.default_rng(SEED)
    names = ["cross_30", "road_field", "cross_90_uneven", "cross_90"]
    out = dict(per_cell={}, pooled={})
    trials_all = []
    for name in names:
        prims = patterns[name]
        edges = raster(prims)
        P = capped_P(edges, cv2.DIST_MASK_3)
        pts_r, n_r, junc, lo = sketch_points(prims)
        G = gram(n_r)
        lam_min, lam_max, vmin, _ = eig2(G)
        prof = l_profile(n_r)
        vmin_deg = axial_deg(vmin)
        l_weak_deg = float(np.degrees(PHI[int(np.argmin(prof))]))
        aniso = lam_min / max(lam_max, EPS)
        for lvl in E3_ORDER:
            kw = E3_LEVELS[lvl]
            thetas_v, thetas_l, thetas_cor, mags = [], [], [], []
            n_drop = 0
            for k in range(E3_TRIALS):
                # zlib.crc32, NOT hash(): Python's string hash is randomised
                # per process, which would make the trials unreproducible.
                rng = np.random.default_rng([SEED, zlib.crc32(name.encode()),
                                             zlib.crc32(lvl.encode()), k])
                q = corrupt(pts_r, rng, **kw)
                dx, dy, _ = argmin_offset(P, q)
                mag = math.hypot(dx, dy)
                mags.append(mag)
                if mag < E3_MIN_ERR_PX:
                    n_drop += 1
                    continue
                ed = axial_deg((dx, dy))
                thetas_v.append(fold90(ed, vmin_deg))
                thetas_l.append(fold90(ed, l_weak_deg))
                # corrupted-sketch G: what the deployed system actually holds
                Dq = np.stack([p.dist(q) for p in prims], 1)
                ownq = np.argmin(Dq, 1)
                nq = np.zeros((len(q), 2))
                for kk, p in enumerate(prims):
                    sel = ownq == kk
                    if sel.any():
                        nq[sel] = p.normals(q[sel])
                _, _, vq, _ = eig2(gram(nq))
                thetas_cor.append(fold90(ed, axial_deg(vq)))
                trials_all.append((name, lvl, ed, vmin_deg))
            key = "%s|%s" % (name, lvl)
            ok = len(thetas_v) > 0
            tv = np.array(thetas_v) if ok else None
            out["per_cell"][key] = dict(
                pattern=name, level=lvl, lambda_min=lam_min, lambda_max=lam_max,
                anisotropy=float(aniso), vmin_deg=vmin_deg, l_weak_deg=l_weak_deg,
                n_trials=E3_TRIALS, n_used=len(thetas_v), n_dropped_below_floor=n_drop,
                err_px_median=float(np.median(mags)),
                err_m_median=float(np.median(mags) * GSD),
                mean_angle_vmin=float(tv.mean()) if ok else None,
                median_angle_vmin=float(np.median(tv)) if ok else None,
                share_within_45=float((tv <= 45.0).mean()) if ok else None,
                mean_angle_lweak=float(np.mean(thetas_l)) if ok else None,
                mean_angle_vmin_corrupted=float(np.mean(thetas_cor)) if ok else None)
            log("  E3 %-16s %-7s aniso=%.3f n=%2d/%2d med|e|=%.2fm "
                "mean_ang=%5s share45=%5s (vs l-weak %5s)" %
                (name, lvl, aniso, len(thetas_v), E3_TRIALS,
                 np.median(mags) * GSD,
                 ("%.1f" % tv.mean()) if ok else "n/a",
                 ("%.2f" % (tv <= 45.0).mean()) if ok else "n/a",
                 ("%.1f" % np.mean(thetas_l)) if ok else "n/a"))

    # pooled over the three anisotropic patterns.  Permutation null: re-pair
    # each trial's error direction with another trial's weak direction, which
    # is the synthetic counterpart of the real-data block permutation.  The
    # uniform-axial null (mean 45) is reported alongside as a reference only.
    aniso_names = ["cross_30", "road_field", "cross_90_uneven"]
    rngp = np.random.default_rng(SEED + 7)

    def pooled_stats(sel):
        ed = np.array([t[2] for t in sel])
        wd = np.array([t[3] for t in sel])
        obs = np.array([fold90(a, b) for a, b in zip(ed, wd)])
        perm = np.empty(E3_PERM)
        for i in range(E3_PERM):
            w = rngp.permutation(wd)
            perm[i] = np.mean([fold90(a, b) for a, b in zip(ed, w)])
        boot = np.empty(E3_BOOT)
        for i in range(E3_BOOT):
            boot[i] = obs[rngp.integers(0, len(obs), len(obs))].mean()
        return dict(
            patterns=aniso_names, n=int(len(obs)),
            mean_angle=float(obs.mean()), median_angle=float(np.median(obs)),
            ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            share_within_45=float((obs <= 45).mean()),
            perm_null_mean=float(perm.mean()),
            perm_null_ci95=[float(np.percentile(perm, 2.5)),
                            float(np.percentile(perm, 97.5))],
            p_perm_one_sided=float((perm <= obs.mean()).mean()),
            uniform_null_mean=45.0)

    for lvl in E3_ORDER + ["all"]:
        sel = [t for t in trials_all if t[0] in aniso_names
               and (lvl == "all" or t[1] == lvl)]
        if len(sel) < 5:
            log("  E3 POOLED %-7s n=%d  too few usable trials for a statistic"
                % (lvl, len(sel)))
            continue
        st = pooled_stats(sel)
        out["pooled"][lvl] = st
        log("  E3 POOLED %-7s n=%3d mean=%.1f deg CI95=[%.1f %.1f] share45=%.2f "
            "perm-null=%.1f p=%.4f (uniform null 45)" %
            (lvl, st["n"], st["mean_angle"], st["ci95"][0], st["ci95"][1],
             st["share_within_45"], st["perm_null_mean"], st["p_perm_one_sided"]))
    return out


# ===================================================================== E4
def native_grid_cost(P, pts, H):
    """Cost over a +-H px integer window, in normalised units (low = good)."""
    r = pts[:, 1].astype(np.int64)
    c = pts[:, 0].astype(np.int64)
    n = 2 * H + 1
    g = np.empty((n, n))
    for i, dy in enumerate(range(-H, H + 1)):
        rr = np.clip(r + dy, 0, IMG - 1)
        for j, dx in enumerate(range(-H, H + 1)):
            cc = np.clip(c + dx, 0, IMG - 1)
            g[i, j] = P[rr, cc].mean()
    return g


def pooled_cost(P, pts):
    """The deployed path: sketch raster -> coarse_pool -> valid_correlate."""
    ys, xs, lo = window_slice()
    Wr = np.zeros((IMG, IMG), np.float32)
    Wr[np.clip(pts[:, 1].astype(int), 0, IMG - 1),
       np.clip(pts[:, 0].astype(int), 0, IMG - 1)] = 1.0
    Wc = coarse_pool(Wr[ys, xs], COARSE_F)
    Hh, Ww = P.shape
    ph, pw = (-Hh) % COARSE_F, (-Ww) % COARSE_F
    Pp = np.pad(P, ((0, ph), (0, pw)), constant_values=1.0)
    Pc = Pp.reshape(Pp.shape[0] // COARSE_F, COARSE_F,
                    Pp.shape[1] // COARSE_F, COARSE_F).mean(axis=(1, 3))
    return valid_correlate(Wc, Pc), lo / float(COARSE_F)


def e4_one(P, pts, log, tag):
    """Both resolutions for one sketch: cost at the true minimum, at the
    period-shifted alternatives, and the deployed peak ratio."""
    H = int(2.5 * E4_PERIOD_PX)
    grid = native_grid_cost(P, pts, H)
    c_true = float(grid[H, H])
    alt = {}
    for lbl, (dy, dx) in (("+x", (0, E4_PERIOD_PX)), ("-x", (0, -E4_PERIOD_PX)),
                          ("+y", (E4_PERIOD_PX, 0)), ("-y", (-E4_PERIOD_PX, 0)),
                          ("+x+y", (E4_PERIOD_PX, E4_PERIOD_PX))):
        alt[lbl] = float(grid[H + dy, H + dx]) * CAP_M
    cost, true_rc = pooled_cost(P, pts)
    ratio, best, second, c2 = peak_ratio_gate(cost, exclusion_radius=EXCL)
    dr, dc = int(second[0] - best[0]), int(second[1] - best[1])
    per_cells = E4_PERIOD_PX // COARSE_F
    is_period = (abs(abs(dr) - per_cells) <= 1 and dc == 0) or \
                (abs(abs(dc) - per_cells) <= 1 and dr == 0) or \
                (abs(abs(dr) - per_cells) <= 1 and abs(abs(dc) - per_cells) <= 1)
    rec = dict(
        native=dict(C1_m=c_true * CAP_M, alt_costs_m=alt,
                    cost_at_half_period_m=float(grid[H, H + E4_PERIOD_PX // 2]) * CAP_M,
                    cost_at_quarter_period_m=float(grid[H, H + E4_PERIOD_PX // 4]) * CAP_M),
        deployed=dict(cost_shape=list(cost.shape), true_cell=round(true_rc, 2),
                      best_cell=[int(best[0]), int(best[1])],
                      best_cell_minus_true=[int(best[0] - round(true_rc)),
                                            int(best[1] - round(true_rc))],
                      C1=float(cost[best]), C2=float(c2), peak_ratio=ratio,
                      C1_m=float(cost[best]) * CAP_M, C2_m=float(c2) * CAP_M,
                      runner_up_delta_cells=[dr, dc], period_in_cells=int(per_cells),
                      runner_up_is_period_shift=bool(is_period),
                      exclusion_radius_cells=EXCL))
    log("  E4[%s] native C1=%.4f m  alternatives (m): %s" %
        (tag, c_true * CAP_M, {k: round(v, 4) for k, v in alt.items()}))
    log("  E4[%s] deployed C1=%.5f C2=%.5f rho=%.4f  runner-up delta=(%d,%d) "
        "cells, period=%d cells, is_period=%s" %
        (tag, float(cost[best]), c2, ratio, dr, dc, per_cells, is_period))
    return rec, grid


def run_e4(patterns, log):
    prims = patterns["grid"]
    edges = raster(prims)
    P = capped_P(edges, cv2.DIST_MASK_3)
    pts_r, n_r, junc, lo = sketch_points(prims)
    lam_min, lam_max, vmin, _ = eig2(gram(n_r))
    prof = l_profile(n_r)

    clean, grid_clean = e4_one(P, pts_r, log, "clean")
    rng = np.random.default_rng(SEED + 11)
    pts_c = corrupt(pts_r, rng, p_del=0.30, jit=1.0, n_spur=2)
    noisy, grid_noisy = e4_one(P, pts_c, log, "corrupted")

    # ---- X1: native vs pooled argmin on a NON-periodic pattern (road_field),
    # clean and corrupted, so the comparison is not confounded by exact ties.
    rf = patterns["road_field"]
    Prf = capped_P(raster(rf), cv2.DIST_MASK_3)
    prf, _, _, lo_rf = sketch_points(rf)
    x1 = {}
    for tag, pp in (("clean", prf),
                    ("corrupted", corrupt(prf, np.random.default_rng(SEED + 13),
                                          p_del=0.40, jit=1.5, n_spur=3))):
        g = native_grid_cost(Prf, pp, 24)
        i, j = np.unravel_index(np.argmin(g), g.shape)
        cst, trc = pooled_cost(Prf, pp)
        _, b, _, _ = peak_ratio_gate(cst, exclusion_radius=EXCL)
        x1[tag] = dict(
            native_argmin_m=[float(round((j - 24) * GSD, 3)),
                             float(round((i - 24) * GSD, 3))],
            pooled_argmin_m=[float(round((b[1] - trc) * COARSE_F * GSD, 3)),
                             float(round((b[0] - trc) * COARSE_F * GSD, 3))],
            pooled_cell_m=COARSE_F * GSD)
        log("  X1 road_field %-9s native argmin (x,y) m=%s ; pooled argmin "
            "(x,y) m=%s ; 1 pooled cell = %.2f m" %
            (tag, x1[tag]["native_argmin_m"], x1[tag]["pooled_argmin_m"],
             COARSE_F * GSD))

    out = dict(period_px=E4_PERIOD_PX, period_m=E4_PERIOD_PX * GSD,
               m_raster=int(len(pts_r)), lambda_min=lam_min, lambda_max=lam_max,
               l_min=float(prof.min()),
               clean_sketch=clean, corrupted_sketch=noisy,
               corruption="p_del=0.30, jitter sigma=1.0 px, 2 spurious strokes",
               cross_check_X1=x1,
               native_grid_clean=grid_clean, native_grid_noisy=grid_noisy)
    log("  E4 period=%.1f m  lambda_min=%.4f lambda_max=%.4f l_min=%.4f "
        "(full rank, locally strict)" %
        (E4_PERIOD_PX * GSD, lam_min, lam_max, float(prof.min())))
    return out


# ================================================================== figure
def make_figure(e12, e3, e4, patterns):
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.2))
    cols = plt.cm.tab10(np.linspace(0, 1, 10))

    # (a) E1 relative error vs step, deployed arm and ideal arm
    a = ax[0, 0]
    names = ["line_single", "cross_90", "cross_30", "square", "circle", "road_field"]
    for k, nm in enumerate(names):
        rows = e12[nm]["arms"]["ideal"]["rows"]
        s = [r["step_m"] for r in rows]
        y = [100 * r["rel_max"] for r in rows]
        a.plot(s, y, "-", color=cols[k], lw=1.2, label=nm.replace("_", " "))
        rows = e12[nm]["arms"]["deployed"]["rows"]
        a.plot([r["step_m"] for r in rows], [100 * r["rel_max"] for r in rows],
               "--", color=cols[k], lw=0.9, alpha=0.7)
    a.axhline(10, color="k", lw=0.8, ls=":")
    a.axvline(CAP_M, color="gray", lw=0.8, ls="-.")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("step size $\\|\\delta\\|$ (m)")
    a.set_ylabel("worst-direction relative error (%)")
    a.set_title("(a) E1  solid: ideal, dashed: deployed", fontsize=8)
    a.legend(fontsize=5.2, ncol=2, loc="upper left")
    a.tick_params(labelsize=7)

    # (b) E2 measured slope vs lambda_min and l_min
    b = ax[0, 1]
    lam = [e12[n]["lambda_min"] for n in names]
    lmn = [e12[n]["l_min"] for n in names]
    mea = [e12[n]["slopes"]["ideal"]["min"] for n in names]
    b.plot([0, 0.7], [0, 0.7], "k:", lw=0.8)
    b.scatter(lam, mea, s=22, c="tab:red", marker="s", label="$\\lambda_{\\min}(G)$")
    b.scatter(lmn, mea, s=22, c="tab:blue", marker="o", label="$\\ell_{\\min}$")
    for k, nm in enumerate(names):
        b.annotate(nm.replace("_", " "), (lmn[k], mea[k]), fontsize=5,
                   xytext=(3, -6), textcoords="offset points")
    b.set_xlabel("predicted rise per metre")
    b.set_ylabel("measured min slope")
    b.set_title("(b) E2  weakest-direction cost rise", fontsize=8)
    b.legend(fontsize=6, loc="upper left")
    b.tick_params(labelsize=7)

    # (c) E3 mean angle per pattern and level (only cells with usable trials)
    c = ax[1, 0]
    pats = ["cross_30", "road_field", "cross_90_uneven", "cross_90"]
    lv = [l for l in E3_ORDER
          if any(e3["per_cell"]["%s|%s" % (p, l)]["n_used"] > 0 for p in pats)]
    w = 0.8 / max(len(lv), 1)
    for i, l in enumerate(lv):
        vals, ns = [], []
        for p in pats:
            r = e3["per_cell"]["%s|%s" % (p, l)]
            vals.append(r["mean_angle_vmin"] if r["n_used"] else 0.0)
            ns.append(r["n_used"])
        pos = np.arange(len(pats)) + (i - (len(lv) - 1) / 2.0) * w
        c.bar(pos, vals, w, label=l, color=cols[i])
        for x, v, nn in zip(pos, vals, ns):
            c.annotate("%d" % nn, (x, v), fontsize=4.5, ha="center",
                       xytext=(0, 1), textcoords="offset points")
    c.axhline(45, color="k", lw=0.9, ls=":")
    c.axhline(e3["pooled"]["all"]["perm_null_mean"], color="tab:red", lw=0.9, ls="--")
    c.set_xticks(np.arange(len(pats)))
    c.set_xticklabels([p.replace("_", " ") for p in pats], fontsize=5.5, rotation=12)
    c.set_ylabel("mean angle to $v_{\\min}$ (deg)")
    c.set_title("(c) E3  dotted: uniform null 45, dashed: permutation null",
                fontsize=7.5)
    c.legend(fontsize=5.5, ncol=2)
    c.tick_params(labelsize=7)

    # (d) E4 cost profile along x through the true minimum
    d = ax[1, 1]
    for g, lbl, col in ((e4["native_grid_clean"], "clean sketch", "tab:purple"),
                        (e4["native_grid_noisy"], "corrupted sketch", "tab:orange")):
        H = (g.shape[0] - 1) // 2
        xs = (np.arange(g.shape[1]) - H) * GSD
        d.plot(xs, g[H] * CAP_M, "-", color=col, lw=1.1, label=lbl)
    for k in (-2, -1, 1, 2):
        d.axvline(k * e4["period_m"], color="gray", lw=0.7, ls=":")
    d.set_xlabel("shift along x (m)")
    d.set_ylabel("capped chamfer cost (m)")
    d.set_title("(d) E4  repeated pattern, $\\rho$=%.3f (corrupted)" %
                e4["corrupted_sketch"]["deployed"]["peak_ratio"], fontsize=8)
    d.legend(fontsize=6)
    d.tick_params(labelsize=7)

    fig.tight_layout(pad=0.6)
    fig.savefig(os.path.join(OUT, "fig_synthetic.pdf"))
    fig.savefig(os.path.join(OUT, "fig_synthetic.png"), dpi=200)
    plt.close(fig)


# ===================================================================== main
def main():
    t0 = time.time()
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log("SYNTHETIC GEOMETRY STUDY  (review R2)   seed=%d" % SEED)
    log("canvas=%dpx  window=%dpx  gsd=%.2f m/px  cap=%.1f m (=%.0f px)  "
        "pool=%d  excl=%d cells" % (IMG, WIN, GSD, CAP_M, CAP_M / GSD, COARSE_F, EXCL))
    log("cv2=%s numpy=%s" % (cv2.__version__, np.__version__))
    log("")
    patterns = make_patterns()

    log("--- E1 first-order accuracy / E2 lambda_min ---")
    e12 = run_e1_e2(patterns, log)
    log("")
    log("--- E3 error direction under corruption ---")
    e3 = run_e3(patterns, log)
    log("")
    log("--- E4 repeated pattern, global ambiguity ---")
    e4 = run_e4(patterns, log)
    log("")

    make_figure(e12, e3, e4, patterns)
    e4_json = {k: v for k, v in e4.items() if not k.startswith("native_grid")}
    blob = dict(
        meta=dict(seed=SEED, img_px=IMG, window_px=WIN, gsd_m_per_px=GSD,
                  cap_m=CAP_M, coarse_factor=COARSE_F, exclusion_radius_cells=EXCL,
                  steps_px=STEPS_PX, n_dirs_e1=N_DIRS_E1,
                  rel_include_frac=REL_INCLUDE_FRAC, rel_tol=REL_TOL,
                  slope_step_px=SLOPE_STEP_PX,
                  e3_trials=E3_TRIALS, e3_min_err_px=E3_MIN_ERR_PX,
                  e3_levels=E3_LEVELS, e3_perm=E3_PERM, e3_boot=E3_BOOT,
                  cv2=cv2.__version__, numpy=np.__version__,
                  cost_source="verbatim copies of run_classical.py:39-67 and "
                              "bench_sketch_vs_roma.py:162-199; capped_P per "
                              "buildc_common.py:280-288",
                  runtime_s=None),
        E1_E2=e12, E3=e3, E4=e4_json)
    blob["meta"]["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "SYNTHETIC_GEOMETRY.json"), "w") as f:
        json.dump(blob, f, indent=1, default=float)
    log("wrote SYNTHETIC_GEOMETRY.json, fig_synthetic.pdf, fig_synthetic.png")
    log("total runtime %.1f s (CPU only)" % (time.time() - t0))
    with open(os.path.join(OUT, "RUN_LOG.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
