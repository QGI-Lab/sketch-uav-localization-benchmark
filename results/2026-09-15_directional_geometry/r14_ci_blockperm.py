"""R14 (faculty round 3): bootstrap CIs on the mean folded angle, and a
permutation test that re-pairs error directions with weak directions only
WITHIN the same scene (block permutation), so flight-level dependence cannot
manufacture the alignment. Reads pairs_cache.jsonl of RUN 034. No ground truth
beyond the error vector already in that cache.
    python r14_ci_blockperm.py > R14_CI_BLOCKPERM.txt
"""
import json, numpy as np
rng = np.random.default_rng(20260916)
recs = [json.loads(l) for l in open('pairs_cache.jsonl', encoding='utf-8')]
err = np.array([r['err_m'] for r in recs]); scene = np.array([r['scene'] for r in recs])
ed = np.array([r['err_dir_deg'] for r in recs])

def fold(a, b): return np.abs((a - b + 90.0) % 180.0 - 90.0)
subsets = [('all', err > -1), ('err > 5 m', err > 5), ('5 < err <= 30 m', (err > 5) & (err <= 30)), ('err > 30 m', err > 30)]
for vname, wkey in [('v_min(G)', 'vmin_deg'), ('L1 minimiser', 'l1_weak_deg')]:
    wd = np.array([r[wkey] for r in recs])
    th = fold(ed, wd)
    print('variant:', vname)
    print('  %-18s %4s %6s %16s %8s %10s %10s' % ('subset', 'n', 'mean', '95% CI (boot)', '<45', 'p_perm', 'p_block'))
    for name, m in subsets:
        t = th[m]; n = len(t)
        boot = [rng.choice(t, n).mean() for _ in range(10000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        obs = t.mean(); e_s, w_s, sc = ed[m], wd[m], scene[m]
        # pooled permutation (as in RUN 034) and within-scene block permutation
        P = 10000; cnt_pool = 0; cnt_block = 0
        idx = np.arange(n)
        blocks = [np.where(sc == s)[0] for s in sorted(set(sc))]
        for _ in range(P):
            p = rng.permutation(idx)
            if fold(e_s, w_s[p]).mean() <= obs: cnt_pool += 1
            pb = idx.copy()
            for b in blocks: pb[b] = b[rng.permutation(len(b))]
            if fold(e_s, w_s[pb]).mean() <= obs: cnt_block += 1
        print('  %-18s %4d %6.1f [%5.1f, %5.1f]      %5.2f %10.4f %10.4f' % (name, n, obs, lo, hi, (t < 45).mean(), (cnt_pool + 1) / (P + 1), (cnt_block + 1) / (P + 1)))
    print('  scenes in 5<err<=30:', {s: int(((err > 5) & (err <= 30) & (scene == s)).sum()) for s in sorted(set(scene))})
