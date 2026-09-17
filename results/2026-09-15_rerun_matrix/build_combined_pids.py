"""One balanced UAV-VisLoc population, replacing the n=45 / n=75 split.

Why the split cannot simply be unioned. The two lists overlap on scene 11, so
the union is 119 pairs of which 39, a full third, come from that one scene
purely because it appears in both lists:

    scene      n45   n75   union   available
    visloc02     0    25      25        1071
    visloc03    15     0      15         768
    visloc09    15     0      15         766
    visloc10     0    25      25         144
    visloc11    15    25      39         590

The mixture ratio there is an artifact of two unrelated sampling decisions, not
a design. Reporting a median over it answers a question about a population
nobody defined.

What this builds instead: N_PER_SCENE frames from each of the five scenes, at
uniform stride, one rule for every scene. No overlap, no scene over-represented,
and the sampling states in one sentence.

The stride rule is `pick_pids`' own, `ids[::step][:N]` with
`step = len(ids) // N`, so this is the existing convention applied to five
scenes rather than a new one invented here.

HOW BOTH WERE SAMPLED, since this was got wrong once. In BOTH populations the
SCENES were chosen by hand and the FRAMES within them mechanically: 02/10/11
for the n=75, 03/09/11 for the n=45 (hard-coded as `bench_sketch_vs_roma.SITES`),
with frames then taken at uniform stride, `ids[::step][:N]`. Neither population
has hand-picked frames. The lead confirmed the n=45 scenes were chosen the same
way, 2026-09-15.

So combining them pools two hand-chosen scene sets, not a hand-chosen set with
a mechanical one. The scope sentence in the paper should say five UAV-VisLoc
scenes chosen for little building and low texture, with frames drawn uniformly
inside each, and should NOT claim the scene selection was mechanical.

WHAT THIS REPLACES. Every UAV-VisLoc number currently in the paper is on the
n=75 (scenes 02/10/11). Scenes 03 and 09 appear nowhere in it; the n=45 was
only ever an internal replication and rotation-probe population. Switching to
this list therefore changes every reported UAV-VisLoc figure.

    python results/2026-09-15_rerun_matrix/build_combined_pids.py
"""
import io
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
BENCH = os.path.join(ROOT, "results", "2026-09-08_sketch_vs_roma_bench")
POOL = os.path.join(ROOT, "results", "2026-09-01_buildc50_visloc_final",
                    "POOL_C50_COMBINED.json")

SCENES = ["visloc02", "visloc03", "visloc09", "visloc10", "visloc11"]
N_PER_SCENE = 25
OUT = os.path.join(BENCH, "visloc_combined125_pids.json")


def main():
    pool = json.load(io.open(POOL, encoding="utf-8"))
    pool = pool.get("pairs", pool)

    out = []
    for scene in SCENES:
        ids = sorted(k for k in pool if k.startswith(scene + "_"))
        if len(ids) < N_PER_SCENE:
            sys.exit("FATAL: %s has only %d frames, need %d"
                     % (scene, len(ids), N_PER_SCENE))
        step = max(1, len(ids) // N_PER_SCENE)
        picked = ids[::step][:N_PER_SCENE]
        assert len(picked) == N_PER_SCENE, (scene, len(picked))
        out.extend(picked)

    assert len(out) == len(set(out)), "duplicate pid in the combined list"

    # how much of the existing work this list reuses
    old45 = set(json.load(io.open(os.path.join(BENCH, "_all45_pids.json"),
                                  encoding="utf-8")))
    old75 = set(json.load(io.open(os.path.join(
        BENCH, "visloc_texture75_pids.json"), encoding="utf-8")))
    reused = len(set(out) & (old45 | old75))

    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print("wrote %s" % os.path.relpath(OUT, ROOT))
    print("  %d pairs, %d scenes, %d each, uniform stride"
          % (len(out), len(SCENES), N_PER_SCENE))
    print("  per scene:", dict(Counter(p.split("_")[0] for p in out)))
    print("  duplicates: 0, no scene over-represented")
    print("  %d of %d already appear in the old n45/n75 lists"
          % (reused, len(out)))
    print()
    print("  SCOPE: scenes 02/10/11 were chosen for texture-poverty, 03 and 09")
    print("         were not. Describe this as five UAV-VisLoc scenes, not as")
    print("         a texture-poor subset.")


if __name__ == "__main__":
    main()
