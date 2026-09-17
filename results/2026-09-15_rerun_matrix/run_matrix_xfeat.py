"""XFeat + LighterGlue on the combined 125, the arm the matrix left out.

The rerun matrix covered Canny, distillation-only, full training and RoMa v2,
but not XFeat, whose only UAV-VisLoc records are on the OLD populations:
`round6_xfeat_visloc_n45_HPC.jsonl` (0 of 45 answered) and
`round6_xfeat_visloc_n75_HPC.jsonl` (2 of 75). Quoting those beside 125-pair
numbers would put three matchers on two different populations in one sentence,
which is the blur faculty comment 5 objects to.

Nothing is reimplemented here. `round6_xfeat_visloc_hpc.main()` drives entirely
off its own `POPULATIONS` list, so this swaps that list for the combined 125 and
calls it. Model loading, the vendored XFeat import, the kornia check, the
keypoint budget, the fit and the resume logic are all that module's own code.

Cheap: XFeat ran at 21.5 ms per pair on the n=75 population, so 125 pairs is a
few seconds of compute. The wall time is model loading, not matching.

    python run_matrix_xfeat.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
BENCH = os.path.join(ROOT, "results", "2026-09-08_sketch_vs_roma_bench")

sys.path.insert(0, BENCH)
os.chdir(BENCH)

import round6_xfeat_visloc_hpc as X   # noqa: E402

# One population, written into the matrix directory beside the other cells.
# The module resolves pids_path relative to its OWN directory (BENCH), which is
# where visloc_combined125_pids.json lives, so the name alone is correct here.
X.POPULATIONS = [
    ("combined125",
     "visloc_combined125_pids.json",
     os.path.join(HERE, "matrix_xfeat_visloc_combined125.jsonl")),
]

if __name__ == "__main__":
    print("XFeat + LighterGlue, combined 125 (5 scenes x 25)", flush=True)
    print("output:", X.POPULATIONS[0][2], flush=True)
    X.main()
