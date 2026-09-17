# A benchmark for sketch-based UAV localisation

Anonymous release for double-blind review. Code and data splits only. No imagery
and no model weights are redistributed here.

This bundle lets a reader check every headline number in the paper without
downloading anything, and sets out what else is needed to rerun the matchers
from raw imagery. Section 2 names every public source that imagery comes from.

Start here:

```
python verify_paper_tables.py
```

That takes under a second, needs only the Python standard library, and prints
every table in the paper recomputed from the per-pair result rows in this
bundle.

---

## 1. What the benchmark is

A drone flying without satellite positioning takes a picture of the ground. The
question is where the drone was. The answer is found by matching that picture
against a map made from satellite or aerial imagery.

The benchmark holds that task fixed and compares matchers on it. Each item in
the benchmark is a **pair**: one downward drone frame, and one crop of an
overhead map that is known to contain the place the frame was taken. A matcher
is given the pair and must return a position. The benchmark records, for every
pair and every matcher:

- whether the matcher returned an answer at all,
- how far that answer was from the reference position, in metres,
- how long it took,
- whatever confidence score the matcher produces.

Two things are kept separate throughout, because they are different failures. A
matcher that declines to answer is not the same as a matcher that answers
wrongly. Both are reported.

Three families of matcher are run on the same pairs:

| family | what it is |
|---|---|
| sketch | draw the boundaries in both images, then slide one drawing over the other and take the best-fitting spot. Three variants of the drawing step: an untrained Canny edge detector, a small network distilled from it, and the same network with an added margin term |
| dense | RoMa v2, a dense image matcher, followed by a rigid fit |
| sparse | XFeat with LighterGlue, a keypoint matcher, followed by a rigid fit |

The sketch matcher needs a heading, that is a guess at which way the drone was
pointing. The dense and sparse matchers do not. Because of that, the
UAV-VisLoc results are reported under three different heading rules, and one of
those rules uses the reference position and is therefore a ceiling rather than a
result. Section 5 says exactly which is which. Getting this wrong is the single
easiest way to misread these files.

## 2. The public sources

Nothing here is redistributed. Everything named below is publicly available from
its own provider under its own terms. Get it there and cite it there.

### The drone side, two datasets

**UAVScenes.** Drone flights with reference positions good to about a metre,
over a town, an airport, a coastal island and a valley. The benchmark uses
five sequences: `amtown01`, `amvalley01`, `hkairport_gnss01`,
`hkairport_gnss_evening` and `hkisland01`. This bundle uses 114 pairs drawn
from them. It also supplies the LiDAR used for the orthorectified render on the
two sequences with relief.

**UAV-VisLoc.** Drone images over Chinese scenes, chosen here for low texture,
which is the case the sketch matcher is meant for. The benchmark uses scenes
02, 03, 09, 10 and 11, 25 pairs from each, 125 in total.

### The map side, and where it differs between the two

**UAV-VisLoc is self-contained.** It ships one satellite GeoTIFF per scene, and
those are the maps the benchmark matches against. Nothing else is needed.

**UAVScenes is not.** It ships no overhead map, so the map side of all 114
UAVScenes pairs is rendered from a public web basemap: **Esri World Imagery**,
through its tile service at

```
https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}
```

Tiles are fetched at the zoom whose ground resolution meets the requested
sampling distance, capped at zoom 18, and cached locally. The `render` field in
the pool file says how those tiles were turned into a map for each pair: `flat`
assumes flat ground, and `ortho` uses the sequence's own LiDAR to orthorectify.

Two things follow, and a reader rebuilding UAVScenes pairs should know both.
Esri World Imagery is served under Esri's own terms of use, which this licence
neither changes nor extends. And a web basemap is refreshed over time, so tiles
fetched later may not be the tiles used here. Record the access date, as the
original run did.

Any other slippy-map tile service with the same `{z}/{x}/{y}` shape can be
substituted, which is the honest way to remove the dependence on one provider.

## 3. How the pairs and the splits are defined

Everything that defines the benchmark is in `data/`.

### `data/uavscenes114_pool_reduced.json`

The 114 UAVScenes pairs. One entry per pair, keyed by pair identifier. That
identifier is the `pid` used in every result row, so a row and a pair definition
can always be joined.

| field | meaning |
|---|---|
| `pair` | pair identifier, also the `pid` in every result row |
| `flight` | the UAVScenes sequence this pair comes from |
| `terrain` | terrain class, one of town, airport, island, valley. This is the grouping used in the per-terrain table |
| `frame_idx` | frame index inside that sequence. With `flight`, this is what resolves the pair against the public dataset |
| `render` | how the map side was drawn. `flat` assumes flat ground. `ortho` uses a LiDAR-orthorectified render, used on the two sequences with relief |
| `label_source` | how the reference position for this pair was obtained |
| `gt_derived` | `true` means the reference position was used to place the pair. It is used to choose and score pairs. It is never given to a matcher |
| `budget_drone`, `budget_win`, `budget_ortho` | ground sampling distance in metres per pixel, for the drone crop, the search window and the map render |

**Pairs were chosen at a uniform stride through the flight, in
reference-position space.** They were not chosen by trying a matcher and keeping
the ones that worked. That matters: a pool selected by matcher success measures
the matcher that selected it.

**What was dropped from this file for release.** Three fields held absolute
paths into a private cache of precomputed crops: `npz_dir`, `prepack_dir`,
`orthos_dir`. They are listed in the file's own `dropped_fields` key. They
pointed at derived `.npz` files, not at anything a third party could use. Every
identifier needed to rebuild a pair from the public dataset, that is `flight`,
`frame_idx`, `render` and the three ground sampling distances, is kept.

### `data/uavscenes114_split.json`

The fixed split of those 114 pairs: 85 train, 17 validation, 12 test. Plain
lists of pair identifiers.

The split exists because one of the drawing variants is trained. Read it this
way, and the paper says so too: **a split inside each flight is not a test on a
new flight.** The 29 pairs outside training are held out within the same
flights. A genuinely unseen-flight result is not in this benchmark.

The untrained Canny variant and both the dense and sparse matchers are trained
on nothing, so for those three the split does not apply and all 114 pairs are
equally held out.

### `data/visloc125_pids.json`

The 125 UAV-VisLoc pair identifiers, a plain list. Twenty five per scene, taken
at a uniform stride, from scenes 02, 03, 09, 10 and 11. `build_combined_pids.py`
is the script that built this list.

No training split is defined on UAV-VisLoc. Every UAV-VisLoc pair is held out
for every matcher.

## 4. What is in the bundle

```
verify_paper_tables.py            recompute every table, no downloads needed
SCRUB_REPORT.md                   what was removed from these files before release
LICENSE                           MIT

data/
  uavscenes114_pool_reduced.json  the 114 UAVScenes pairs
  uavscenes114_split.json         85 / 17 / 12
  visloc125_pids.json             the 125 UAV-VisLoc pairs

results/2026-09-15_rerun_matrix/
  run_matrix_sketch.py            driver, sketch matcher, all three drawing variants
  run_matrix_roma.py              driver, dense matcher
  run_matrix_xfeat.py             driver, sparse matcher
  build_combined_pids.py          builds the 125-pair UAV-VisLoc list
  gated_comparison.py             confidence score against error, both populations
  GATED_COMPARISON.txt            its output
  matrix_*.jsonl                  one row per pair per matcher, 21 files

results/2026-09-15_paper_figures/
  make_figures_2026_09_15.py      the three figures, and the numbers in their captions
  FIGURE_NUMBERS_2026-09-15.json  those numbers

results/2026-09-15_boundary_geometry/
  boundary_geometry.py            boundary structure tensor G per pair

results/2026-09-15_directional_geometry/
  directional_geometry.py         which way an error points against boundary geometry
  pairs_cache.jsonl               its per-pair output, 114 rows
  r14_ci_blockperm.py             confidence intervals and a within-scene permutation test
  R14_CI_BLOCKPERM.txt            its output

results/2026-09-16_r9_inwindow_recount/
  r9_recount.py                   recount with both matchers held to the same allowed area
  R9_RECOUNT.md, R9_RECOUNT.json  its output

results/2026-09-16_r10_matched_coverage/
  r10_matched_coverage.py         wrong answers admitted at matched coverage
  R10_MATCHED_COVERAGE.md/.json   its output

results/2026-09-16_synthetic_geometry/
  synthetic_geometry.py           synthetic study of the local-geometry claim
  SYNTHETIC_GEOMETRY.json         its output
  RUN_LOG.txt                     its console log
```

## 5. The per-pair result rows

`results/2026-09-15_rerun_matrix/matrix_*.jsonl`. One JSON object per line, one
line per pair. These are where every number in the paper comes from.

The file name says what the row is: `matrix_<arm>_<population>[_<heading rule>][_cc1][_gate].jsonl`.

- `<arm>` is `canny`, `distill`, `learned`, `roma` or `xfeat`
- `<population>` is `uavscenes` (114 pairs) or `visloc_combined125` (125 pairs)
- `shipped` means the heading the dataset supplies. `oracle` means the heading
  was chosen using the reference position
- `cc1` means a connected-component filter was applied to all three drawing
  variants alike, so the comparison between them is about training and not about
  filtering
- `gate` means the confidence score was recorded in the same pass

### Which file to read

| file | status |
|---|---|
| the six `*_cc1_gate.jsonl` | **current.** Read these for the sketch arms |
| `matrix_roma_uavscenes.jsonl`, `matrix_roma_visloc_combined125.jsonl` | **current.** The dense matcher |
| `matrix_xfeat_visloc_combined125.jsonl` | **current.** The sparse matcher |
| the six `*_cc1.jsonl` without `_gate` | earlier pass, same settings, no confidence score recorded |
| the six plain files without `cc1` | earlier pass, before the filter was equalised across drawing variants. `gated_comparison.py` reads these. They also carry rows from a first attempt that failed on a missing file, marked in `exclude_reason` |

Two files that existed alongside these were deliberately left out: they were run
on an earlier 75-pair UAV-VisLoc scene list that the 125-pair list replaced.
Keeping them would invite a reader to recompute a retired number. `SCRUB_REPORT.md`
lists them.

### Field dictionary

Common to all arms:

| field | meaning |
|---|---|
| `pid` | pair identifier, joins to `data/` |
| `arm` | which matcher |
| `pop` | `uavscenes` or `visloc` |
| `answered` | did the matcher return a position at all |
| `exclude_reason` | why not, when it did not |
| `region_m` | side of the search window in metres |
| `eff` | ground sampling distance of the search window, metres per pixel |
| `grid_shape` | size of the cost grid searched |
| `jitter` | how the window centre was placed. `uniform` means the offset was drawn uniformly over the full half-width of the window, from a fixed seed, so the truth is equally likely anywhere inside |
| `t_ms` | wall time for this pair in milliseconds, on the machine named in the paper. It is not a claim about other hardware |

Sketch arms:

| field | meaning |
|---|---|
| `shipped_angle_deg`, `shipped_err_m` | heading supplied by the dataset, and the error at it |
| `cost_angle_deg`, `cost_err_m` | heading picked by lowest matching cost, and the error at it. This rule uses no reference position |
| `oracle_diag_angle_deg`, `oracle_diag_err_m` | heading picked using the reference position, and the error at it. **See the warning below** |
| `center_hit_err_m` | error you get by returning the window centre and doing no matching at all. The baseline every result must beat |
| `peak_ratio` | confidence score. Best cost divided by the best cost outside a neighbourhood of it. Larger means a more decisive match |
| `gate_angle_deg`, `gate_threshold`, `answered_gated` | the heading the score was measured at, the fixed threshold, and whether the pair passes it |
| `sweep_cost_std` | spread of the cost over the heading sweep |
| `cc_filter` | whether the connected-component filter was on |

Dense and sparse arms nest their result under `roma` or `xfeat_lg`:

| field | meaning |
|---|---|
| `E`, `N` | predicted offset in metres, east and north |
| `err_m` | distance from the reference position |
| `n_conf` | correspondences above the confidence cutoff, before fitting |
| `n_inliers` | correspondences the rigid fit kept. Also used as the confidence score |
| `n_matches`, `n_kp_frame`, `n_kp_map` | sparse arm only: matches found, and keypoints detected in each image |
| `cuda_peak_gb` | peak GPU memory for this pair, on the machine named in the paper |
| `crop`, `roma_crop` | the crop handed to the matcher, so both families can be shown to have seen the same pixels |

### Two flags that decide whether a row may be quoted

**`gt_in_estimator`.** `true` means the reference position was used inside the
estimate, not only to score it. Every `oracle` row is like this: 72 headings were
tried and the one whose answer landed closest to the truth was kept. That is a
ceiling on what any heading source could achieve. It is a diagnostic. It is not
a result, and presenting it as a satellite-free result would be wrong.

**`citable`.** `false` on exactly the rows where `gt_in_estimator` is `true`.

The `oracle_diag_note` field repeats the warning in the row itself, so it
travels with the data.

Rows on UAVScenes carry `gt_in_estimator: false`. The heading there is read off
the drone's own motion, and those rows are the satellite-free ones.

Separately, `gt_derived: true` in the pool file means the reference position was
used to **choose and place** the pair. That is fine and is not the same thing.
Selection uses the truth; the estimator never does.

## 6. How to run each script

### Group one: nothing to download

These read only the files in this bundle. Python 3, plus `numpy` for two of
them, plus `matplotlib` and `scipy` for two more.

| command | what it prints or writes |
|---|---|
| `python verify_paper_tables.py` | every headline table, recomputed. Standard library only |
| `cd results/2026-09-15_rerun_matrix && python gated_comparison.py` | how many pairs each confidence score admits at each threshold, and the error among them, both populations |
| `cd results/2026-09-15_paper_figures && python make_figures_2026_09_15.py` | the three figures as PDF and PNG, and `FIGURE_NUMBERS_2026-09-15.json` with the numbers quoted in their captions |
| `cd results/2026-09-16_r10_matched_coverage && python r10_matched_coverage.py` | wrong answers admitted when both matchers are held to the same coverage, thresholds chosen leave-one-scene-out. Writes `.md` and `.json` |
| `cd results/2026-09-15_directional_geometry && python r14_ci_blockperm.py` | bootstrap confidence intervals on the mean error direction, and a permutation test that only re-pairs within a scene |
| `cd results/2026-09-16_synthetic_geometry && python synthetic_geometry.py` | the synthetic study. CPU only, seeded, about one minute. Writes `SYNTHETIC_GEOMETRY.json` and a figure |

Requirements for group one: `numpy` for `r14_ci_blockperm.py`,
`numpy` + `matplotlib` for `make_figures_2026_09_15.py`, and
`numpy` + `opencv-python` + `scipy` + `matplotlib` for `synthetic_geometry.py`.
Nothing else, and no GPU.

### Group two: needs the datasets

These read the drone and map imagery, so they need the sources in section 2
downloaded and the per-pair crops built. For anything touching UAVScenes that
means the basemap tiles as well as the dataset.

| script | what it needs beyond this bundle |
|---|---|
| `run_matrix_sketch.py` | both datasets, the basemap tiles for UAVScenes, and the matcher library. For the two trained drawing variants it also needs their weights, which are not redistributed here. The untrained Canny variant needs no weights |
| `run_matrix_roma.py` | the same imagery, and RoMa v2, which is public and has its own repository and licence |
| `run_matrix_xfeat.py` | UAV-VisLoc, and XFeat with LighterGlue, both public with their own licences |
| `boundary_geometry.py` | the UAVScenes crops, to measure boundary structure per pair |
| `directional_geometry.py` | the UAVScenes crops. Its per-pair output `pairs_cache.jsonl` **is** in this bundle, so the analysis on top of it runs without the imagery |
| `r9_recount.py` | the per-pair packed crops, to rebuild each search box |

**Be plain about the limit here.** The drivers in this bundle are the exact
record of how each matcher was run: every setting, every threshold, every
random seed, the full order of operations. They are not a self-contained
program. They import a matcher library that pulls in a larger package, and that
package is not in this release. What this bundle guarantees is that every number
in the paper can be recomputed from the per-pair rows included, which group one
does. Rerunning the matchers themselves from raw imagery needs more than is
here.

### Environment variables

`run_matrix_sketch.py` and `run_matrix_roma.py` are configured by environment
variables, all beginning `BENCH_`. The ones that matter:

| variable | values |
|---|---|
| `BENCH_ARM` | `canny`, `distill`, `learned` |
| `BENCH_POP` | `uavscenes`, `visloc` |
| `BENCH_ROT` | `shipped`, `oracle`. The driver refuses `oracle` on UAVScenes on purpose, because it would put the reference position inside the one arm that is otherwise clean |
| `BENCH_CC_FILTER` | `1` (default) or `0` |
| `BENCH_PIDS` | a pair identifier list to run instead of the default |
| `BENCH_CKPT_LEARNED`, `BENCH_CKPT_DISTILL` | weight files for the two trained variants |
| `BENCH_ROMA_CROP` | `shared` (default) or `footprint` |

These names were renamed for this release. `SCRUB_REPORT.md` says so.

## 7. Where each table and figure in the paper comes from

| in the paper | script | file it reads |
|---|---|---|
| Answer rate and error by terrain, 114 pairs | `verify_paper_tables.py`, and `make_figures_2026_09_15.py` for the caption numbers | `matrix_canny_uavscenes_shipped_cc1_gate.jsonl`, `matrix_roma_uavscenes.jsonl` |
| The 81 pairs both matchers answer | `verify_paper_tables.py` | same two files |
| Figure: share answered and not answered, by terrain | `make_figures_2026_09_15.py` | same two files |
| Figure: per-pair error, dense against sketch | `make_figures_2026_09_15.py` | same two files |
| Figure: coverage against accuracy for the confidence score | `make_figures_2026_09_15.py`, and `r10_matched_coverage.py` for the leave-one-scene-out version | the three `*_uavscenes_shipped_cc1_gate.jsonl`, `matrix_roma_uavscenes.jsonl` |
| Area under the curve for spotting an error above 30 m | `gated_comparison.py`, `r10_matched_coverage.py` | the `_gate` files and the two dense files |
| Drawing-step ablation, 125 UAV-VisLoc pairs | `verify_paper_tables.py` | the three `*_visloc_combined125_oracle_cc1_gate.jsonl` |
| Median error under the three heading rules, and the window-centre baseline | `verify_paper_tables.py` | `matrix_canny_visloc_combined125_oracle_cc1_gate.jsonl` |
| Sparse keypoint matcher on 125 pairs | `verify_paper_tables.py` | `matrix_xfeat_visloc_combined125.jsonl` |
| Both matchers held to the same allowed area | `r9_recount.py` | `R9_RECOUNT.md` is the output, included |
| Error direction against boundary geometry, with intervals | `r14_ci_blockperm.py` | `pairs_cache.jsonl` |
| Synthetic study of the local-geometry claim | `synthetic_geometry.py` | nothing, it generates its own examples |
| Qualitative pair figure | not reproducible from this bundle | needs the imagery |

## 8. Licence

MIT, see `LICENSE`. The copyright line reads "Anonymous authors" because this is
the review copy. The final version will carry the real names.

MIT covers the code and the split and pool files in this bundle, and nothing
else. It does not cover, and cannot grant anything about:

- **UAVScenes** and **UAV-VisLoc**, which stay under their own licences. Get
  them from their own authors and cite those authors
- **Esri World Imagery**, the basemap the UAVScenes map side is rendered from,
  which is served under Esri's own terms of use
- **RoMa v2**, **XFeat** and **LighterGlue**, each public with its own
  repository and its own licence

## 9. Two honest limits, stated once

**The split is within flights, not across them.** The 29 held-out pairs come
from the same five sequences as the 85 training pairs. Results on them do not
show transfer to a new flight.

**The oracle heading is a ceiling.** Every UAV-VisLoc row marked
`gt_in_estimator: true` had its heading chosen using the reference position.
That number bounds what a perfect heading source could deliver. It is not what
the system does.
