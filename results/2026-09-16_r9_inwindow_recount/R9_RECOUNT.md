# R9 in-window recount of the RUN 032 matrix (RoMa v2 vs sketch)

Answers faculty comment R9 (`docs/AMA_REVIEW_2026-09-16.md`): a RoMa v2 answer that falls outside the sketch matcher's reachable position box is counted as unanswered, and the head-to-head and per-terrain tables are re-derived on that basis. No matcher was re-run. CPU only, no GPU. Writer: `r9_recount.py`; per-pair records: `R9_RECOUNT.json`.

Facts and numbers only. No verdict is drawn here.

## The box

Same derivation as `../2026-09-15_equal_position_range/` (which did it on the older 120 px-jitter table). `rotation_probe.prep_pid` places the truth at `true_c = (off[0] - x0)/8`, `true_r = (off[1] - y0)/8` on the sketch's cost grid of shape `grid_shape`; the sketch returns one cell, so its reachable set is a lattice of `ow x oh` positions at a pitch of `8 * eff` m. The sketch estimate is a pure translation, so the displacement set is the same for the principal point the dense arm scores; in RoMa's coordinates the bounding box is

```
E in [tE - true_c*cell, tE + (ow-1-true_c)*cell]
N in [tN - (oh-1-true_r)*cell, tN + true_r*cell]     cell = 8*eff, (tE,tN) = roma.true_EN
```

The crop `x0, y0` is taken from each RoMa row. Both runners draw the crop with `random.Random(B._stable_pid_seed(pid))` and `jitter_px = SEARCH_MARGIN_M / eff` (uniform over the full half-width), so the crop is the same in both arms; check 2 below tests that rather than assuming it. The in/out test is on the axis-aligned box (`E_min <= E <= E_max AND N_min <= N <= N_max`), not a radius.

**UAVScenes** geometry (`off`, `eff`, `x0y0_big`, `anchor_EN`) comes from `../2026-09-09_rtk_uniform_pool_pilot/prepack{,_ortho}/<pid>.npz`.

**UAV-VisLoc**: only 5 of the 125 pids have a local prepack (`../2026-09-08_sketch_vs_roma_bench/visloc_bench_data/`). The VisLoc writer (`../2026-08-26_uav_visloc_adapt/a dataset build script not in this release` lines 558-559, 585, 588) defines `off = (int(round(anchor_x)), int(round(anchor_y)))`, `x0y0_big = (0, 0)` and `anchor_EN = (anchor_x*G, -anchor_y*G)`, so `off = (round(tE/eff), round(-tN/eff))` is recovered exactly from each RoMa row's own `true_EN` and `eff`. That reconstruction is checked against the 5 local prepacks (check 4) and against the sketch row's `center_hit_err_m` on every row that carries `true_EN` (check 2). Rows with no `true_EN` (RoMa `fit_failed`) have no box and need none: there is no answer to classify.

## uavscenes

Sketch arm: sketch shipped heading (citable) (`shipped_err_m`).

### Sanity checks

| check | result |
|---|---|
| rows / rows with a box / without | 114 / 114 / 0 |
| RoMa answered rows with no derivable box | 0 |
| (1) truth inside the box | 114 inside, 0 outside |
| (2) recomputed center-hit vs sketch row `center_hit_err_m` | max abs diff 0.0497 m over 114 rows (stored to 0.1 m) |
| (3) RoMa `err_m` recomputed from stored `E,N,true_EN` | max abs diff 0.0512 m |
| `roma.true_EN` vs prepack `anchor_EN` | max abs diff 0.0000 m |
| `roma.x0y0_big` vs prepack `x0y0_big` | max abs diff 0.0000 px |
| eff values | [0.2] |
| grid shapes | ['76x76'] |
| geometry sources | {"prepack_prepack": 69, "prepack_prepack_ortho": 45} |

### Counts and medians

| scene | n | RoMa no-fit | RoMa answered | inside box | outside box | RoMa median, answered (m) | RoMa median, inside only (m) | sketch median, all pairs (m) |
|---|---|---|---|---|---|---|---|---|
| amtown01 | 33 | 17 | 16 | 6 | 10 | 194.8 | 1.8 | 2.2 |
| amvalley01 | 29 | 3 | 26 | 15 | 11 | 4.4 | 3.1 | 5.2 |
| hkairport | 36 | 8 | 28 | 21 | 7 | 2.2 | 1.6 | 18.3 |
| hkisland01 | 16 | 5 | 11 | 8 | 3 | 6.5 | 3.6 | 3.2 |
| all | 114 | 33 | 81 | 50 | 31 | 3.9 | 2.1 | 5.0 |

Coverage accompanies every median: read each inside-only median with its `inside box` count.

### RoMa answers outside the box

| scene | outside | error median / min / max (m) | distance beyond the box edge median / min / max (m) | smallest possible error outside the box, `d_edge` min / median (m) |
|---|---|---|---|---|
| amtown01 | 10 | 218.8 / 128.3 / 358.3 | 151.6 / 35.5 / 211.3 | 0.8 / 17.4 |
| amvalley01 | 11 | 138.1 / 4.1 / 271.6 | 81.2 / 2.3 / 214.1 | 0.0 / 18.8 |
| hkairport | 7 | 122.6 / 2.9 / 230.1 | 73.2 / 2.5 / 112.3 | 0.0 / 11.7 |
| hkisland01 | 3 | 111.6 / 60.4 / 165.1 | 53.2 / 24.5 / 68.3 | 1.8 / 9.3 |
| all | 31 | 166.1 / 2.9 / 358.3 | 85.1 / 2.3 / 214.1 | 0.0 / 13.5 |

Inside-box RoMa errors run 0.0 to 78.6 m over 50 answers.

Unlike the older 120 px-jitter table, where the truth was never closer than 36 m to a box edge, the matrix run's uniform jitter over the full half-width lets the truth sit anywhere in the box: `d_edge` (the smallest error any outside position can have) is min 0.0 m, median 13.5 m, max 51.2 m here, and the box centre sits `center_hit_err_m` median 51.4 m, max 81.8 m from truth. So an outside answer is not necessarily a large error: 7 of 31 outside answers have an error below the inside-set maximum, 4 are below 10 m and 4 below 20 m. Box span per axis: 120.0 to 120.0 m (grid shapes 76x76; `eff` per pair 0.2 to 0.2 m/px).

### Head-to-head, pairs where both arms answer (tie = within 0.1 m)

| scene | unrestricted: n, RoMa closer / sketch closer / tie | RoMa-inside only: n, RoMa closer / sketch closer / tie | moved out (RoMa was closer / sketch was closer) | RoMa median, inside set (m) | sketch median, inside set (m) |
|---|---|---|---|---|---|
| amtown01 | 16: 3 / 12 / 1 | 6: 3 / 2 / 1 | 10 (0 / 10) | 1.8 | 2.4 |
| amvalley01 | 26: 14 / 12 / 0 | 15: 12 / 3 / 0 | 11 (2 / 9) | 3.1 | 4.6 |
| hkairport | 28: 19 / 7 / 2 | 21: 17 / 2 / 2 | 7 (2 / 5) | 1.6 | 2.6 |
| hkisland01 | 11: 3 / 7 / 1 | 8: 3 / 4 / 1 | 3 (0 / 3) | 3.6 | 4.5 |
| all | 81: 39 / 38 / 4 | 50: 35 / 11 / 4 | 31 (4 / 27) | 2.1 | 3.3 |

On the 31 moved pairs the sketch errors run 0.7 to 109.5 m; the sketch answered 114 of 114 pairs (31 after its peak-ratio gate, not used here).

## visloc

Sketch arm: sketch ORACLE angle: GT-SELECTED CEILING, gt_in_estimator=true, not a system result (`oracle_diag_err_m`).

### Sanity checks

| check | result |
|---|---|
| rows / rows with a box / without | 125 / 101 / 24 |
| RoMa answered rows with no derivable box | 0 |
| (1) truth inside the box | 101 inside, 0 outside |
| (2) recomputed center-hit vs sketch row `center_hit_err_m` | max abs diff 0.0499 m over 101 rows (stored to 0.1 m) |
| (3) RoMa `err_m` recomputed from stored `E,N,true_EN` | max abs diff 0.0501 m |
| (4) reconstructed `off` vs local prepack `off` | max abs diff 0.0000 px over 5 local prepacks |
| (4) `roma.true_EN` vs local `pairs` `anchor_EN` | max abs diff 0.0000 m over 5 local prepacks |
| eff values | [0.2, 0.2021, 0.20448, 0.20598, 0.20704, 0.2117, 0.21514, 0.21819, 0.21881, 0.22178, 0.22545, 0.22929, 0.23059, 0.2311, 0.23121, 0.23212, 0.23225, 0.2375, 0.23772, 0.23893, 0.24214, 0.24259] |
| grid shapes | ['63x63', '64x64', '65x65', '66x66', '67x67', '69x69', '70x70', '71x71', '72x72', '73x73', '74x74', '75x75', '76x76'] |
| geometry sources | {"off_from_row_trueEN": 96, "none": 24, "off_from_row_trueEN+local_prepack_verified": 5} |

### Counts and medians

| site | n | RoMa no-fit | RoMa answered | inside box | outside box | RoMa median, answered (m) | RoMa median, inside only (m) | sketch median, all pairs (m) |
|---|---|---|---|---|---|---|---|---|
| visloc02 | 25 | 10 | 15 | 4 | 11 | 147.3 | 24.4 | 21.0 |
| visloc03 | 25 | 0 | 25 | 15 | 10 | 28.1 | 19.7 | 15.8 |
| visloc09 | 25 | 3 | 22 | 6 | 16 | 166.6 | 26.6 | 26.9 |
| visloc10 | 25 | 8 | 17 | 3 | 14 | 157.2 | 15.2 | 23.8 |
| visloc11 | 25 | 3 | 22 | 12 | 10 | 39.4 | 33.1 | 21.8 |
| all | 125 | 24 | 101 | 40 | 61 | 63.9 | 23.2 | 21.8 |

Coverage accompanies every median: read each inside-only median with its `inside box` count.

### RoMa answers outside the box

| site | outside | error median / min / max (m) | distance beyond the box edge median / min / max (m) | smallest possible error outside the box, `d_edge` min / median (m) |
|---|---|---|---|---|
| visloc02 | 11 | 244.4 / 35.7 / 503.0 | 159.9 / 2.0 / 442.3 | 3.0 / 31.2 |
| visloc03 | 10 | 46.0 / 11.5 / 472.6 | 12.6 / 0.3 / 458.9 | 0.6 / 31.2 |
| visloc09 | 16 | 230.4 / 33.4 / 646.7 | 125.2 / 10.3 / 521.2 | 2.6 / 31.2 |
| visloc10 | 14 | 285.6 / 21.0 / 613.3 | 204.0 / 12.7 / 489.3 | 2.0 / 28.0 |
| visloc11 | 10 | 319.8 / 33.8 / 581.0 | 271.3 / 1.4 / 451.4 | 2.1 / 29.6 |
| all | 61 | 223.0 / 11.5 / 646.7 | 115.5 / 0.3 / 521.2 | 0.6 / 31.0 |

Inside-box RoMa errors run 1.0 to 84.5 m over 40 answers.

Unlike the older 120 px-jitter table, where the truth was never closer than 36 m to a box edge, the matrix run's uniform jitter over the full half-width lets the truth sit anywhere in the box: `d_edge` (the smallest error any outside position can have) is min 0.6 m, median 31.0 m, max 54.4 m here, and the box centre sits `center_hit_err_m` median 34.8 m, max 74.9 m from truth. So an outside answer is not necessarily a large error: 16 of 61 outside answers have an error below the inside-set maximum, 0 are below 10 m and 1 below 20 m. Box span per axis: 118.8 to 121.6 m (grid shapes 63x63, 64x64, 65x65, 66x66, 67x67, 69x69, 70x70, 71x71, 72x72, 73x73, 74x74, 75x75, 76x76; `eff` per pair 0.2 to 0.24259 m/px).

### Head-to-head, pairs where both arms answer (tie = within 0.1 m)

The sketch arm here is the oracle-angle ceiling (`oracle_diag_err_m`, gt_in_estimator=true, citable=false). The counts in the previous table (answered / inside / outside) do not touch the sketch arm and stand on their own.

| site | unrestricted: n, RoMa closer / sketch closer / tie | RoMa-inside only: n, RoMa closer / sketch closer / tie | moved out (RoMa was closer / sketch was closer) | RoMa median, inside set (m) | sketch median, inside set (m) |
|---|---|---|---|---|---|
| visloc02 | 15: 2 / 13 / 0 | 4: 2 / 2 / 0 | 11 (0 / 11) | 24.4 | 35.6 |
| visloc03 | 25: 10 / 15 / 0 | 15: 8 / 7 / 0 | 10 (2 / 8) | 19.7 | 15.8 |
| visloc09 | 22: 2 / 20 / 0 | 6: 1 / 5 / 0 | 16 (1 / 15) | 26.6 | 18.9 |
| visloc10 | 17: 3 / 14 / 0 | 3: 2 / 1 / 0 | 14 (1 / 13) | 15.2 | 35.3 |
| visloc11 | 22: 4 / 18 / 0 | 12: 4 / 8 / 0 | 10 (0 / 10) | 33.1 | 13.8 |
| all | 101: 21 / 80 / 0 | 40: 17 / 23 / 0 | 61 (4 / 57) | 23.2 | 16.1 |

On the 61 moved pairs the sketch errors run 1.1 to 93.2 m; the sketch answered 125 of 125 pairs (0 after its peak-ratio gate, not used here).

## Citability

No `run.json` exists for this reanalysis. Per the project record-keeping policy, nothing here is citable until that record exists. The VisLoc sketch arm is the oracle ceiling and is not citable in any case.
