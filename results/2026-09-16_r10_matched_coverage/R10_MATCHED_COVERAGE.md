# R10: wrong admissions at matched coverage, thresholds chosen on other pairs

Source rows: RUN 032, `results/2026-09-15_rerun_matrix/` (no matcher re-run). Script: `r10_matched_coverage.py`; full numbers, per-fold thresholds and admitted pids: `R10_MATCHED_COVERAGE.json`.

**Rule.** A pair is admitted when it was answered and its score is at or above the threshold. The threshold is the lowest admitted score on the calibration pairs that gives the largest admitted share at or below the target (closest from below; a tie that would overshoot is dropped). Unanswered pairs (sketch: no answer; RoMa: no fit, `n_inliers` 0) are never admitted. Wrong = error above 30 m. Share is of the full population (114 UAVScenes pairs, 125 UAV-VisLoc pairs), not of the answered pairs.

**Error fields.** UAVScenes: sketch `shipped_err_m`, RoMa `roma.err_m`. UAV-VisLoc: sketch `oracle_diag_err_m`, the heading ground truth selected (a labelled ceiling, `gt_in_estimator: true`, not a system result; the `peak_ratio` in those rows was recorded at that same heading, `gate_angle_deg == oracle_diag_angle_deg` on all 125 rows); RoMa `roma.err_m`.

**Scenes.** UAVScenes: amtown01 33, amvalley01 29, hkairport 36, hkisland01 16. UAV-VisLoc: visloc02, 03, 09, 10, 11 with 25 each.

Columns: admitted | share of population % | wrong (>30 m) | median error of admitted, m | max error of admitted, m.

## 1. Leave-one-scene-out (threshold chosen on the other scenes, applied unchanged to the held-out scene, pooled)

### UAVScenes, 114 pairs

| score | target % | admitted | share % | wrong >30 m | median m | max m |
|---|---|---|---|---|---|---|
| sketch peak_ratio (Canny) | 10 | 10 | 8.8 | 0 | 1.8 | 5.9 |
| sketch peak_ratio (Canny) | 20 | 26 | 22.8 | 2 | 1.8 | 96.1 |
| sketch peak_ratio (Canny) | 30 | 32 | 28.1 | 2 | 1.6 | 96.1 |
| RoMa v2 n_inliers | 10 | 13 | 11.4 | 0 | 0.6 | 2.2 |
| RoMa v2 n_inliers | 20 | 22 | 19.3 | 0 | 0.8 | 4.7 |
| RoMa v2 n_inliers | 30 | 34 | 29.8 | 0 | 1.8 | 8.2 |
| sketch peak_ratio (distilled drawer) | 10 | 15 | 13.2 | 0 | 1.0 | 22.3 |
| sketch peak_ratio (distilled drawer) | 20 | 19 | 16.7 | 1 | 2.1 | 105.0 |
| sketch peak_ratio (distilled drawer) | 30 | 34 | 29.8 | 4 | 2.1 | 115.5 |
| sketch peak_ratio (full drawer) | 10 | 10 | 8.8 | 0 | 1.2 | 6.3 |
| sketch peak_ratio (full drawer) | 20 | 24 | 21.1 | 1 | 1.3 | 104.9 |
| sketch peak_ratio (full drawer) | 30 | 35 | 30.7 | 4 | 1.4 | 104.9 |

### UAV-VisLoc, 125 pairs

| score | target % | admitted | share % | wrong >30 m | median m | max m |
|---|---|---|---|---|---|---|
| sketch peak_ratio (Canny) | 10 | 12 | 9.6 | 5 | 20.4 | 82.1 |
| sketch peak_ratio (Canny) | 20 | 25 | 20.0 | 12 | 21.0 | 89.5 |
| sketch peak_ratio (Canny) | 30 | 37 | 29.6 | 16 | 21.8 | 89.5 |
| RoMa v2 n_inliers | 10 | 14 | 11.2 | 6 | 29.5 | 76.8 |
| RoMa v2 n_inliers | 20 | 30 | 24.0 | 13 | 29.0 | 123.5 |
| RoMa v2 n_inliers | 30 | 42 | 33.6 | 18 | 29.0 | 123.5 |
| sketch peak_ratio (distilled drawer) | 10 | 10 | 8.0 | 5 | 28.6 | 62.1 |
| sketch peak_ratio (distilled drawer) | 20 | 25 | 20.0 | 10 | 21.6 | 74.0 |
| sketch peak_ratio (distilled drawer) | 30 | 35 | 28.0 | 15 | 25.9 | 85.2 |
| sketch peak_ratio (full drawer) | 10 | 12 | 9.6 | 5 | 26.8 | 42.9 |
| sketch peak_ratio (full drawer) | 20 | 23 | 18.4 | 10 | 27.6 | 65.9 |
| sketch peak_ratio (full drawer) | 30 | 36 | 28.8 | 13 | 24.9 | 97.5 |

### Per-fold detail at the 20 % target, primary scores

| population | score | held-out scene | threshold (chosen on the other scenes) | admitted on calibration | held-out admitted / pairs | held-out wrong >30 m | held-out median m |
|---|---|---|---|---|---|---|---|
| uavscenes | sketch peak_ratio (Canny) | amtown01 | 1.1111 | 16 / 81 (19.8%) | 13 / 33 | 2 | 1.6 |
| uavscenes | sketch peak_ratio (Canny) | amvalley01 | 1.1425 | 17 / 85 (20.0%) | 2 / 29 | 0 | 4.2 |
| uavscenes | sketch peak_ratio (Canny) | hkairport | 1.1260 | 15 / 78 (19.2%) | 9 / 36 | 0 | 1.2 |
| uavscenes | sketch peak_ratio (Canny) | hkisland01 | 1.1360 | 19 / 98 (19.4%) | 2 / 16 | 0 | 2.1 |
| uavscenes | RoMa v2 n_inliers | amtown01 | 1472 | 16 / 81 (19.8%) | 0 / 33 | 0 | -- |
| uavscenes | RoMa v2 n_inliers | amvalley01 | 1136 | 17 / 85 (20.0%) | 3 / 29 | 0 | 2.6 |
| uavscenes | RoMa v2 n_inliers | hkairport | 558 | 15 / 78 (19.2%) | 16 / 36 | 0 | 0.8 |
| uavscenes | RoMa v2 n_inliers | hkisland01 | 1016 | 19 / 98 (19.4%) | 3 / 16 | 0 | 0.4 |
| visloc | sketch peak_ratio (Canny) | visloc02 | 1.0143 | 20 / 100 (20.0%) | 7 / 25 | 2 | 20.7 |
| visloc | sketch peak_ratio (Canny) | visloc03 | 1.0145 | 20 / 100 (20.0%) | 5 / 25 | 1 | 10.9 |
| visloc | sketch peak_ratio (Canny) | visloc09 | 1.0146 | 20 / 100 (20.0%) | 4 / 25 | 3 | 54.0 |
| visloc | sketch peak_ratio (Canny) | visloc10 | 1.0143 | 20 / 100 (20.0%) | 7 / 25 | 4 | 51.5 |
| visloc | sketch peak_ratio (Canny) | visloc11 | 1.0158 | 19 / 100 (19.0%) | 2 / 25 | 2 | 34.3 |
| visloc | RoMa v2 n_inliers | visloc02 | 76 | 20 / 100 (20.0%) | 1 / 25 | 1 | 53.5 |
| visloc | RoMa v2 n_inliers | visloc03 | 57 | 19 / 100 (19.0%) | 14 / 25 | 5 | 24.8 |
| visloc | RoMa v2 n_inliers | visloc09 | 73 | 20 / 100 (20.0%) | 4 / 25 | 1 | 27.6 |
| visloc | RoMa v2 n_inliers | visloc10 | 73 | 20 / 100 (20.0%) | 4 / 25 | 1 | 18.1 |
| visloc | RoMa v2 n_inliers | visloc11 | 68 | 20 / 100 (20.0%) | 7 / 25 | 5 | 33.9 |

## 2. Threshold chosen on all 114 UAVScenes pairs, applied unchanged to all 125 UAV-VisLoc pairs

The threshold was chosen on UAVScenes only; no UAV-VisLoc pair took part in choosing it. Every UAV-VisLoc pair is an evaluation pair here.

| score | target % | threshold | admitted on UAVScenes (chooser) | highest score among the 125 UAV-VisLoc pairs | UAV-VisLoc admitted | share % | wrong >30 m | median m | max m |
|---|---|---|---|---|---|---|---|---|---|
| sketch peak_ratio (Canny) | 10 | 1.1817 | admitted 11 there, 9.6% | 1.0478 | 0 | 0.0 | 0 | -- | -- |
| sketch peak_ratio (Canny) | 20 | 1.1356 | admitted 22 there, 19.3% | 1.0478 | 0 | 0.0 | 0 | -- | -- |
| sketch peak_ratio (Canny) | 30 | 1.0878 | admitted 34 there, 29.8% | 1.0478 | 0 | 0.0 | 0 | -- | -- |
| RoMa v2 n_inliers | 10 | 1864 | admitted 11 there, 9.6% | 266 | 0 | 0.0 | 0 | -- | -- |
| RoMa v2 n_inliers | 20 | 1016 | admitted 22 there, 19.3% | 266 | 0 | 0.0 | 0 | -- | -- |
| RoMa v2 n_inliers | 30 | 529 | admitted 34 there, 29.8% | 266 | 0 | 0.0 | 0 | -- | -- |
| sketch peak_ratio (distilled drawer) | 10 | 1.0231 | admitted 11 there, 9.6% | 1.0115 | 0 | 0.0 | 0 | -- | -- |
| sketch peak_ratio (distilled drawer) | 20 | 1.0138 | admitted 22 there, 19.3% | 1.0115 | 0 | 0.0 | 0 | -- | -- |
| sketch peak_ratio (distilled drawer) | 30 | 1.0106 | admitted 34 there, 29.8% | 1.0115 | 1 | 0.8 | 1 | 62.1 | 62.1 |
| sketch peak_ratio (full drawer) | 10 | 1.0156 | admitted 11 there, 9.6% | 1.0102 | 0 | 0.0 | 0 | -- | -- |
| sketch peak_ratio (full drawer) | 20 | 1.0116 | admitted 22 there, 19.3% | 1.0102 | 0 | 0.0 | 0 | -- | -- |
| sketch peak_ratio (full drawer) | 30 | 1.0087 | admitted 33 there, 28.9% | 1.0102 | 2 | 1.6 | 1 | 22.6 | 42.7 |

Note on the UAV-VisLoc sketch rows of section 1: the same leave-one-scene-out admitted pairs scored with the shipped-heading and cost-selected-heading errors instead of the oracle-heading error. The admission score was computed at the oracle heading, so this pairs a score with an answer from a different heading; given only so the shipped-heading numbers are not hidden.
- sketch peak_ratio (Canny), leave-one-scene-out at 20 %: shipped_err_m: 22 wrong of 25, median 61.0 m; cost_err_m: 22 wrong of 25, median 60.4 m
- sketch peak_ratio (distilled drawer), leave-one-scene-out at 20 %: shipped_err_m: 24 wrong of 25, median 66.0 m; cost_err_m: 21 wrong of 25, median 57.0 m
- sketch peak_ratio (full drawer), leave-one-scene-out at 20 %: shipped_err_m: 22 wrong of 23, median 71.4 m; cost_err_m: 22 wrong of 23, median 62.1 m

## 3. In-sample reference at the 20 % target (threshold chosen and evaluated on the same pairs; NOT held out)

| population | score | threshold | admitted | share % | wrong >30 m | median m | max m |
|---|---|---|---|---|---|---|---|
| uavscenes | sketch peak_ratio (Canny) | 1.1356 | 22 | 19.3 | 1 | 1.9 | 50.2 |
| uavscenes | RoMa v2 n_inliers | 1016 | 22 | 19.3 | 0 | 0.8 | 4.7 |
| uavscenes | sketch peak_ratio (distilled drawer) | 1.0138 | 22 | 19.3 | 1 | 2.1 | 105.0 |
| uavscenes | sketch peak_ratio (full drawer) | 1.0116 | 22 | 19.3 | 1 | 1.3 | 104.9 |
| visloc | sketch peak_ratio (Canny) | 1.0145 | 25 | 20.0 | 12 | 21.0 | 89.5 |
| visloc | RoMa v2 n_inliers | 72 | 25 | 20.0 | 12 | 29.9 | 123.5 |
| visloc | sketch peak_ratio (distilled drawer) | 1.0022 | 25 | 20.0 | 10 | 21.6 | 74.0 |
| visloc | sketch peak_ratio (full drawer) | 1.0037 | 25 | 20.0 | 10 | 27.0 | 65.9 |

## 4. Compact table (primary scores, 20 % target)

| population | score | protocol | admitted / pairs | wrong >30 m | median m |
|---|---|---|---|---|---|
| UAVScenes | sketch peak_ratio (Canny) | leave-one-scene-out | 26 / 114 | 2 | 1.8 |
| UAVScenes | sketch peak_ratio (Canny) | in-sample (reference) | 22 / 114 | 1 | 1.9 |
| UAVScenes | RoMa v2 n_inliers | leave-one-scene-out | 22 / 114 | 0 | 0.8 |
| UAVScenes | RoMa v2 n_inliers | in-sample (reference) | 22 / 114 | 0 | 0.8 |
| UAV-VisLoc | sketch peak_ratio (Canny) | leave-one-scene-out | 25 / 125 | 12 | 21.0 |
| UAV-VisLoc | sketch peak_ratio (Canny) | threshold from UAVScenes | 0 / 125 | 0 | -- |
| UAV-VisLoc | sketch peak_ratio (Canny) | in-sample (reference) | 25 / 125 | 12 | 21.0 |
| UAV-VisLoc | RoMa v2 n_inliers | leave-one-scene-out | 30 / 125 | 13 | 29.0 |
| UAV-VisLoc | RoMa v2 n_inliers | threshold from UAVScenes | 0 / 125 | 0 | -- |
| UAV-VisLoc | RoMa v2 n_inliers | in-sample (reference) | 25 / 125 | 12 | 29.9 |

UAV-VisLoc sketch errors are at the ground-truth-selected heading (ceiling, not a system result). Leave-one-scene-out: each scene's threshold was chosen on the other scenes of the same population and evaluated on that scene only. Threshold from UAVScenes: chosen on the 114 UAVScenes pairs, evaluated on the 125 UAV-VisLoc pairs. In-sample: chosen and evaluated on the same pairs.
