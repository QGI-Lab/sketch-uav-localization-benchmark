# Scrub report

Every file here was copied out of a working tree and passed through one text substitution pass before being written. This report says what was replaced and where. The removed strings are described rather than quoted, so that reading this report does not undo the anonymisation.

## The substitution rules

| id | what was removed | what it became |
|---|---|---|
| S1 | absolute path to the compute cluster scratch tree, which contains the account name of one of the authors | `<DATA_ROOT>/...` |
| S2 | cluster login host and account name | `<user>@<cluster>` |
| S3 | bare cluster account name of one of the authors | `<user>` |
| S4 | internal project name used as a directory segment in a path-repair helper | `legacy` (the helper is a no-op on any tree but the original one) |
| S5 | pointer to an internal policy document by name | a plain-language statement of what the policy says |
| S6 | environment variable prefix carrying the internal project name | `BENCH_` |
| S7 | Python variable name carrying the internal project name | `PROJ` |
| S8 | per-pair local cache directory fields, which pointed at files not in this release | removed (see `dropped_fields` in that file) |
| S9 | free-text note naming an internal design document, a build tag and an internal script | a neutral description of what the pool is |

## Where each rule fired

| file | rule | times |
|---|---|---|
| `results/2026-09-15_rerun_matrix/run_matrix_sketch.py` | S1 | 5 |
| `results/2026-09-15_rerun_matrix/run_matrix_sketch.py` | S4 | 3 |
| `results/2026-09-15_rerun_matrix/run_matrix_sketch.py` | S5 | 2 |
| `results/2026-09-15_rerun_matrix/run_matrix_sketch.py` | S6 | 24 |
| `results/2026-09-15_rerun_matrix/run_matrix_roma.py` | S1 | 2 |
| `results/2026-09-15_rerun_matrix/run_matrix_roma.py` | S4 | 3 |
| `results/2026-09-15_rerun_matrix/run_matrix_roma.py` | S6 | 11 |
| `results/2026-09-15_boundary_geometry/boundary_geometry.py` | S5 | 1 |
| `results/2026-09-15_boundary_geometry/boundary_geometry.py` | S7 | 4 |
| `results/2026-09-15_directional_geometry/directional_geometry.py` | S4 | 1 |
| `results/2026-09-15_directional_geometry/directional_geometry.py` | S6 | 3 |
| `results/2026-09-15_directional_geometry/directional_geometry.py` | S7 | 3 |
| `results/2026-09-16_r9_inwindow_recount/r9_recount.py` | S5 | 1 |
| `results/2026-09-15_rerun_matrix/matrix_canny_uavscenes_shipped.jsonl` | S1 | 69 |
| `results/2026-09-15_rerun_matrix/matrix_distill_uavscenes_shipped.jsonl` | S1 | 69 |
| `results/2026-09-15_rerun_matrix/matrix_learned_uavscenes_shipped.jsonl` | S1 | 69 |
| `data/uavscenes114_pool_reduced.json` | S8 | 342 |
| `data/uavscenes114_pool_reduced.json` | S9 | 1 |

## Two more edits, both to stop a script writing an absolute path into its own output

Two of the analysis scripts recorded the directory they read from, or the file they
wrote, as a full path. Run anywhere, that path names the machine and the user. Both
were changed to record a name relative to the script instead.

| file | what changed |
|---|---|
| `results/2026-09-16_r10_matched_coverage/r10_matched_coverage.py` | the `source_dir` field of its JSON output now holds the source directory name, not its full path |
| `results/2026-09-15_paper_figures/make_figures_2026_09_15.py` | the `path` field recorded for each figure now holds the file name, not its full path |

Neither changes a number. The reference outputs in this bundle were regenerated after
the change and match the originals exactly apart from those two fields.

## Rules S4, S6 and S7 change behaviour, and that is deliberate

S6 renames the environment variables the two matcher drivers read, for example the
variable that chooses which drawing arm to run. The names in this copy all begin
`BENCH_`. The usage lines in each script's docstring were renamed in the same pass,
so the scripts and their own instructions agree. S7 renames one path variable inside
two analysis scripts. S4 renames a directory segment inside a path-repair helper that
only ever fires on the original cluster tree.

None of the three changes any number this bundle reports.

## What was left out on purpose

These files sit next to the ones included and were not copied.

| left out | why |
|---|---|
| every `*.slurm` job script | carries the cluster account, partition and queue names |
| every `slurm_*.out` job log | carries account name, host names and absolute paths on every line |
| a local smoke-test result file | marked do-not-upload at source; it is a 45-pair partial pass, not a result |
| the `*_LOCAL.json` twins of the pool and the split | identical content with local absolute paths added |
| `matrix_roma_visloc.jsonl`, `matrix_canny_visloc_oracle.jsonl` | an earlier 75-pair scene list that the 125-pair list replaced; keeping them invites a reader to recompute a retired number |
| model weight files | binary, and not needed to recompute any number in this bundle |
| all imagery, `.npz` and `.npy` caches | both source datasets are public and are referenced rather than redistributed |

## What was checked and found clean

A scan of the finished bundle for author names, e-mail addresses, institution
names, private flight names, repository hosts, Windows user paths, home paths
and cluster paths returns nothing outside this report.
