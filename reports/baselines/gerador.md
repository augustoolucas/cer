# Generator baseline exploration — full nested study (task 2)

Bounded local CPU exploration of the **gerador** dataset (574 endpoints, outer folds A/B/C/D). Numbers below come from the run artifacts and a separate numpy recomputation of the saved prediction CSVs. No gearbox run was made.

| item | value |
| --- | --- |
| repository state | `55306f5` + local runner fixes (see §12; uncommitted) |
| runner+`cer_rul` code hash | `4afad1ce15f88a3b857c1f0815a6c4dd1f365b9c86bfe33d6913cd7b14f6e142` |
| `baseline_experiments.py` file sha256 | `5c2831bb3ce2e9aba8451d78a1852d2c408a479700e1aa5053535140ba9a1092` |
| main run | fp `717be61e363ea09d` · `results/baselines/gerador/717be61e363ea09d` |
| time/cadence control run | fp `4500dafde38e4de6` · `results/baselines/gerador/controls/gerador/4500dafde38e4de6` |
| source CSV sha256 | `3cea9d774f19212880cee8aace0563f52003606fe27fa56a6af8d81d481be767` |
| endpoint-set sha256 | `f8fefaa2fb72a9810f9730674b353c2e9740941b6ae22b233cb20dd4cfda4777` |
| versions | {'catboost': 'catboost 1.2.10', 'numpy': '2.5.4', 'optuna': '5.0.0', 'pandas': '3.1.1', 'psutil': '7.2.2', 'python': '3.12.3', 'sklearn': '1.9.1'} |
| invalidated prior run | `results/baselines/gerador/364e1fd26ffbcdcc` — see §12, do not cite |

## 1. Commands and timings (UTC, 2026-11-13)

```
venv/bin/python baseline_experiments.py search   --dataset gerador  > main_search_r2.log      # 02:00:37 → 02:14:57 (854 s wall)
venv/bin/python baseline_experiments.py evaluate --dataset gerador  > main_evaluate_r2.log    # 02:15:59 → 02:16:29 (30 s)
venv/bin/python baseline_experiments.py verify   --dataset gerador  > main_verify_r2.log      # 02:16:42 → 02:16:56 (14 s, 409 PASS)
venv/bin/python baseline_experiments.py report   --dataset gerador
# predeclared time/cadence-only control, separate config and output root
venv/bin/python baseline_experiments.py search   --dataset gerador --config results/baselines/gerador/controls/config_cadence_elapsed_only.json --output-root results/baselines/gerador/controls   # 02:15:03 → 02:15:54 (51 s)
venv/bin/python baseline_experiments.py evaluate …   # 02:16:29 → 02:16:38; report; verify → evaluated_complete
```

| stage | fits started | unique payloads (ok/failed) | cache hits | scheduler wall s | stage elapsed s | peak worker USS MiB | max workers | budget charged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main search | 1455 | 1365/90 | 36 | 274.0 | 854 | 56.2 | 5 | 1455/8000 |
| main evaluate | 59 | 59/0 | 0 | 16.8 | 30 | 30.7 | 5 | 1514/8000 |
| control search | 48 | 48/0 | 0 | 3.8 | 51 | 13.1 | 3 | 48/8000 |
| control evaluate | 16 | 16/0 | 0 | 2.3 | 9 | 11.3 | 4 | 64/8000 |

`runner.wall_s` is the fit-scheduler loop time only; stage elapsed includes feature-table builds, proposal generation, scoring and artifact writes. Budget: 8000 fits, 812 reserved for final refits; main used 1514 in total (1455 search + 59 final), the rest was deliberately not spent. Fits are single-threaded (BLAS/OpenMP=1, `n_jobs=1`), 120 s hard timeout, RAM policy 1.5 GiB reserve / ≤65% of initial availability / 256 MiB floor.

## 2. Protocol

Outer = leave-one-equipment-out. The search receives only outer-training rows; inner = leave-one-group-out over all remaining 3 groups. For each inner group, compute the MAE separately for every `Falha` and average those MAEs equally. The candidate score is the equal mean of these group scores over the three inner groups, in percentage points (pp). Tie: within 0.1 pp lower complexity, then lower inner RMSE, then stable candidate ID. Screening 8 trials/family, top-3 tuned with up to 40 extra proposals and the declared patience rule, ≤12 one-axis ablations, conditional ≤16 parameter-only retune at ≥0.5 pp, fixed 3-seed confirmation for the top-2 stochastic families. The primary baseline is the full inner selector; outer labels are read only by `evaluate`, after all four selections were frozen.

## 3. Primary baseline — inner-selected recipe per outer fold

| outer | chosen | family(ies) | inner score pp | complexity | how |
| --- | --- | --- | --- | --- | --- |
| A | `5f4f42aa23f0` | random_forest+svr | 9.3122 | 74 | mean |
| B | `de6dabdc5ac6` | knn | 15.0318 | 41 | single |
| C | `3e6f45483361` | knn | 11.3097 | 62 | single |
| D | `78a2b88e7626` | catboost | 17.1673 | 20 | single |

**Outer A** (`5f4f42aa23f0`, inner 9.3122 pp):
- `046e88a9ddd1` — random_forest params={"max_depth": null, "max_features": 0.5, "min_samples_leaf": 3, "n_estimators": 155, "n_jobs": 1}; features=current/legacy/w24h/bins32/sensors+elapsed/dropvar; pre=signed_log1p_standard(ind=False,clip=True); weights=component; post clip01=True ema=0.25 running_min=True
- `2a299a6a4214` — svr params={"C": 8.891586501144559, "cache_size": 128, "epsilon": 0.043888560049072045, "gamma": "scale", "max_iter": 20000}; features=summary/legacy/w24h/bins16/sensors+elapsed/rel10; pre=minmax(ind=True,clip=False); weights=group_equal; post clip01=True ema=1.0 running_min=True

**Outer B** (`de6dabdc5ac6`, inner 15.0318 pp):
- `de6dabdc5ac6` — knn params={"n_neighbors": 14, "p": 2, "weights": "uniform"}; features=summary/dedup/w6h/bins16/sensors+elapsed/rel30/dropvar; pre=none(ind=False,clip=False); weights=none; post clip01=True ema=0.5 running_min=False

**Outer C** (`3e6f45483361`, inner 11.3097 pp):
- `3e6f45483361` — knn params={"n_neighbors": 37, "p": 2, "weights": "distance"}; features=robust/legacy/w6h/bins64/sensors+elapsed/rel10/dropvar; pre=minmax(ind=False,clip=True); weights=none; post clip01=False ema=0.25 running_min=True

**Outer D** (`78a2b88e7626`, inner 17.1673 pp):
- `78a2b88e7626` — catboost params={"allow_writing_files": false, "depth": 4, "iterations": 177, "l2_leaf_reg": 8.497783914361122, "learning_rate": 0.15234924110165157, "loss_function": "MAE", "thread_count": 1}; features=current/dedup/w24h/bins32/sensors+elapsed/rel30/dropvar; pre=signed_log1p_standard(ind=False,clip=True); weights=none; post clip01=True ema=1.0 running_min=False

Selections are fold-specific: nothing is transferred between outer folds.

## 4. Outer out-of-fold metrics

Two different aggregations are reported separately. **Equal-group macro** has two steps: within one held-out group compute the metric separately for every `Falha` (MAE, RMSE, signed bias) and average those per-`Falha` values equally; then average the four group values equally. Macro RMSE is therefore 32.090 pp, which is **not** the average of the pooled within-group RMSes (that value is 33.060 pp). **Pooled point** = every out-of-fold record weighted once (574 records; RMSE 35.419 pp). The three numbers weight observations differently and are not interchangeable.

| outer | n | macro MAE pp | macro RMSE pp | macro bias pp |
| --- | --- | --- | --- | --- |
| A | 352 | 30.759 | 35.906 | -30.734 |
| B | 34 | 24.060 | 26.672 | 24.060 |
| C | 138 | 28.694 | 31.787 | 28.689 |
| D | 50 | 21.829 | 33.996 | 20.816 |
| **equal-group average** | 574 | **26.335** | **32.090** | **10.708** |

**Pooled point metrics** (all 574 records): MAE **29.088 pp**, RMSE **35.419 pp**, bias **-8.712 pp**.

The sign flip between macro bias (positive) and pooled bias (negative) is a weighting effect: fold A holds 352 of 574 records and carries a −30.7 pp bias, while the three small folds are positive. Neither number is "the" bias; both are reported.

**Independent recomputation.** A separate numpy implementation (component = `Falha`, component means, then equal weighting) reproduced the stored per-fold component MAEs and the stored macro MAE for all 8 evaluations (main + control, A–D) to <1e-9. Targets were re-read directly from `Features_gerador_score_RUL-porcentagem-data.csv`: 574/574 `y_true` values match (0 mismatches); the four test folds are disjoint and cover exactly 574 unique endpoints (352/34/138/50).

## 5. Slices and prediction ranges

| outer | MAE saudável pp | MAE pós-FPT pp | MAE y≤0.2 pp (n) | pred_raw finite | pred_raw outside [0,1] | pred_post outside [0,1] |
| --- | --- | --- | --- | --- | --- | --- |
| A | 15.374 | 35.283 | 9.047 (56) | 0/352 | N/A | 0.0% |
| B | N/A | 24.060 | 38.796 (10) | 34/34 | 0.0% | 0.0% |
| C | 0.035 | 30.933 | 34.182 (54) | 138/138 | 0.0% | 0.0% |
| D | 0.710 | 35.908 | 84.360 (2) | 50/50 | 14.0% | 0.0% |

Fold A’s chosen candidate is an ensemble; its `pred_raw` column is stored empty (no single raw model output exists for a mean of two pipelines), so the raw-range statistic is N/A there rather than 100%. Range checks use only finite stored values. `pred_post` is finite and inside [0,1] for 574/574 records in the main study (0.0% outside in every fold). Fold D raw predictions are 14.0% outside [0,1] before the selected clip, which is what the post-processing step then fixes.

Worst fold: A (30.76 pp macro MAE); best: D (21.83 pp). Slice numbers are descriptive (phase is never an input; `y≤0.2` covers only 2–56 records per fold).

## 6. Stability

Selections: A random_forest+svr ensemble (mean of post-processed predictions), B `knn`, C `knn`, D `catboost`. Only `knn` repeats; no recipe is carried across folds, and feature representation/grid/post-processing differ per fold. The honest reading is fold-wise selection, not one globally stable configuration.

Fixed 3-seed confirmation (offsets 0,1,2, same hyperparameters and group masks; the averaged-OOF candidate is what can be selected, never the luckiest single seed):

| outer | family | seed scores pp | mean pp | std pp | averaged OOF pp |
| --- | --- | --- | --- | --- | --- |
| A | random_forest | 9.888, 10.453, 10.372 | 10.238 | 0.249 | 10.220 |
| A | catboost | 15.441, 15.697, 14.254 | 15.130 | 0.629 | 14.927 |
| B | extra_trees | 18.810, 18.755, 18.782 | 18.782 | 0.022 | 18.774 |
| B | random_forest | 24.689, 24.645, 24.802 | 24.712 | 0.066 | 24.715 |
| C | extra_trees | 17.334, 16.974, 17.252 | 17.187 | 0.154 | 17.256 |
| C | catboost | 20.199, 20.510, 20.321 | 20.343 | 0.128 | 20.466 |
| D | extra_trees | 20.052, 20.427, 20.624 | 20.368 | 0.238 | 20.364 |
| D | catboost | 16.271, 18.389, 17.195 | 17.285 | 0.867 | 17.167 |

## 7. Search effort and pruning

| outer | trials | ok | failed | screen | tune | ablation | ablation_retune | seed_conf | legacy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 118 | 108 | 10 | 56 | 49 | 11 | 0 | 2 | 0 |
| B | 119 | 109 | 10 | 59 | 48 | 10 | 0 | 2 | 0 |
| C | 110 | 100 | 10 | 58 | 40 | 10 | 0 | 2 | 0 |
| D | 114 | 107 | 7 | 57 | 28 | 11 | 16 | 2 | 0 |

Fit accounting for the whole main study: **1514 fits started in total** (1455 search + 59 final refits) plus **36 cache hits**, which are reused payloads and not actual fits. Unique payloads: 1365 ok / 90 failed in the search stage and 59 / 0 in the final-refit stage, 0 timeouts, 0 worker failures. The 90 failures are convergence/solver failures recorded with reasons and never scored: 6.19% of search fits (90/1455) and 5.94% of all fits (90/1514), below the 25% expansion gate, which is why tuning continued.

- **A** — evaluated 11 one-axis variants; configured cap 12; retune: not run: ablation gain 0.2900 pp < 0.5000 pp; seeds: confirmed ['random_forest', 'catboost'] with fixed offsets [0,1,2]; legacy presets: evaluated 7 declared legacy quick presets under the current inner LOGO (descriptive control, outside selection)
- **B** — evaluated 10 one-axis variants; configured cap 12; retune: not run: ablation gain 0.0000 pp < 0.5000 pp; seeds: confirmed ['extra_trees', 'random_forest'] with fixed offsets [0,1,2]; legacy presets: evaluated 7 declared legacy quick presets under the current inner LOGO (descriptive control, outside selection)
- **C** — evaluated 10 one-axis variants; configured cap 12; retune: not run: ablation gain 0.3259 pp < 0.5000 pp; seeds: confirmed ['extra_trees', 'catboost'] with fixed offsets [0,1,2]; legacy presets: evaluated 7 declared legacy quick presets under the current inner LOGO (descriptive control, outside selection)
- **D** — evaluated 11 one-axis variants; configured cap 12; retune: retuned catboost parameters only for up to 16 proposals; seeds: confirmed ['catboost', 'extra_trees'] with fixed offsets [0,1,2]; legacy presets: evaluated 7 declared legacy quick presets under the current inner LOGO (descriptive control, outside selection)

The `legacy` column is 0 by design: legacy preset results are recorded in `selection.json → legacy_controls` rather than in the trial log, because they are descriptive and must not be mistaken for search trials.

Pruned before fitting: invalid combinations (e.g. relative/variance requested without the sensors group) are rejected and counted as invalid trials. Tuning stopped by the declared plateau rule; parameter retune ran only for D (`catboost`), the only fold whose ablation gain cleared 0.5 pp. The seven legacy quick presets are evaluated **descriptively only** — their scores are recorded in `selection.json` but never enter family state, the anchor, the tie-break or the final choice (regression test `LegacyControlIsolationTests`).

## 8. Descriptive per-family metrics (not the primary baseline)

Inner best score per family (pp):

| family | A | B | C | D |
| --- | --- | --- | --- | --- |
| catboost | 14.93 | 27.22 | 20.47 | 17.17 |
| elasticnet | 18.35 | 20.30 | 25.94 | 19.32 |
| extra_trees | 25.03 | 18.77 | 17.26 | 20.36 |
| hist_gb | 12.25 | 20.63 | 19.59 | 20.11 |
| huber | N/A | 32.62 | 27.46 | 43.59 |
| knn | 14.25 | 15.03 | 11.31 | 28.87 |
| mean | 33.91 | 34.73 | 29.76 | 32.24 |
| median | 36.06 | 36.91 | 30.36 | 34.25 |
| random_forest | 10.22 | 24.72 | 20.29 | 21.19 |
| ridge | 13.94 | 17.32 | 21.09 | 23.81 |
| svr | 11.45 | 23.69 | 13.98 | 22.53 |

Outer equal-group MAE per family (descriptive; each family’s inner best recipe refit on the outer-training rows). The fold baseline is the inner selector, not the family that looks best on outer data.

| family | A | B | C | D |
| --- | --- | --- | --- | --- |
| catboost | 31.38 | 24.71 | 23.74 | 21.83 |
| elasticnet | 43.51 | 18.29 | 31.83 | 24.34 |
| extra_trees | 28.32 | 24.10 | 35.62 | 21.50 |
| hist_gb | 24.49 | 26.53 | 41.83 | 22.85 |
| huber | N/A | 24.25 | 11.60 | 27.00 |
| knn | 15.66 | 24.06 | 28.69 | 14.07 |
| mean | 33.31 | 25.32 | 36.80 | 33.94 |
| median | 36.86 | 26.21 | 38.92 | 33.78 |
| random_forest | 40.66 | 21.46 | 20.31 | 21.68 |
| ridge | 20.27 | 27.77 | 41.82 | 24.56 |
| svr | 20.89 | 22.45 | 21.37 | 11.84 |

## 9. Predeclared time/cadence-only control (untuned)

Config `results/baselines/gerador/controls/config_cadence_elapsed_only.json`, declared before any outer metric was read: every recipe choice pinned to its declared default except `feature_groups=cadence+elapsed` (no sensor columns); families mean/median/Ridge(α=1)/HistGB(100 iters, leaf 10, 15 leaves, lr 0.1, L2 1, absolute_error, no early stopping); one screening trial, no tuning, no ablations, no ensembles, no legacy presets; separate output root. It measures the acquisition/elapsed shortcut under the same nested protocol.

| outer | chosen (inner) | inner pp | macro MAE pp | macro RMSE pp | macro bias pp |
| --- | --- | --- | --- | --- | --- |
| A | ridge | 15.390 | 22.429 | 25.631 | -22.429 |
| B | ridge | 22.149 | 18.988 | 22.966 | 18.126 |
| C | hist_gb | 22.515 | 14.565 | 17.382 | 12.105 |
| D | hist_gb | 23.827 | 15.527 | 21.185 | 5.942 |
| **equal-group average** | — | — | **17.877** | **21.791** | **3.436** |

Control pooled point metrics: MAE 19.733 pp, RMSE 23.363 pp, bias -9.253 pp (574 records).

Control per-family inner scores (pp):

| family | A | B | C | D |
| --- | --- | --- | --- | --- |
| mean | 33.91 | 34.73 | 29.76 | 32.24 |
| median | 36.06 | 36.91 | 30.36 | 34.25 |
| ridge | 15.39 | 22.15 | 35.23 | 27.97 |
| hist_gb | 16.40 | 26.90 | 22.51 | 23.83 |

**Finding: this fixed, untuned time/cadence-only pipeline scored better out-of-fold (equal-group macro MAE 17.877 pp) than the searched full-feature pipeline (26.335 pp) in this run.** That is a statement about these two pipelines on these four folds — the control uses different recipes, a different feature space and no tuning, so it does not show that the sensor features carry no information, nor that time causes the labels. The honest summary is: a simple acquisition-time baseline is a strong reference here and should be reported alongside any learned result.

Declared legacy quick presets evaluated under this run’s inner LOGO (descriptive only; equal-group macro MAE is not computed for them because they are not refit on outer data):

| preset | A | B | C | D |
| --- | --- | --- | --- | --- |
| mean_summary_none | 33.91 | 34.73 | 29.76 | 32.24 |
| ridge1_summary_none | 38.77 | 44.20 | 57.37 | 62.48 |
| ridge10_current_component | 44.02 | 35.35 | 55.30 | 43.03 |
| svr10_summary_none | 27.65 | 32.63 | 26.51 | 30.14 |
| svr_current_component_phase | 41.64 | 46.67 | 37.91 | 31.80 |
| et128_summary_none | 27.85 | 31.99 | 21.91 | 35.24 |
| et_current_component | 28.37 | 38.31 | 29.78 | 39.78 |

The control artifacts print `legacy_control: not run: no legacy presets declared in this config`, which is correct for that config (it declares none).

## 10. Comparison with the previous quick protocol (ZIP `111f7eafc72b5673.zip`)

The old archive used a different selection protocol (single 12 h quick presets, one representation per preset, its own fold scoring), so only the **same family, same aggregation** is compared. A useful comparability anchor: the old `mean` family’s equipment-macro MAE was 32.342 pp, and the fixed `mean` family in this study scores a near-identical equal-group average on the new folds (32.342 pp), which is what one expects from an untuned constant.

Descriptive equal-equipment MAE comparison (pp): previous quick ExtraTrees 29.667; new inner-selected pipeline 26.335; fixed time/cadence control 17.877. These are averages over the same four held-out groups, not per-fold-versus-global comparisons. Different selection procedures and this exploratory design prevent attributing the differences to any single change.

## 11. Reproducibility and verification scope

```
main search log   results/baselines/gerador/main_search_r2.log
main artifacts    results/baselines/gerador/717be61e363ea09d/{manifest.json,ledger.json,stage_costs.json,outer/[A-D]/{selection,trials,evaluation,predictions,candidates},features,fits}
control config    results/baselines/gerador/controls/config_cadence_elapsed_only.json
control artifacts results/baselines/gerador/controls/gerador/4500dafde38e4de6/…
report            reports/baselines/gerador.md
```

`verify` (409 PASS checks, `RESULT: evaluated_complete`) re-checks fingerprints, CSV hash, held-out-equipment exclusion from each search’s training groups, candidate payload integrity, prediction coverage/targets and metric reproducibility. It is a mechanical consistency check of the recorded flow; it cannot prove that no outer label influenced the inner search. That property rests on the code path (search is constructed from outer-training rows only and `evaluate` is the only stage that reads outer targets) plus the synthetic permutation regression test that flips held-out labels and confirms identical selections.

## 12. Run invalidation and the scope deviation

The first completed study (`364e1fd26ffbcdcc`, logs `main_search.log`/`main_evaluate.log`, runner hash `ce99f5ca…`) was **invalidated** by this review: `_legacy_controls` fed its descriptive scores into per-family `best`/anchor state, so stop decisions could differ from the approved protocol even though no legacy preset won a fold. Its directory is read-only and contains `INVALIDATED.md` plus `source_snapshot/baseline_experiments.py` (raw sha256 `763ee241e5ae516c178845d2d791f7ac4456959a34d9a477e8b4df213d42df6f`). The present study re-ran the same predeclared config and seeds on the corrected source; the selections happened to be identical, but only this run is reported.

Two earlier runner fixes were also needed before any study (disclosed, not hidden): the first `search` attempt died with `KeyError: max_iter` in `build_estimator` for the declared legacy SVR presets. Fixes: a `REQUIRED_PARAMS` completeness guard that raises `InvalidRecipe` (a config defect is now a recorded invalid candidate, not a fatal worker error) and filling definitional `const` parameters from the declared family space. Focused tests: `DeclaredPresetCompletenessTests`, `LegacyControlIsolationTests` (state before/after legacy controls identical; selection identical with and without them). No notebook, CSV, protocol or model-space change.

## 13. Limitations

- No MAPE, no significance testing, no "optimal" or universal-recipe claims.
- One fold-wise inner selection per outer group; with 4 folds, differences of a few pp are not resolved by this design.
- Outer metrics are out-of-fold but selected nothing; the control was fixed before outer metrics were read.
- The 90 recorded fit failures are not imputed; families with such failures may be slightly under-represented in the descriptive tables.
- Ensemble `pred_raw` is a stored placeholder, not a model output; all reported metrics use `pred_post`.
- Gearbox was not trained; nothing here transfers to the gearbox datasets.

## 14. Curvas (artefatos locais, descritivos)

Plots gerados **apenas** das predições OOF congeladas do run principal e do controle de tempo/cadência já executado (nenhum ajuste novo). Eixo x = data do registro, y = RUL em %; preto = alvo, azul = escolha primária da seleção interna, laranja = controle de tempo/cadência casado, cinza tracejado = referência fixa `mean`. Um painel por `Falha`.

- [gerador_outerA_by_fault.png](../../results/baselines/gerador/717be61e363ea09d/plots/gerador_outerA_by_fault.png)
- [gerador_outerB_by_fault.png](../../results/baselines/gerador/717be61e363ea09d/plots/gerador_outerB_by_fault.png)
- [gerador_outerC_by_fault.png](../../results/baselines/gerador/717be61e363ea09d/plots/gerador_outerC_by_fault.png)
- [gerador_outerD_by_fault.png](../../results/baselines/gerador/717be61e363ea09d/plots/gerador_outerD_by_fault.png)
