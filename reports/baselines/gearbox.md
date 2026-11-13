# Gearbox baseline exploration — full nested study (task 3)

Bounded local CPU exploration of the **gearbox** dataset: 822 per-series endpoints over 3 held-out events (E1 = measurements 1–6, E2 = 7–12, E3 = 13–18), 6 measurements per event, plus a separate strict joint-event track (137 synchronized event timestamps). All numbers come from the run artifacts and an independent numpy recomputation. No generator winner was transferred.

| item | value |
| --- | --- |
| repository state | `6e315ee` (approved task-2 commit) + no source changes |
| runner+`cer_rul` code hash | `4afad1ce15f88a3b857c1f0815a6c4dd1f365b9c86bfe33d6913cd7b14f6e142` |
| main run | fp `101e2f1fc3d406b1` · `results/baselines/gearbox/101e2f1fc3d406b1` |
| time/cadence control run | fp `6932403970ae9c17` · `results/baselines/gearbox/controls/gearbox/6932403970ae9c17` |
| source CSV sha256 | `332d452bbe2a1c2d4cb4a3b046e5643e386545f0db3cad323dbfd7defdbdd0e4` |
| endpoint-set sha256 | `dc2e303323c9d6174434fe8d08de5b9b05766f801319665acce0f482cdfafa45` |
| versions | {'catboost': 'catboost 1.2.10', 'numpy': '2.5.4', 'optuna': '5.0.0', 'pandas': '3.1.1', 'psutil': '7.2.2', 'python': '3.12.3', 'sklearn': '1.9.1'} |

## 1. Commands and timings (UTC, 2026-11-13)

```
venv/bin/python baseline_experiments.py search   --dataset gearbox  > main_search.log     # 02:36:32 → 02:54:56 (1104 s elapsed)
venv/bin/python baseline_experiments.py evaluate --dataset gearbox  > main_evaluate.log   # 02:55:01 → 02:56:05 (64 s)
venv/bin/python baseline_experiments.py verify   --dataset gearbox  > main_verify.log     # 02:56:41 → 02:57:02 (21 s, 478 PASS)
venv/bin/python baseline_experiments.py report   --dataset gearbox
# predeclared time/cadence-only control (separate config and output root)
venv/bin/python baseline_experiments.py search   --dataset gearbox --config results/baselines/gearbox/controls/config_cadence_elapsed_only.json --output-root results/baselines/gearbox/controls   # 02:34:48 → 02:36:26 (98 s)
venv/bin/python baseline_experiments.py evaluate …   # 02:56:05 → 02:56:36 (31 s); report; verify → evaluated_complete
```

| stage | fits started | cache hits | unique payloads ok/failed | scheduler wall s | stage elapsed s | peak worker USS MiB | max workers | budget charged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main search | 1248 | 28 | 1199/49 | 357.4 | 1104 | 57.5 | 5 | 1248/8000 |
| main evaluate | 78 | 0 | 78/0 | 29.4 | 64 | 30.2 | 5 | 1326/8000 |
| control search | 84 | 6 | 84/0 | 13.3 | 98 | 15.8 | 5 | 84/8000 |
| control evaluate | 30 | 0 | 30/0 | 6.6 | 31 | 19.1 | 5 | 114/8000 |

`runner.wall_s` is the fit-scheduler loop only; elapsed includes nine feature-grid builds (~49 s each on first use), proposal generation, scoring and artifact writes. Main study used 1326 fits of 8000 (609 reserved for final refits); the rest was not spent. Unique payloads (source of truth): **1326 = 1277 ok + 49 failed**, i.e. 3.7% of all fits failed (49/1326) and 3.9% of search fits (49/1248); all failures are recorded convergence/solver failures with reasons and were never scored. Cache hits (28) are reused payloads, not extra fits. Fits are single-threaded (BLAS/OpenMP=1, `n_jobs=1`), 120 s hard timeout, RAM policy 1.5 GiB reserve / ≤65% availability / 256 MiB floor.

## 2. Protocol

Outer = leave-one-event-out (E1, E2, E3). Inner = 2 folds over the remaining 2 events (train one, validate the other). For each inner group the MAE is computed separately for every `Falha` (one per measurement series) and averaged equally; the candidate score is the equal mean of the two group scores, in pp. Tie: within 0.1 pp lower complexity, then lower inner RMSE, then stable candidate ID. Screening 8 trials per family for 9 learned families plus mean/median references, top-3 tuned with the declared patience rule, ≤12 one-axis ablations, ≤16 parameter-only retune at ≥0.5 pp, 3-seed confirmation for the top-2 stochastic families, uniform ensembles. Bounded strategy probes (Ridge/ET/HGB only) cover pooled-anonymous, measurement-onehot, 4 component experts and 6 measurement experts; expansion beyond the probe budget requires ≥0.5 pp inner gain. The joint event track is searched separately on 137 strictly aligned event rows.

## 3. Primary baseline — per-series inner selection

| outer | chosen | family | strategy | inner pp | complexity | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | `0342cf2bd308` | catboost | pooled_anonymous | 6.3198 | 93 | [0, 1, 2] |
| E2 | `88f62aed83d3` | random_forest | pooled_anonymous | 14.4370 | 93 | [0, 1, 2] |
| E3 | `563b933b4c6a` | hist_gb | pooled + measurement one-hot | 7.6905 | 45 | — |

**E1** (`0342cf2bd308`, inner 6.3198 pp) — catboost strategy=pooled_anonymous; features=robust/legacy/w24h/bins64/sensors+cadence/rel30; pre=signed_log1p_standard(ind=False,clip=True); weights=component; post clip01=True ema=0.5 running_min=True; params={"allow_writing_files": false, "depth": 3, "iterations": 318, "l2_leaf_reg": 10.709674892985614, "learning_rate": 0.07946859979528123, "loss_function": "MAE", "thread_count": 1}

**E2** (`88f62aed83d3`, inner 14.4370 pp) — random_forest strategy=pooled_anonymous; features=robust/dedup/w12h/bins32/sensors+cadence/rel30; pre=standard(ind=False,clip=True); weights=none; post clip01=True ema=0.25 running_min=True; params={"max_depth": 4, "max_features": 0.5, "min_samples_leaf": 16, "n_estimators": 229, "n_jobs": 1}

**E3** (`563b933b4c6a`, inner 7.6905 pp) — hist_gb strategy=pooled_anonymous; features=current/legacy/w24h/bins64/sensors/rel30/onehot; pre=robust(ind=False,clip=True); weights=group_equal; post clip01=True ema=0.5 running_min=True; params={"early_stopping": false, "l2_regularization": 0.600185725543042, "learning_rate": 0.042487384450975406, "loss": "absolute_error", "max_iter": 164, "max_leaf_nodes": 27, "min_samples_leaf": 18}

Selections are per outer event; no winner is transferred between events.

Stored provenance: E3’s recipe serialises `strategy="pooled_anonymous"` with `features.measurement_onehot=true` (trial −4, ablation phase). The one-hot flag is an opt-in metadata option of the **main pooled search** — the “anonymous” label is nominal, and the bounded pooled-anonymous probe excludes the metadata column. E3 was therefore not selected by the bounded one-hot probe. E1’s `ridge` family best, by contrast, comes from the bounded probe (`strategy_screen`, trial −11001).

## 4. Series-track out-of-fold metrics (822 endpoints)

Equal-event macro = within each held-out event, per-`Falha` MAE/RMSE/signed bias averaged equally over the 6 measurements, then equal average over the 3 events. Pooled point = all 822 records weighted once. Reported separately. Note that the stored per-fold `rmse_pp` in `evaluation.json` is the pointwise RMSE over that event’s records, so it differs slightly from the macro RMSE shown here (the MAE and bias columns do match the stored values).

| outer | n | components | macro MAE pp | macro RMSE pp | macro bias pp |
| --- | --- | --- | --- | --- | --- |
| E1 | 426 | 6 | 11.360 | 15.241 | -4.007 |
| E2 | 198 | 6 | 13.190 | 17.820 | 12.008 |
| E3 | 198 | 6 | 11.475 | 16.087 | -3.022 |
| **equal-event average** | 822 | — | **12.008** | **16.383** | **1.660** |

Pooled point metrics: MAE **11.828 pp**, RMSE **16.187 pp**, bias **0.088 pp**.

## 5. Slices, ranges and calibration

| outer | MAE saudável pp | MAE pós-FPT pp | MAE y≤0.2 pp (n) | pred_raw finite | pred_raw outside [0,1] | pred_post outside [0,1] |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | 4.98 | 16.90 | 29.04 (24) | 426/426 | 1.4% | 0.0% |
| E2 | 1.48 | 20.80 | 33.26 (30) | 198/198 | 0.0% | 0.0% |
| E3 | 1.53 | 16.45 | 18.72 (36) | 198/198 | 0.0% | 0.0% |

Range statistics use finite stored values only; a NaN/empty `pred_raw` (ensemble candidate) is N/A, never counted as out of range. `pred_post` is inside [0,1] for 822/822 records.

## 6. Stability and seed spread

Fixed 3-seed confirmation (offsets 0,1,2, identical hyperparameters and masks; the averaged OOF candidate is selectable, the luckiest seed never is):

| outer | family | seed scores pp | mean pp | std pp | averaged OOF pp |
| --- | --- | --- | --- | --- | --- |
| E1 | extra_trees | 12.451, 12.558, 12.463 | 12.491 | 0.048 | 12.492 |
| E1 | catboost | 6.014, 7.276, 6.275 | 6.522 | 0.544 | 6.320 |
| E2 | extra_trees | 17.465, 17.498, 17.253 | 17.406 | 0.109 | 17.390 |
| E2 | random_forest | 14.518, 14.421, 14.388 | 14.442 | 0.055 | 14.437 |
| E3 | extra_trees | 14.258, 14.917, 14.468 | 14.548 | 0.275 | 14.520 |
| E3 | catboost | 13.580, 13.546, 13.691 | 13.606 | 0.062 | 13.514 |

Selected: catboost (E1), random_forest (E2), hist_gb (E3) — no family repeats, so this is fold-wise selection rather than one stable gearbox configuration.

## 7. Strategy probes (bounded) and expansion gate

| outer | tuned pooled reference pp | pooled probe | onehot probe | component-expert probe | measurement-expert probe | eligible ≥0.5 pp | expanded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 6.014 | 18.303 | 18.303 | 17.410 | 16.686 | none | none |
| E2 | 14.518 | 21.046 | 20.918 | 19.645 | 19.633 | none | none |
| E3 | 7.691 | 22.985 | 22.187 | 14.671 | 14.258 | none | none |

The gate reference is that event’s fully tuned **main search** score (which itself may sample the optional measurement metadata), not the strict probe score, so no probe came within 0.5 pp and no strategy was expanded further (probe budget ≤2 recipes per family per strategy). Probe results were still eligible as individual candidates: E1’s ridge family best uses `measurement_experts` and comes from the probe trace (`strategy_screen`). E3’s selected pipeline also carries `measurement_onehot=true`, but that flag was sampled by the main pooled search, so it is metadata provenance, not evidence that the bounded one-hot probe won. All choices were made on inner scores; no outer result was used to choose a strategy. Every expert fit excludes the entire held-out event (all six measurements), and a missing expert is a recorded failure, never a silent global fallback.

## 8. Search effort and pruning

| outer | trials | ok | failed | screen | tune | ablation | retune | seed conf | strategy probes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 148 | 140 | 8 | 52 | 43 | 11 | 16 | 2 | 24 |
| E2 | 127 | 117 | 10 | 59 | 30 | 12 | 0 | 2 | 24 |
| E3 | 122 | 114 | 8 | 49 | 37 | 10 | 0 | 2 | 24 |

- **E1** — evaluated 11 one-axis variants; configured cap 12; retune: retuned catboost parameters only for up to 16 proposals; seeds: confirmed ['catboost', 'extra_trees'] with fixed offsets [0,1,2]; strategy: pooled reference 6.0145 pp; eligible []; expanded []; per-strategy rec
- **E2** — evaluated 12 one-axis variants; configured cap 12; retune: not run: ablation gain 0.1041 pp < 0.5000 pp; seeds: confirmed ['random_forest', 'extra_trees'] with fixed offsets [0,1,2]; strategy: pooled reference 14.5183 pp; eligible []; expanded []; per-strategy re
- **E3** — evaluated 10 one-axis variants; configured cap 12; retune: not run: ablation gain 0.0449 pp < 0.5000 pp; seeds: confirmed ['catboost', 'extra_trees'] with fixed offsets [0,1,2]; strategy: pooled reference 7.6905 pp; eligible []; expanded []; per-strategy rec

Recorded failure reasons (never scored): ElasticNet/Huber objective did not converge; SVR solver terminated early (max_iter=20000); lbfgs failed to converge (max_iter=500).

Three invalid-recipe rejections (relative/variance without the sensors group) were counted as invalid, not fits. The legacy quick presets were evaluated descriptively only (7 presets, recorded under `legacy_controls`, never in selection state).

## 8b. Constant references and per-role breakdown (series track, descriptive)

Both constant references are evaluated under the same protocol and are **OOF equal-event**
numbers, not inner scores: `mean` **30.376297 pp** and `median` **30.150404 pp** (MAE) versus
the selected baseline **12.008249 pp**. The gap is large and consistent across the three events
(mean: 29.144 / 31.276 / 30.709 pp; median: 26.654 / 31.956 / 31.841 pp).

Per-measurement-role MAE of the chosen primary predictions (pp):

| role | E1 | E2 | E3 | equal-event average |
| --- | --- | --- | --- | --- |
| HSS radial (pos. 1) | 10.937 | 11.648 | 11.103 | 11.22892 |
| HSS axial (pos. 2) | 13.065 | 12.368 | 11.942 | 12.45826 |
| IMS (pos. 3) | 10.107 | 13.779 | 12.274 | 12.05344 |
| LSS (pos. 4) | 9.541 | 12.117 | 11.894 | 11.18403 |
| Planetaria 0° (pos. 5) | 12.228 | 14.914 | 10.215 | 12.45212 |
| Planetaria 90° (pos. 6) | 12.280 | 14.316 | 11.422 | 12.67271 |

A secondary **physical 4-component** view (HSS = mean of positions 1–2, IMS = 3, LSS = 4,
Planetaria = mean of 5–6) uses **different weights from the primary metric** (components are
equally weighted within an event rather than the six roles) and is therefore reported only as a
complement: HSS 11.844, IMS 12.053, LSS 11.184, Planetaria 12.562 pp (equal-event). No winner is
chosen from this view, and it is never compared with outer-fold selection.

## 9. Joint event track (137 aligned event timestamps)

Six measurement blocks per event timestamp, strict Date/target/phase alignment, no filling; 71 rows for E1 and 33 each for E2/E3 (137 total). This track is **not** pooled with the 822 series records: one event row is one synchronized timestamp, and `Falha` carries the event id (one component per event).

| outer | n | macro MAE pp | macro RMSE pp | macro bias pp | chosen | inner pp |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | 71 | 23.106 | 34.850 | 20.715 | ridge | 16.815 |
| E2 | 33 | 12.777 | 18.603 | 10.779 | ensemble | 18.656 |
| E3 | 33 | 16.969 | 22.870 | -14.362 | ensemble | 11.117 |
| **equal-event average** | 137 | **17.617** | **25.441** | **5.711** | — | — |

Joint pooled point (137 rows): MAE 19.139 pp, RMSE 28.962 pp, bias 9.872 pp.

## 10. Aggregation track (mean/median of the six series predictions per event)

The combination rule (mean vs median) was selected on inner OOF only; the outer application uses the same rule on the same six primary per-series predictions.

| outer | chosen how | inner pp (mean / median) | n | macro MAE pp | macro RMSE pp | macro bias pp |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | median | 6.011 / 5.825 | 71 | 9.545 | 12.661 | -3.568 |
| E2 | mean | 13.680 / 14.414 | 33 | 13.175 | 17.643 | 12.008 |
| E3 | median | 7.607 / 7.452 | 33 | 10.994 | 15.404 | -3.564 |
| **equal-event average** | — | — | 137 | **11.238** | **15.236** | **1.625** |

Aggregation pooled point (137 rows): MAE 10.768 pp, RMSE 14.670 pp, bias 0.185 pp. This is the same 137 timestamps as §9 with a different input, so the two tables must not be merged or ranked against the 822-row series table.

## 11. Predeclared time/cadence-only control (untuned, 5 families)

Config `results/baselines/gearbox/controls/config_cadence_elapsed_only.json`, declared before any outer result was read: every recipe choice pinned to its default except `feature_groups=cadence+elapsed`; families mean/median/Ridge(α=1)/ExtraTrees(128, unlimited depth, leaf 1, `max_features=1.0`)/HistGB(100 iters, leaf 10, 15 leaves, lr 0.1, L2 1, absolute_error, no early stopping); one screening trial, no tuning/ablations/ensembles/legacy presets; `gearbox_strategies.screening=["pooled_anonymous"]`; specialist probes capped at 1 recipe with no expansion; separate output root. The ExtraTrees member is required because the joint event track only searches the declared specialist families — it is a fixed compatibility reference, never a test-derived choice. **This control has 5 families vs the generator control’s 4**, so the two controls are not machine-identical references.

| track | macro MAE pp | macro RMSE pp | macro bias pp | pooled MAE pp | n |
| --- | --- | --- | --- | --- | --- |
| series (822) | 17.519 | 22.841 | -0.367 | 17.219 | 822 |
| joint event (137) | 18.201 | 26.663 | 5.797 | 19.097 | 137 |
| aggregation (137) | 13.840 | 17.590 | -1.804 | 11.960 | 137 |

| outer | control chosen | inner pp | series macro MAE pp |
| --- | --- | --- | --- |
| E1 | hist_gb | 13.346 | 16.439 |
| E2 | extra_trees | 20.906 | 18.364 |
| E3 | extra_trees | 21.626 | 17.752 |

**Finding: on the gearbox data the searched pipeline had a lower MAE than the fixed time/cadence-only reference on these three events (12.008 pp vs 17.519 pp series macro MAE).** This is the opposite direction of the generator study (searched 26.335 pp vs control 17.877 pp) and a statement about these two pipelines on these three events: recipes, feature spaces, tuning and family sets differ, so it does not establish that sensor features contain information in general, nor that time is causal in either dataset.

## 12. Independent recomputation

A separate numpy implementation (per-`Falha` metric, equal averages, then equal event averages) reproduced every stored per-event MAE and signed bias for the series, joint and aggregation tracks to <1e-9, plus the pooled values; stored `rmse_pp` is pointwise and is therefore cross-checked as the pooled-style quantity rather than against macro RMSE. Targets were re-read from `data/Features_gearbox_score_todos_RUL-porcentagem-data.csv`: **822/822 series `y_true` and 137/137 joint `y_true` values match (0 mismatches)**, the three test folds are disjoint and cover exactly 822 unique series endpoints (426/198/198), the joint track has 137 unique ids (71/33/33), and sampled candidate payload hashes match the canonical digest recomputed independently. `verify` (478 PASS, `RESULT: evaluated_complete`) re-checks fingerprints, CSV hash, event exclusion from training groups, payload integrity, coverage and metric reproducibility — a mechanical consistency check, **not** proof that no outer label influenced the inner search; that rests on the code path plus the synthetic permutation regression test.

## 13. Reproducibility

```
main search log   results/baselines/gearbox/main_search.log
main artifacts    results/baselines/gearbox/101e2f1fc3d406b1/{manifest.json,ledger.json,stage_costs.json,outer/[E1-E3]/{selection,trials,evaluation,predictions,candidates},outer/[E1-E3]/event/…,features,fits}
control config    results/baselines/gearbox/controls/config_cadence_elapsed_only.json
control artifacts results/baselines/gearbox/controls/gearbox/6932403970ae9c17/…
report            reports/baselines/gearbox.md (+ reports/baselines/README.md)
```

Re-running `search` on an existing run directory resumes and adds no duplicate fits; `evaluate` refuses to run before a frozen selection exists; `verify` re-validates hashes, payloads and coverage.

## 14. Limitations

- Three outer events only; each inner fold trains on a single event, so the design has very little diversity and the ±pp differences between families are not resolved.
- Sampling frames differ by track: 822 per-series endpoints, 137 joint event timestamps, 137 aggregated rows. They are reported separately and never pooled or ranked against each other.
- Availability: the joint track only contains timestamps where all six measurements exist with the same target and phase; 0 rows were dropped on this dataset, but the track describes synchronized operation, not every measurement timestamp.
- No MAPE, significance testing, universal-optimality or causal-sensor claims; no outer-chosen winner is presented as the “best family”.
- 49 recorded fit failures are not imputed.

## 15. Curvas (artefatos locais, descritivos)

Plots gerados **apenas** das predições OOF congeladas dos runs principal e de controle (nenhum ajuste novo). Eixo x = data do registro, y = RUL em %; preto = alvo, azul = escolha primária por série, laranja = controle tempo/cadência casado por endpoint/data/`Falha`, cinza tracejado = referência fixa `mean`. Nos painéis de evento: preto = alvo, azul = `joint` escolhido, verde pontilhado = agregação das 6 séries (regra escolhida no inner), laranja = agregação do controle — a trilha de 137 timestamps é separada das 822 séries.

Séries (6 painéis por evento, um por papel de medição):

- [gearbox_series_outerE1.png](../../results/baselines/gearbox/101e2f1fc3d406b1/plots/gearbox_series_outerE1.png)
- [gearbox_series_outerE2.png](../../results/baselines/gearbox/101e2f1fc3d406b1/plots/gearbox_series_outerE2.png)
- [gearbox_series_outerE3.png](../../results/baselines/gearbox/101e2f1fc3d406b1/plots/gearbox_series_outerE3.png)

Evento (um painel por evento):

- [gearbox_event_outerE1.png](../../results/baselines/gearbox/101e2f1fc3d406b1/plots/gearbox_event_outerE1.png)
- [gearbox_event_outerE2.png](../../results/baselines/gearbox/101e2f1fc3d406b1/plots/gearbox_event_outerE2.png)
- [gearbox_event_outerE3.png](../../results/baselines/gearbox/101e2f1fc3d406b1/plots/gearbox_event_outerE3.png)
