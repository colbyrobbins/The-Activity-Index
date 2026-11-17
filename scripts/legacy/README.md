# Earlier studies (kept for the record)

These scripts are the first two stages of the project. They still run, but the final result does not depend on their outputs
(except `12_build_puma_panel.py`, which stayed in `scripts/` because the area builder reads its panel).

| Folder | Question | Outcome | Write-up |
|---|---|---|---|
| `tract_study/` | Do 2015-18 bikeshare features predict a tract's income-rank change between ACS 2014-18 and 2019-23? | No robust gain over free census data; about 60% of the target is survey noise. | `docs/project_summary.md`, `docs/eda_findings.md`, `docs/noise_ceiling.md` |
| `puma_study/` | Does a PUMA's yearly bikeshare change track its yearly income change (ACS 1-year, 2016-19)? | No bikeshare group beat the income-history baseline. | `docs/puma_study.md` |

Run them from the repo root, e.g. `python scripts/legacy/tract_study/10_baselines.py`.
