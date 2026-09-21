# RQ1: State-of-the-art comparison

`accuracy/` contains per-run mMAPE for DHDA, SeMPL, BEETLE, ARF, SRP, DaL,
and SELeCT. Each method has 30 values for each of the 12 paper systems. The
existing six methods retain their original full-precision values; the
`select_regressor` column contains the subsequently recovered SELeCT results.

`runtime/` contains raw total adaptation time per run. Its `select_regressor`
column is the SELeCT timing result formerly stored separately under `rq1-add`.
All 360 SELeCT timing values were verified as exact matches before publication
in this directory.

`stats/all_systems_result.csv` is generated from the 12 files in `accuracy/`.
Each cell has the form `rank_median_(IQR)`. Ranks use the parametric
Scott-Knott ESD 2.x procedure used by the experiment, with lower mMAPE receiving
the better rank. The implementation follows the original `ScottKnottESD`
parametric partition rule and Cohen's d negligible-effect threshold (`|d| <
0.2`) without requiring R. Median and IQR are calculated directly from the 30
runs.

Regenerate the table from the repository root:

```bash
python3 raw-data/scripts/build_rq1_stats.py
```

Check that the committed table is current without modifying it:

```bash
python3 raw-data/scripts/build_rq1_stats.py --check
```

Column names retain their experimental-code names to preserve the raw files:

| Column | Paper label |
| --- | --- |
| `dhda_reuse_ALL_model` | DHDA |
| `dhda_model8` | DHDA-ICSE |
| `SeMLP` | SeMPL |
| `select_regressor` | SELeCT |
