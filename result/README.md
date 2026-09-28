# DHDA experimental data

## Directory layout

```text
raw-data/
├── MANIFEST.csv
├── README.md
├── paper-summary/
│   ├── paper-consistency.csv
│   ├── paper-consistency.md
│   └── rq1_reported_statistics.csv
├── rq1/
│   ├── README.md
│   ├── accuracy/
│   ├── runtime/
│   └── stats/
├── rq2/
│   ├── README.md
│   └── accuracy/
├── rq3/
│   ├── README.md
│   └── accuracy/
├── rq4/
│   ├── README.md
│   ├── accuracy/
│   └── runtime/
├── rq5/
│   ├── README.md
│   └── normalized_runtime/
└── scripts/
    ├── build_rq1_stats.py
    └── verify_data.py
```

Each system CSV contains 30 experimental runs. CSV files are byte-for-byte
copies of the selected legacy files. `MANIFEST.csv` records row counts, headers,
SHA-256 checksums, and legacy source paths.

## Paper-to-data mapping

| Paper question | Contents | Legacy source |
| --- | --- | --- |
| RQ1 | State-of-the-art accuracy and raw adaptation time | `rq1-fff/csv`, merged time files in `rq5-time/csv` |
| RQ2 | DHDA with different local models | `rq3-fff/csv` |
| RQ3 | Component ablations | `rq2-fff/csv` |
| RQ4 | Sensitivity to alpha | `rq4-sensitive/accuracy`, `rq4-sensitive/time` |
| RQ5 | Adaptation time normalized by stream length | `rq5-time/normalized_csv` |


## Verification

Run from the repository root:

```bash
python3 raw-data/scripts/verify_data.py
```

The script verifies the manifest, checks that the shared DHDA results are
identical across RQ1--RQ3, and compares reproducible medians and IQRs with the
paper tables. Known paper/data discrepancies are documented in
`paper-summary/paper-consistency.md`.

## Scope

The paper evaluates 12 systems.
