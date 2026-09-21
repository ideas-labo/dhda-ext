# RQ2: Different local models

The CSV files contain 30-run mMAPE distributions for DHDA paired with RF, HT,
kNN, and LR, together with their standalone counterparts. This data was stored
under the legacy name `rq3-fff`; the directory has been renamed to match the
current paper.

`dhda_reuse_ALL_model` is DHDA with the default Random Forest local model.
Columns prefixed with `DHDA-` are the corresponding DHDA-paired learners.

The DeepArch legacy CSV had the `DHDA-HT` and `HT` headers reversed. Its header
has been corrected so the method labels and values match the paper table; the
30 recorded values themselves were not changed.
