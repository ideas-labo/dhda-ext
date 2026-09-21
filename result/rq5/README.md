# RQ5: Model adaptation time

`normalized_runtime/` contains adaptation time normalized by the number of
timesteps in each stream. These are processed data used to reproduce the RQ5
means and error bars. The corresponding raw total times are in
`../rq1/runtime/`.

The exact overall means reproduce the paper's rounded values: DHDA 0.9532,
DHDA-ICSE 0.7366, SeMPL 1.2796, BEETLE 1.1275, ARF 0.6697, SRP 0.8267, DaL
2.5641, and SELeCT 0.1524 seconds per timestep.
