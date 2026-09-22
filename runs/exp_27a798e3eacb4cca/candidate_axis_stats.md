# Candidate axis statistics v1

## Raw abundance versus verified-condition coverage

Both source files cover all 102 labels. `train.npz` has 19,439 rows, 163–198 per website; `valid.npz` has 2,160 rows, 18–22 per website. Thus the files have ample numerical counts for 3-shot if one falsely treats file membership as condition. That calculation is deliberately not admitted: no source says the two files are different dates, sessions, runs, collectors or environments.

Under the conservative definition, the only verified historical acquisition condition is Day0. Therefore every website has exactly one verified condition, 0/102 have at least two, coverage is 0%, and no support→independent-condition query episode is feasible. There is no earlier→later direction within historical source. At most one could make an ordinary i.i.d. Day0 split, which is not temporal or condition shift.

The existing canonical source split has 18,553 admitted unique train rows after excluding 168 train rows whose admitted-input hash appears in official valid and 718 further train duplicate rows. It then randomly allocates, per class, 2 reference, 20 source-holdout and the remainder supervised-train rows. Those roles all originate in `train.npz` and are explicitly not condition labels.

Cross-site stability cannot rescue the gate: raw sample abundance is stable across all sites, but verified condition count is uniformly one. Detailed axis decisions are in `candidate_axis_stats.csv`; machine-readable values are in `summary.json`.
