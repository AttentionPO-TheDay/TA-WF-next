# Data and observation-boundary audit

All hashes were checked against the formal v3 manifest. `train/day14/day30/day90/day150/day270` SHA-256 values are respectively `2994271…`, `eaa52ab…`, `16b8bd90…`, `eaa25c23…`, `9fbc3116…`, `35e965b9…`; `splits_v3.json` is `0f322e9a…`. Arrays are `float64 X[N,10000]` and `float64 y[N]`; labels are integral 0–101. Only `sign(X[:5000])` enters direction results.

| date | raw rows | canonical/analyzed | per-site n | valid length p10/p50/p90 | reaches L=5000 | right-censored terminal |
|---|---:|---:|---:|---:|---:|---:|
| day0 | 19,439 | 18,553 | 156–190 | 356/1037/3458 | 926 (5.0%) | 752 (4.1%) |
| day14 | 22,603 | 22,603 | 201–236 | 362/1059/3578 | 1,184 (5.2%) | 945 (4.2%) |
| day30 | 22,867 | 22,867 | 202–235 | 364/1079/3625 | 1,141 (5.0%) | 921 (4.0%) |
| day90 | 28,599 | 28,599 | 239–293 | 346/1119/3291 | 1,102 (3.9%) | 837 (2.9%) |
| day150 | 24,064 | 24,064 | 203–496 | 365/1169/3615 | 1,135 (4.7%) | 876 (3.6%) |
| day270 | 19,935 | 19,935 | 85–204 | 358/1125/3253 | 833 (4.2%) | 635 (3.2%) |

Day0 uses exactly the 18,553-row union of the formal v3 source roles. Future date-internal admitted-input hashing removed zero rows and found zero cross-label conflicts. Every analyzed row had a nonempty finite prefix, zero leading/interior zero positions before its final nonzero value, and only a trailing zero suffix; therefore trailing zeros are treated as padding. Full-L traces are only 4.2–5.2% per date. At L, 3.2–4.2% of all traces continue in the same direction and are marked right-censored; another 0.9–1.3% have an observed direction transition immediately after L.

Class/date counts are not equal (per-site ranges above, notably Day150 max 496 and Day270 min 85). Primary aggregation is website-level, Day0 noise is estimated per website, and the fixed equal-500 sensitivity uses only the 96 websites represented by traces of length at least 500 in every date. The complete per-site counts are in `artifacts/summary_raw.json`.
