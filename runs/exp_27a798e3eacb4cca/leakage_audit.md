# Leakage audit v1

## Reused formal evidence

The v3 source manifest hashes the admitted representation as SHA-256 of `int8 sign(X[:5000])`. It excludes 168 train rows overlapping official valid and 718 additional duplicate train rows, retaining 18,553 canonical train rows. The v4 content audit reports 18,553/18,553 unique hashes for both raw full rows and admitted inputs across the three derived source roles, zero cross-role duplicate groups, and zero remaining admitted-input intersections with valid. The later recoverability audit reports `valid` itself has 2,151 unique content groups among 2,160 rows (9 duplicate rows), zero content intersection among admitted source roles, and passed its declared exact-hash checks.

These checks are useful but insufficient for episode provenance: exact equality cannot identify two slices, crops, perturbations, retransformed views, or repeated captures from one session when no parent/session ID exists. Hash uniqueness proves only byte/input uniqueness under the chosen representation, not independent acquisition.

## Required rules for any future episode dataset

1. Group by immutable raw capture/trace ID before any split; every derivative, slice, crop, padding variant, TAM/raw view and augmentation descendant inherits that group.
2. Group by session/crawl/run/client/collector where repeated traces can share acquisition context; a group may appear on only one side of an episode.
3. Conditions must be sourced from documented acquisition metadata, not inferred from row order, file role or model features.
4. Perform exact raw-byte and admitted-input hashes across all sides; conflicting-label duplicates are fatal, same-label duplicates stay in one group.
5. Add a declared near-duplicate rule appropriate to signed timestamp traces and validate it without query-label tuning; near-duplicate clusters stay on one side.
6. Freeze the condition map, site map, group map and support/query manifests before training or scoring. Query labels never choose axes, thresholds or exclusions.
7. Support has at most the current 3 labeled rows/site; query must retain at least one independent raw group/site in a different verified condition.

Because the released historical source lacks the IDs needed for rules 1–3, no episode-level no-leakage guarantee can currently be made. This is an independent reason the gate cannot PASS or become CONDITIONAL merely from high row counts.
