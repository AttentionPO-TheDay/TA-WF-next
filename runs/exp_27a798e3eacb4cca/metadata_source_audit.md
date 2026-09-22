# Metadata source audit v1

## Stored source schema

`TemporalDrift/train.npz` and `valid.npz` contain only `X` and `y`. The formal local NPZ audit records `X` as `(19439,10000)` and `(2160,10000)` float64 arrays and `y` as float64 integer-valued labels 0–101. `X` is a signed packet timestamp sequence: absolute value is time relative to the beginning of that one trace, sign is direction, and the tail is zero padding. It is not wall-clock capture time and cannot order different visits.

The release-level audit explicitly reports no sample/session/capture timestamp/collector ID. There is also no persisted raw filename, pcap ID, run ID, network path, VM/client ID, browser instance, or augmentation-parent ID. A row is locally addressable only as `<NPZ filename>:<zero-based row index>`; this is an audit key, not acquisition metadata.

## Candidate fields and provenance

| Candidate | Stored/source | Verified meaning | Decision |
|---|---|---|---|
| `y` | NPZ | website class 0–101; paper protocol supports a fixed 102-site set, while the stable site map is not released | class key only, not a condition |
| signed values in `X` | NPZ + official processor audit | direction plus within-trace relative packet timestamp | input feature, not collection date/session |
| zero padding / effective length | derived from `X` | trace length under collection/preprocessing | content-dependent feature; not an acquisition condition |
| NPZ file (`train` vs `valid`) | release files | Day0 source training and checkpoint-selection roles | split role only; no evidence of distinct time/session/environment |
| row index/order | container | serialized position | no documented temporal or run ordering |
| filesystem mtime / ZIP member metadata | local copy/container | local artifact/container metadata | not acquisition metadata |
| date | paper/file family | Day0 is 2024-03-13; future files are later collection dates | only one historical source date is available under this Job's boundary |
| site/session/capture/run/client/VM/network environment | absent | paper states SG cloud collection and common Tor setup at corpus level | no row-level condition identifier |
| original file batch/repeated collection ID | absent | no mapping from rows to original pcap/log/batch | unavailable |
| augmentation/derivation parent | absent for raw NPZ | no row-level derivation graph | unavailable; TAM is derived and excluded as evidence |

The paper-level provenance supports real drift across Day0/14/30/90/150/270, but those later dates are evaluation/development arrays, not historical source conditions permitted to define this training task. Their existence does not create multiple conditions inside Day0 source.

Evidence: `configs/datasets.json`; `scripts/run_temporal_screening.py` (`sample_id`, split rule, `provenance_limit`); `splits_v3.json`; old formal `proteus_six_dataset_protocol_audit.md` sections 2–4; `TemporalDrift_Provenance_Followup_2026-09-12.md`.
