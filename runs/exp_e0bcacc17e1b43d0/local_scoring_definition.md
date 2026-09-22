# Local scoring definition

All definitions were frozen before future scoring. A trace is local-eligible only when raw positions 0..499 are finite and nonzero; otherwise local metrics are missing and fusion remains global. Exact signed maximal-run RLE of these 500 directions passed decode round-trip. The final observed run is retained exactly as in the prior primary definition and may be right-censored.

The primary scalar is the overlap-mass-weighted covering-run length on zero-based raw positions 50..99: `sum(overlap_packets * full_observed_run_length) / 50`. For each supported class, v3 supervised-train supplies the median and `1.4826*MAD`; the scale is floored by the median positive class scale and `1e-6`. Raw class score is `-|x-median_c|/scale_c`.

Classes 4, 8, 23 and 43 have no Day0 trace under the frozen 500-packet budget. No template is fabricated for them. Supported raw scores are class-standardized within the 98 supported classes; each unsupported class receives exactly zero local contribution. Fusion uses `Gz + alpha*L`, so unsupported classes retain their standardized global score. A local-ineligible trace receives all-zero local contribution and remains exactly global.

Alpha is selected only on official Day0 valid all-class accuracy from `[0,.05,.10,.20,.30,.50,.75,1]`, ties to smaller alpha. No margin gate, temperature, future batch statistic or future label is used. The source-defined bypass sensitivity disables fusion when frozen global top-1 is unsupported; it is not the primary rule.

Incoming fraction is `1-outgoing_fraction` and is not duplicated as a separate candidate. The two 50-packet neighboring run controls, the same-window direction controls and the prefix-150 direction controls use the identical template and fusion protocol.

| feature | pooled scale floor | chosen alpha | best Day0-valid accuracy |
|---|---:|---:|---:|
| run_mw_50_99 | 2.513007 | 0.00 | 0.724074 |
| run_mw_0_49 | 0.81543 | 0.00 | 0.724074 |
| run_mw_100_149 | 3.85476 | 0.00 | 0.724074 |
| direction_out_fraction_50_99 | 0.118608 | 0.10 | 0.725000 |
| direction_transition_density_50_99 | 0.15128571 | 0.00 | 0.724074 |
| prefix150_out_fraction | 0.059304 | 0.00 | 0.724074 |
| prefix150_transition_density | 0.077115101 | 0.00 | 0.724074 |
