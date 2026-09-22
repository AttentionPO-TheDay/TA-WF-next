# Burst-Level Temporal Drift Characterization

## Verdict: SUPPORTS_FOLLOWUP

This verdict means only that a tightly scoped burst-structure hypothesis is worth a later test. It does not establish a method, innovation, deployable selector, or external confirmation, and it does not authorize training.

Across 136,621 canonical traces, exact signed maximal-run RLE had 0 round-trip failures. Direction-only whole-trace attributes did not pass the joint gate: the stable attributes were non-discriminative (`run_median` and outgoing p90 have median different-site distance 0), while useful-looking attributes exceeded their website-specific Day0 q95 too often. Examples are outgoing-run mean (stability 0.741, separation 4.21), transition density (0.616, 6.75), and adjacent log-run correlation (0.645, 7.72).

The limited positive evidence is positional and length-controlled. On the pre-registered 500-packet common budget, 96/102 websites have usable traces in all dates. For raw indices 50–99, covering-run mass-weighted length stays within Day0 q95 in 87.7% of website×date cells, while different-site/same-site distance ratios are Day14 6.20, Day30 6.42, Day90 4.73, Day150 3.27, Day270 2.43. Removing the terminal/right-censored run leaves this early window unchanged. Related run-length and transition measures show the same broad early-prefix pattern; relative-order aggregate shift is similar for K=5/10/20. Thus the evidence is not explained solely by trailing padding, unequal full trace length, the final censored run, or choosing K=10.

The evidence is still bounded. Full-L traces are only 4.2–5.2%; full-L-only all-site comparison is impossible. The equal-500 gate omits six websites and late-date stability weakens. Same burst percentile is not the same loading stage, and no resource/JS/ad mechanism is inferred. Future labels were used to establish same-site stability and different-site separation, so the indices/attribute ranking is descriptive development evidence and cannot be copied into a deployment rule.

One follow-up hypothesis is permitted, but not implemented here: learn a fixed early-prefix run-organization reliability model from historical labeled data, then let only the current **unlabeled** batch estimate distributional reliability/calibration before combining it with a packet-sequence predictor. The window/rule must be selected without current true labels and compared against length-matched packet-level controls; the present future-label ranking is evaluation evidence, not deployable information.

Timing is available but secondary: absolute timestamps show 0.12–0.22% adjacent reversals, and 55.0–57.6% of bursts have zero duration (almost all single-packet). Rates exclude duration<=0 and contain no infinities. No timing result is attributed to signed burst-size RLE.

## Completion checks

- Training/adaptation/GPU runs: 0 / 0 / no.
- Input data, formal v3 split, checkpoints and prior artifacts: read-only; verified data/split hashes match formal records.
- Future labels: descriptive same-site/different-site statistics only; no weights, thresholds, selectors or methods fitted.
- Required tables/reports: complete. Detailed per-site shifts, joint metrics and position rows remain in CSV rather than being hidden by plots.
- No Packet CNN/Burst CNN/DF comparison and no follow-on training Job created.

Host decision remains open: this technical verdict supports only a bounded follow-up test, not adoption of the route.
