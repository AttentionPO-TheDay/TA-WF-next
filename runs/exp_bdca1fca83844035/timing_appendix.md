# Timing appendix

Timing is separate from all direction-only conclusions. Formal project documentation identifies X as signed relative packet timestamp, and finite values are available. `abs(X)` nevertheless has small local reversals (table), so timing results are exploratory rather than a semantically clean primary layer.

| date | adjacent absolute-time reversals | single-packet bursts | duration <= 0 bursts | median trace-level inter-burst gap | median finite size/duration rate |
|---|---:|---:|---:|---:|---:|
| day0 | 0.191% | 55.935% | 55.935% | 0.01969 | 59.12 |
| day14 | 0.195% | 56.061% | 56.061% | 0.01897 | 61.79 |
| day30 | 0.217% | 55.974% | 55.974% | 0.02029 | 56.64 |
| day90 | 0.122% | 55.896% | 55.896% | 0.01414 | 78.17 |
| day150 | 0.151% | 55.038% | 55.038% | 0.01230 | 91.79 |
| day270 | 0.170% | 57.591% | 57.591% | 0.01662 | 72.12 |

Burst duration is `last_abs_timestamp - first_abs_timestamp`; inter-burst gap is `next_first - current_last`. Single-packet bursts have duration 0. Rates are calculated only where duration > 0; no infinite value is constructed. Zero-duration proportions (55.0–57.6%) are explicitly excluded from rate summaries. The near equality of single-packet and zero-duration fractions shows the exclusion is overwhelmingly the registered single-packet case (Day270 has one additional zero-duration multi-packet burst). Timing does not support or explain the direction-only burst-size finding.
