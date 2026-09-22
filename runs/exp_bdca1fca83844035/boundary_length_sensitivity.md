# Boundary and length sensitivity

| primary attribute (best stability first) | fraction within Day0 q95 | median same-site shift | median different-site distance | separation ratio |
|---|---:|---:|---:|---:|
| run_median | 1.000 | 0.000 | 0.000 | 0.000 |
| outgoing_run_p90 | 0.914 | 0.000 | 0.000 | 0.000 |
| outgoing_run_mean | 0.741 | 0.026 | 0.108 | 4.211 |
| run_p90 | 0.676 | 0.950 | 4.000 | 4.316 |
| run_max | 0.659 | 2.000 | 6.500 | 3.500 |
| incoming_run_p90 | 0.647 | 1.000 | 4.500 | 4.900 |
| adjacent_log_length_corr | 0.645 | 0.014 | 0.107 | 7.716 |
| outgoing_packet_fraction | 0.639 | 0.013 | 0.069 | 5.241 |

The two globally most stable attributes are degenerate: `run_median` has stability 1.000 but both same-site and different-site median distances are 0; outgoing run p90 has stability 0.914 but different-site median distance is also 0. No primary whole-trace attribute meets the pre-registered joint gate.

Removing the L-boundary terminal run leaves the leading rankings and values essentially unchanged. A full-L-only comparison is not valid across all sites: just 833–1,184 traces/date reach L and at least one website is absent. Day0 length-quartile edges are 580/1037/1738 packets. Fixed 500-packet analysis retains 14,876 Day0 and 16,393–23,048 future traces but only 96/102 websites in every date.

| equal-500 whole-prefix attribute | fraction within Day0 q95 | separation ratio |
|---|---:|---:|
| outgoing_run_mean | 0.819 | 2.496 |
| adjacent_log_length_corr | 0.800 | 3.940 |
| burst_count | 0.787 | 4.200 |
| transition_density | 0.781 | 4.200 |

The strongest non-mechanical positional result is raw packet indices 50–99 in the fixed 500-packet sensitivity: mass-weighted covering-run length stays within the website-specific Day0 q95 in 0.877 of website×date cells. Its different-site/same-site ratios are day14=6.20, day30=6.42, day90=4.73, day150=3.27, day270=2.43. Removing each 500-prefix terminal run gives stability 0.877 and the same ratios for this early window, so it is not caused by the terminal segment. This result covers 96 websites, not all 102.

Relative burst-order K=5/10/20 gives median site×date×bin shifts 0.0350/0.0338/0.0333, so the aggregate shift scale is not driven by choosing K=10. Nevertheless, burst percentiles are not webpage stages, and no ad/JS/async-loading interpretation is made. The fixed-index early-window result avoids burst-percentile bin mechanics; it remains descriptive because its stability ranking uses future labels.
