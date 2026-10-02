# RF preparation completed

All fixed-row label/direction, padding, finite, per-channel count conservation, scalar author-algorithm parity and synthetic CPU forward checks passed. No optimizer, training or future access. Official model unchanged except RFNative factory.

Timing-inclusive condition uses observed signed absolute timestamps without sorting. Mixed-direction backtracking remains a known data-semantic limitation; within-direction statistics are recorded, not corrected. Clipping >=80 seconds is author-defined last-bin accumulation. This interface does not establish timing causal validity or performance.

```json
{
  "source": {
    "rows": 15300,
    "width": 5000,
    "nonfinite": 0,
    "internal_zero_rows": 0,
    "padding_count": 54286700,
    "packet_count": 22213300,
    "ge80_packets": 1923663,
    "ge80_rows": 14798,
    "mixed": {
      "adjacent_pairs": 22198000,
      "negative_deltas": 42250,
      "negative_delta_fraction": 0.0019033246238399856,
      "rows_with_negative_delta": 9762,
      "row_fraction": 0.6380392156862745
    },
    "positive": {
      "adjacent_pairs": 4467899,
      "negative_deltas": 0,
      "negative_delta_fraction": 0.0,
      "rows_with_negative_delta": 0,
      "row_fraction": 0.0
    },
    "negative": {
      "adjacent_pairs": 17714801,
      "negative_deltas": 0,
      "negative_delta_fraction": 0.0,
      "rows_with_negative_delta": 0,
      "row_fraction": 0.0
    },
    "scalar_parity_rows": 64,
    "count_conservation": true
  },
  "valid": {
    "rows": 510,
    "width": 5000,
    "nonfinite": 0,
    "internal_zero_rows": 0,
    "padding_count": 1820047,
    "packet_count": 729953,
    "ge80_packets": 68340,
    "ge80_rows": 494,
    "mixed": {
      "adjacent_pairs": 729443,
      "negative_deltas": 1484,
      "negative_delta_fraction": 0.0020344290095319307,
      "rows_with_negative_delta": 346,
      "row_fraction": 0.6784313725490196
    },
    "positive": {
      "adjacent_pairs": 149005,
      "negative_deltas": 0,
      "negative_delta_fraction": 0.0,
      "rows_with_negative_delta": 0,
      "row_fraction": 0.0
    },
    "negative": {
      "adjacent_pairs": 579928,
      "negative_deltas": 0,
      "negative_delta_fraction": 0.0,
      "rows_with_negative_delta": 0,
      "row_fraction": 0.0
    },
    "scalar_parity_rows": 64,
    "count_conservation": true
  },
  "synthetic_scalar_parity": true,
  "cpu_forward_shape": [
    2,
    102
  ],
  "parameters": 1040562,
  "future_access": false,
  "sorting": false,
  "training": false,
  "elapsed_seconds": 8.078544072806835
}
```
