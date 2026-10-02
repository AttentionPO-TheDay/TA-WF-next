# Fixed source/valid data preparation

Completed on CPU. Exactly the prior native experiment’s 15,300 source rows (150/class) and 510 valid rows (5/class), in unchanged row order and label mapping. No model forward, optimization, or scoring.

Finite and suffix-padding checks passed. Original signs, labels, row lists, raw archive hashes and prior cache hash match exactly. Cached TAM is independently reconstructed for every selected row, with per-row/per-direction packet conservation. Selected source/valid direction overlap is zero. The earlier full-valid overlap audit and frozen source exclusions are reused; full-valid directions are not reanalyzed here.

First 5000 stored-order signed timestamps are float32; no sorting or manufactured interarrival times. Mixed-direction absolute-time backtracking remains a recorded semantic limitation; per-direction order is checked. Times at or beyond 80 seconds accumulate in the final TAM bin, following the prior audited author rule.

NPZ access decompresses whole X/y members; only fixed selected rows are retained or analyzed. Full archive bytes are read for provenance hashes. This repeats authorized source/valid structure and label checks; it does not open future dates, WTT-Time, or AWF and does not add model-selection exposure. Detailed hashes, counts and mapping are in manifest.json.

Execution history: preparation completed twice, each using the identical 15,300 source and 510 valid rows. The second execution only corrected manifest direction names from assumed outgoing/incoming to positive/negative channels. Both prepared caches have the same SHA256. No new rows, labels, predictions, or future files were exposed in the second pass.

```json
{
  "source": {
    "rows": 15300,
    "width": 5000,
    "nonfinite": 0,
    "internal_zero_rows": 0,
    "packet_count": 22213300,
    "padding_count": 54286700,
    "ge80_packets": 1923663,
    "positive_packets": 4483199,
    "negative_packets": 17730101,
    "mixed": {
      "negative_deltas": 42250,
      "rows_with_negative_delta": 9762
    },
    "positive": {
      "negative_deltas": 0,
      "rows_with_negative_delta": 0
    },
    "negative": {
      "negative_deltas": 0,
      "rows_with_negative_delta": 0
    },
    "tam_independent_parity_rows": 15300,
    "prior_direction_labels_rows_exact": true,
    "per_row_channel_count_conservation": true
  },
  "valid": {
    "rows": 510,
    "width": 5000,
    "nonfinite": 0,
    "internal_zero_rows": 0,
    "packet_count": 729953,
    "padding_count": 1820047,
    "ge80_packets": 68340,
    "positive_packets": 149515,
    "negative_packets": 580438,
    "mixed": {
      "negative_deltas": 1484,
      "rows_with_negative_delta": 346
    },
    "positive": {
      "negative_deltas": 0,
      "rows_with_negative_delta": 0
    },
    "negative": {
      "negative_deltas": 0,
      "rows_with_negative_delta": 0
    },
    "tam_independent_parity_rows": 510,
    "prior_direction_labels_rows_exact": true,
    "per_row_channel_count_conservation": true
  }
}
```
