# Local Burst Evidence Error-Correction Diagnostic

## Technical verdict: STOP_LOCAL_CORRECTION_ROUTE

This is a frozen-rule TemporalDrift development diagnostic, not an external confirmation and not a Host research-route decision. No backbone or local model was trained, fine-tuned or adapted.

The formal DF source-only head had aggregate future accuracy 0.580191. The Day0-only static fusion selected alpha 0.00 for the frozen 50–99 run-length scalar and reached 0.580191 (delta +0.000000). It produced 0 wrong→correct, 0 correct→wrong, 0 wrong→different-wrong and Net Repair 0. Positive Net Repair occurred on 0/5 dates and 0 sites after aggregating dates.

Classes 4, 8, 23 and 43 had no Day0 template under the frozen common 500-packet budget and received exactly zero additive local contribution. Across 4656 future samples whose true class was unsupported, fusion caused 0 correct→wrong transitions out of 3592 originally correct predictions, repaired 0, and had Net Repair 0; all damage counts fully against the all-class result. The label-free global-unsupported bypass sensitivity had Net Repair 0 and delta +0.000000; it is not the primary rule.

The primary local scalar was available on 96,037/118,068 future samples (81.34%). Its all-eligible true-class rank was mean 39.90, median 35, with top-1/top-5/top-10 hit 1.34%/7.80%/15.85%. On 40,920 eligible global errors, global true-class rank was mean 10.49 and median 4, whereas local rank was mean 42.67 and median 38. Local rank improved for 4,556 (11.13%), was unchanged for 553, and worsened for 35,811; it pushed 894 errors into local top-5 and 1,123 into top-10, but the broad rank comparison fails the complementarity gate.

The strongest specificity controls by Net Repair were tied at 0 because both neighboring run windows and all but one direction control independently selected alpha 0 on Day0 valid. `direction_out_fraction_50_99` selected alpha 0.1 and changed future predictions, but its 1,434 repairs were outweighed by 1,462 damages (Net Repair -28; delta -0.0237 pp). The primary-minus-best-control gap was 0 against the preregistered required gap 10; there is no evidence for run-length specificity or even positive early-traffic correction.

| date | wrong→correct | correct→wrong | wrong→different-wrong | Net Repair | global acc | fused acc | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| day14 | 0 | 0 | 0 | 0 | 0.712560 | 0.712560 | +0.000000 |
| day30 | 0 | 0 | 0 | 0 | 0.650982 | 0.650982 | +0.000000 |
| day90 | 0 | 0 | 0 | 0 | 0.565474 | 0.565474 | +0.000000 |
| day150 | 0 | 0 | 0 | 0 | 0.505028 | 0.505028 | +0.000000 |
| day270 | 0 | 0 | 0 | 0 | 0.460747 | 0.460747 | +0.000000 |

The verdict follows only the frozen gates in `CORRECTION_PLAN_v1.md` and the pre-future-scoring neutral-class amendment `CORRECTION_PLAN_v2.md`. Even a positive follow-up verdict would mean only that conflict reliability deserves study; it would not establish a new adaptation method. Future labels were used only after prediction for the reported development metrics and attribution. No selector, TTA or follow-on training Job was created.

Detailed ranks, confusion-pair attribution, all transition counts, specificity controls and per-site/date results are retained in the required CSV files. A limitation is that local evidence is available only to traces with the common 500-packet budget; ineligible samples retain the global decision. Scalar class templates cannot represent multimodal within-class local structure, by design of this simple diagnostic.
