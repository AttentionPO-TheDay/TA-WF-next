# Episode feasibility v1

| Proposed construction | Real change semantics | 102-site coverage | 3-shot support | Independent query condition | Direction | Feasible? |
|---|---|---:|---:|---|---|---|
| Day0 train → Day0 valid | not established; release split roles only | 0% verified | numerically yes | no verified condition independence | none | no |
| current random source roles | none; RNG split of canonical rows | 0% | yes | no | none | no |
| row-index blocks | undocumented serialization | 0% | likely | no | none | no |
| trace-length or packet-time bins | content/input feature | 0% | likely | no | within trace only | no |
| random crop/delete/perturb | synthetic augmentation | 0% | configurable | synthetic only | none | forbidden as gate evidence |
| TAM/raw or other preprocessing pair | same trace derivative | 0% | configurable | violates source-family isolation | none | no |
| Day0 → later TemporalDrift date | genuine temporal axis in corpus | 100% at corpus level | source has ample counts | later labels/files are outside historical-task definition and forbidden for this gate | earlier→later | out of scope, not admissible |

No conservative episode constructor remains. In particular, `train`→`valid` would be an i.i.d. source split with unknown grouping, not a condition-shift episode. Random augmentation may later be studied as regularization, but it supplies no evidence that historical data contains real website variation.

Verdict: **FAIL**.
