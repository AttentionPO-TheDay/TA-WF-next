# 部署评分及风险转移

阈值只由reference和获准sourcevalid生成，无query拟合。NAR不能因采用极端分位而得到保证。

| Slice | Candidate | Policy | ΔF1 pp | NAR % | Recovery % | Source NAR % | Ref accepted | Query accepted |
|---|---|---|---:|---:|---:|---:|---:|---:|
| version_045 | frozen | margin_ref_rank_source1 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.0 | 0.0 |
| version_045 | frozen | extreme_q95 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.0 | 0.0 |
| version_045 | frozen | extreme_q99 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.0 | 0.0 |
| version_045 | frozen | agreement_q95 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.0 | 0.0 |
| version_045 | v6_lora | margin_ref_rank_source1 | 1.808±0.777 | 0.710±0.186 | 5.055±2.186 | 0.524±0.291 | 54.0 | 1311.7 |
| version_045 | v6_lora | extreme_q95 | 0.338±0.138 | 0.067±0.082 | 0.824±0.217 | 0.000±0.000 | 6.0 | 173.0 |
| version_045 | v6_lora | extreme_q99 | 0.067±0.037 | 0.036±0.049 | 0.170±0.089 | 0.000±0.000 | 2.0 | 36.7 |
| version_045 | v6_lora | agreement_q95 | 0.257±0.171 | 0.057±0.072 | 0.615±0.310 | 0.000±0.000 | 4.3 | 120.0 |
| version_045 | no_teacher_kl | margin_ref_rank_source1 | 0.907±0.821 | 0.419±0.483 | 2.651±2.528 | 0.158±0.158 | 28.0 | 753.0 |
| version_045 | no_teacher_kl | extreme_q95 | 0.089±0.050 | 0.016±0.008 | 0.286±0.101 | 0.000±0.000 | 3.3 | 80.3 |
| version_045 | no_teacher_kl | extreme_q99 | 0.013±0.020 | 0.005±0.005 | 0.075±0.062 | 0.000±0.000 | 1.0 | 28.7 |
| version_045 | no_teacher_kl | agreement_q95 | 0.013±0.022 | 0.005±0.009 | 0.035±0.061 | 0.000±0.000 | 0.7 | 5.7 |
| version_045 | confident_teacher_kl | margin_ref_rank_source1 | 1.247±1.230 | 0.462±0.550 | 3.627±3.592 | 0.351±0.343 | 40.3 | 1003.7 |
| version_045 | confident_teacher_kl | extreme_q95 | 0.235±0.064 | 0.016±0.008 | 0.649±0.242 | 0.018±0.030 | 4.7 | 136.7 |
| version_045 | confident_teacher_kl | extreme_q99 | 0.048±0.070 | 0.003±0.004 | 0.120±0.163 | 0.018±0.030 | 1.0 | 21.0 |
| version_045 | confident_teacher_kl | agreement_q95 | 0.142±0.078 | 0.005±0.005 | 0.366±0.196 | 0.018±0.030 | 2.7 | 74.0 |
| version_045 | geometry_grad_balance | margin_ref_rank_source1 | 1.051±1.820 | 0.421±0.730 | 3.110±5.386 | 0.296±0.513 | 32.3 | 789.7 |
| version_045 | geometry_grad_balance | extreme_q95 | 0.354±0.148 | 0.059±0.035 | 0.928±0.370 | 0.000±0.000 | 6.3 | 181.0 |
| version_045 | geometry_grad_balance | extreme_q99 | 0.081±0.021 | 0.021±0.029 | 0.224±0.089 | 0.000±0.000 | 2.0 | 44.7 |
| version_045 | geometry_grad_balance | agreement_q95 | 0.268±0.139 | 0.049±0.058 | 0.673±0.274 | 0.000±0.000 | 4.3 | 117.7 |
| network_USA | frozen | margin_ref_rank_source1 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.0 | 0.0 |
| network_USA | frozen | extreme_q95 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.0 | 0.0 |
| network_USA | frozen | extreme_q99 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.0 | 0.0 |
| network_USA | frozen | agreement_q95 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.0 | 0.0 |
| network_USA | v6_lora | margin_ref_rank_source1 | 1.610±0.926 | 1.450±0.667 | 5.572±2.738 | 0.735±0.191 | 76.7 | 213.0 |
| network_USA | v6_lora | extreme_q95 | 0.186±0.112 | 0.022±0.038 | 0.431±0.302 | 0.017±0.030 | 5.7 | 10.3 |
| network_USA | v6_lora | extreme_q99 | 0.054±0.051 | 0.000±0.000 | 0.103±0.090 | 0.000±0.000 | 1.7 | 2.0 |
| network_USA | v6_lora | agreement_q95 | 0.153±0.110 | 0.022±0.038 | 0.330±0.303 | 0.000±0.000 | 4.0 | 6.7 |
| network_USA | no_teacher_kl | margin_ref_rank_source1 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.0 | 0.0 |
| network_USA | no_teacher_kl | extreme_q95 | 0.045±0.049 | 0.000±0.000 | 0.155±0.135 | 0.000±0.000 | 2.0 | 4.7 |
| network_USA | no_teacher_kl | extreme_q99 | 0.036±0.044 | 0.000±0.000 | 0.130±0.121 | 0.000±0.000 | 1.0 | 4.3 |
| network_USA | no_teacher_kl | agreement_q95 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.0 | 0.0 |
| network_USA | confident_teacher_kl | margin_ref_rank_source1 | 0.963±1.061 | 0.833±0.856 | 3.258±3.296 | 0.386±0.338 | 41.7 | 122.3 |
| network_USA | confident_teacher_kl | extreme_q95 | 0.143±0.042 | 0.023±0.040 | 0.410±0.114 | 0.017±0.030 | 4.3 | 9.7 |
| network_USA | confident_teacher_kl | extreme_q99 | 0.032±0.037 | 0.000±0.000 | 0.078±0.080 | 0.017±0.030 | 1.0 | 2.0 |
| network_USA | confident_teacher_kl | agreement_q95 | 0.082±0.061 | 0.000±0.000 | 0.206±0.117 | 0.017±0.030 | 2.3 | 4.3 |
| network_USA | geometry_grad_balance | margin_ref_rank_source1 | 1.463±1.304 | 1.133±0.561 | 4.695±3.268 | 0.525±0.264 | 58.3 | 169.3 |
| network_USA | geometry_grad_balance | extreme_q95 | 0.220±0.043 | 0.000±0.000 | 0.485±0.149 | 0.035±0.030 | 5.3 | 11.7 |
| network_USA | geometry_grad_balance | extreme_q99 | 0.065±0.058 | 0.000±0.000 | 0.154±0.151 | 0.017±0.030 | 1.3 | 4.7 |
| network_USA | geometry_grad_balance | agreement_q95 | 0.174±0.004 | 0.000±0.000 | 0.384±0.068 | 0.018±0.030 | 3.7 | 9.3 |
