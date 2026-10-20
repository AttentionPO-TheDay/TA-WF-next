# 原始与源校准输出

| Method | Policy | ΔF1 pp | NAR % | Recovery % |
|---|---|---:|---:|---:|
| ungated_lora | raw | 0.948±0.055 | 3.559±0.204 | 4.800±0.218 |
| ungated_lora | source_nar1 | 0.555±0.481 | 1.617±1.401 | 2.591±2.271 |
| ungated_lora | source_nar2 | 0.566±0.490 | 1.624±1.407 | 2.639±2.305 |
| ungated_residual | raw | 0.771±0.204 | 2.227±0.226 | 3.450±0.413 |
| ungated_residual | source_nar1 | 0.354±0.386 | 0.796±0.756 | 1.404±1.388 |
| ungated_residual | source_nar2 | 0.355±0.389 | 0.804±0.766 | 1.406±1.392 |
| tent_reference | raw | 1.983±0.207 | 8.513±0.109 | 8.546±0.204 |
| tent_reference | source_nar1 | 0.164±0.203 | 0.292±0.255 | 0.836±0.859 |
| tent_reference | source_nar2 | 0.294±0.413 | 0.422±0.392 | 1.404±1.720 |
| direct_rerank | raw | 0.799±0.088 | 4.807±0.256 | 5.377±0.100 |
| direct_rerank | source_nar1 | 0.431±0.379 | 1.886±1.634 | 2.429±2.106 |
| direct_rerank | source_nar2 | 0.514±0.447 | 2.836±2.458 | 3.287±2.846 |
| hard_band | raw | 0.943±0.056 | 3.574±0.215 | 4.774±0.176 |
| hard_band | source_nar1 | 0.603±0.526 | 1.736±1.504 | 2.763±2.393 |
| hard_band | source_nar2 | 0.612±0.532 | 1.750±1.516 | 2.809±2.434 |
| soft_band | raw | 0.940±0.030 | 3.515±0.210 | 4.704±0.204 |
| soft_band | source_nar1 | 0.602±0.395 | 1.773±1.395 | 2.711±1.938 |
| soft_band | source_nar2 | 0.608±0.400 | 1.778±1.400 | 2.737±1.968 |
| soft_geometry | raw | 0.775±0.052 | 2.783±0.288 | 3.787±0.150 |
| soft_geometry | source_nar1 | 0.484±0.300 | 1.735±0.494 | 2.134±1.282 |
| soft_geometry | source_nar2 | 0.484±0.300 | 1.735±0.494 | 2.134±1.282 |
