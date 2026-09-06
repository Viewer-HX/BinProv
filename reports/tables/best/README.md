# Result table index

This directory contains the detailed measurements referenced by
[`docs/RESULTS.md`](../../../docs/RESULTS.md).

| file | task and configuration | contents |
|---|---|---|
| [`o2o3_7wide_512B.md`](o2o3_7wide_512B.md) | O2 versus O3; seven 2048-byte models evaluated at a 512-byte target stride | Per-model sequence metrics, 7-model ensemble metrics, and binary soft-vote results. The ensemble reaches 75.10% sequence accuracy and 91.49% binary accuracy. |
| [`o2o3_7wide_16.9KB.md`](o2o3_7wide_16.9KB.md) | The same seven O2/O3 models with a radius-16 probability window | Per-model context-aggregated metrics and the 7-model ensemble result. The ensemble reaches 81.10% sequence accuracy. |
| [`o2o3_pretrained_wide_mean.md`](o2o3_pretrained_wide_mean.md) | O2 versus O3; three seeds initialized from the continued 2048-byte MLM | Per-seed sequence accuracy plus the mean and standard deviation: 71.98% ± 0.68 percentage points. |
| [`opt4_3wide_binary.md`](opt4_3wide_binary.md) | O0/O1/O2/O3; three 2048-byte models | Per-model sequence and binary metrics, 3-model ensemble metrics, and binary soft voting. The ensemble reaches 84.42% sequence accuracy and 95.21% binary accuracy. |
| [`opt4_6run_16.9KB.md`](opt4_6run_16.9KB.md) | O0/O1/O2/O3; three 512-byte and three 2048-byte models with radius-16 aggregation | Per-model metrics, 6-model ensemble metrics, context-aggregated results, and binary soft voting. The ensemble reaches 87.62% sequence accuracy and 94.41% binary accuracy. |
