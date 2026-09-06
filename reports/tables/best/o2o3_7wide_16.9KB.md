
### Individual runs

| run                 | accuracy | balanced | n     |
|---------------------|----------|----------|-------|
| r4_ctx2048_dense    | 70.81%   | 70.91%   | 55657 |
| r6_dense2048_seed7  | 71.56%   | 71.08%   | 55657 |
| r6_dense2048_seed13 | 69.42%   | 69.39%   | 55657 |
| r6_dense2048_seed29 | 69.64%   | 69.68%   | 55657 |
| r7_wide2048_seed7   | 72.76%   | 72.66%   | 55657 |
| r7_wide2048_seed13  | 71.50%   | 71.30%   | 55657 |
| r7_wide2048_seed29  | 71.67%   | 71.51%   | 55657 |

### Ensemble

| run                 | accuracy | balanced | n     |
|---------------------|----------|----------|-------|
| r4_ctx2048_dense    | 70.81%   | 70.91%   | 55657 |
| r6_dense2048_seed7  | 71.56%   | 71.08%   | 55657 |
| r6_dense2048_seed13 | 69.42%   | 69.39%   | 55657 |
| r6_dense2048_seed29 | 69.64%   | 69.68%   | 55657 |
| r7_wide2048_seed7   | 72.76%   | 72.66%   | 55657 |
| r7_wide2048_seed13  | 71.50%   | 71.30%   | 55657 |
| r7_wide2048_seed29  | 71.67%   | 71.51%   | 55657 |
| ensemble of 7       | 75.10%   | 74.95%   | 55657 |
| ensemble (log-mean) | 74.78%   | 74.60%   | 55657 |

### Context scaling — r4_ctx2048_dense

| run                          | accuracy | balanced | n     |
|------------------------------|----------|----------|-------|
| radius 16 (16896 bytes seen) | 79.19%   | 79.29%   | 55657 |

### Context scaling — r6_dense2048_seed7

| run                          | accuracy | balanced | n     |
|------------------------------|----------|----------|-------|
| radius 16 (16896 bytes seen) | 81.86%   | 81.16%   | 55657 |

### Context scaling — r6_dense2048_seed13

| run                          | accuracy | balanced | n     |
|------------------------------|----------|----------|-------|
| radius 16 (16896 bytes seen) | 75.66%   | 75.54%   | 55657 |

### Context scaling — r6_dense2048_seed29

| run                          | accuracy | balanced | n     |
|------------------------------|----------|----------|-------|
| radius 16 (16896 bytes seen) | 75.81%   | 75.80%   | 55657 |

### Context scaling — r7_wide2048_seed7

| run                          | accuracy | balanced | n     |
|------------------------------|----------|----------|-------|
| radius 16 (16896 bytes seen) | 76.44%   | 76.20%   | 55657 |

### Context scaling — r7_wide2048_seed13

| run                          | accuracy | balanced | n     |
|------------------------------|----------|----------|-------|
| radius 16 (16896 bytes seen) | 74.78%   | 74.48%   | 55657 |

### Context scaling — r7_wide2048_seed29

| run                          | accuracy | balanced | n     |
|------------------------------|----------|----------|-------|
| radius 16 (16896 bytes seen) | 77.13%   | 76.89%   | 55657 |

### Context scaling — ensemble

| run                          | accuracy | balanced | n     |
|------------------------------|----------|----------|-------|
| radius 16 (16896 bytes seen) | 81.10%   | 80.87%   | 55657 |

### Binary-level soft vote

| run                 | accuracy | balanced | n   |
|---------------------|----------|----------|-----|
| r4_ctx2048_dense    | 87.23%   | 87.23%   | 188 |
| r6_dense2048_seed7  | 85.11%   | 85.11%   | 188 |
| r6_dense2048_seed13 | 84.04%   | 84.04%   | 188 |
| r6_dense2048_seed29 | 84.57%   | 84.57%   | 188 |
| r7_wide2048_seed7   | 89.36%   | 89.36%   | 188 |
| r7_wide2048_seed13  | 89.89%   | 89.89%   | 188 |
| r7_wide2048_seed29  | 89.89%   | 89.89%   | 188 |
| ensemble            | 91.49%   | 91.49%   | 188 |
