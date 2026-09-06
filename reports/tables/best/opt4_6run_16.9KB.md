
### Individual runs

| run                 | accuracy | balanced | n      |
|---------------------|----------|----------|--------|
| r9_opt4_base_seed7  | 81.03%   | 79.77%   | 116321 |
| r9_opt4_base_seed13 | 81.11%   | 79.81%   | 116321 |
| r9_opt4_base_seed29 | 80.98%   | 79.71%   | 116321 |
| r9_opt4_wide_seed7  | 83.77%   | 82.58%   | 116321 |
| r9_opt4_wide_seed13 | 83.61%   | 82.42%   | 116321 |
| r9_opt4_wide_seed29 | 83.85%   | 82.76%   | 116321 |

### Ensemble

| run                 | accuracy | balanced | n      |
|---------------------|----------|----------|--------|
| r9_opt4_base_seed7  | 81.03%   | 79.77%   | 116321 |
| r9_opt4_base_seed13 | 81.11%   | 79.81%   | 116321 |
| r9_opt4_base_seed29 | 80.98%   | 79.71%   | 116321 |
| r9_opt4_wide_seed7  | 83.77%   | 82.58%   | 116321 |
| r9_opt4_wide_seed13 | 83.61%   | 82.42%   | 116321 |
| r9_opt4_wide_seed29 | 83.85%   | 82.76%   | 116321 |
| ensemble of 6       | 84.64%   | 83.58%   | 116321 |
| ensemble (log-mean) | 84.71%   | 83.63%   | 116321 |

### Context scaling — r9_opt4_base_seed7

| run                          | accuracy | balanced | n      |
|------------------------------|----------|----------|--------|
| radius 16 (16896 bytes seen) | 86.96%   | 86.14%   | 116321 |

### Context scaling — r9_opt4_base_seed13

| run                          | accuracy | balanced | n      |
|------------------------------|----------|----------|--------|
| radius 16 (16896 bytes seen) | 87.05%   | 86.17%   | 116321 |

### Context scaling — r9_opt4_base_seed29

| run                          | accuracy | balanced | n      |
|------------------------------|----------|----------|--------|
| radius 16 (16896 bytes seen) | 87.13%   | 86.32%   | 116321 |

### Context scaling — r9_opt4_wide_seed7

| run                          | accuracy | balanced | n      |
|------------------------------|----------|----------|--------|
| radius 16 (16896 bytes seen) | 86.82%   | 85.82%   | 116321 |

### Context scaling — r9_opt4_wide_seed13

| run                          | accuracy | balanced | n      |
|------------------------------|----------|----------|--------|
| radius 16 (16896 bytes seen) | 86.43%   | 85.42%   | 116321 |

### Context scaling — r9_opt4_wide_seed29

| run                          | accuracy | balanced | n      |
|------------------------------|----------|----------|--------|
| radius 16 (16896 bytes seen) | 86.40%   | 85.51%   | 116321 |

### Context scaling — ensemble

| run                          | accuracy | balanced | n      |
|------------------------------|----------|----------|--------|
| radius 16 (16896 bytes seen) | 87.62%   | 86.75%   | 116321 |

### Binary-level soft vote

| run                 | accuracy | balanced | n   |
|---------------------|----------|----------|-----|
| r9_opt4_base_seed7  | 92.82%   | 92.82%   | 376 |
| r9_opt4_base_seed13 | 92.82%   | 92.82%   | 376 |
| r9_opt4_base_seed29 | 94.15%   | 94.15%   | 376 |
| r9_opt4_wide_seed7  | 94.15%   | 94.15%   | 376 |
| r9_opt4_wide_seed13 | 93.35%   | 93.35%   | 376 |
| r9_opt4_wide_seed29 | 93.88%   | 93.88%   | 376 |
| ensemble            | 94.41%   | 94.41%   | 376 |
