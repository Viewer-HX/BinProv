
### Individual runs

| run                 | accuracy | balanced | n      |
|---------------------|----------|----------|--------|
| r9_opt4_wide_seed7  | 83.77%   | 82.58%   | 116321 |
| r9_opt4_wide_seed13 | 83.61%   | 82.42%   | 116321 |
| r9_opt4_wide_seed29 | 83.85%   | 82.76%   | 116321 |

### Ensemble

| run                 | accuracy | balanced | n      |
|---------------------|----------|----------|--------|
| r9_opt4_wide_seed7  | 83.77%   | 82.58%   | 116321 |
| r9_opt4_wide_seed13 | 83.61%   | 82.42%   | 116321 |
| r9_opt4_wide_seed29 | 83.85%   | 82.76%   | 116321 |
| ensemble of 3       | 84.42%   | 83.32%   | 116321 |
| ensemble (log-mean) | 84.45%   | 83.35%   | 116321 |

### Context scaling — r9_opt4_wide_seed7

| run                       | accuracy | balanced | n      |
|---------------------------|----------|----------|--------|
| radius 0 (512 bytes seen) | 83.77%   | 82.58%   | 116321 |

### Context scaling — r9_opt4_wide_seed13

| run                       | accuracy | balanced | n      |
|---------------------------|----------|----------|--------|
| radius 0 (512 bytes seen) | 83.61%   | 82.42%   | 116321 |

### Context scaling — r9_opt4_wide_seed29

| run                       | accuracy | balanced | n      |
|---------------------------|----------|----------|--------|
| radius 0 (512 bytes seen) | 83.85%   | 82.76%   | 116321 |

### Context scaling — ensemble

| run                       | accuracy | balanced | n      |
|---------------------------|----------|----------|--------|
| radius 0 (512 bytes seen) | 84.42%   | 83.32%   | 116321 |

### Binary-level soft vote

| run                 | accuracy | balanced | n   |
|---------------------|----------|----------|-----|
| r9_opt4_wide_seed7  | 94.15%   | 94.15%   | 376 |
| r9_opt4_wide_seed13 | 93.35%   | 93.35%   | 376 |
| r9_opt4_wide_seed29 | 93.88%   | 93.88%   | 376 |
| ensemble            | 95.21%   | 95.21%   | 376 |
