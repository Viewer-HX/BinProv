# Verified best-result tables

Recomputed locally on 2026-09-05 from `BinProv-probs-20260903.tar`
(SHA-256 `aeb1de809b6e1ad36145633b14073a483b5c9d078047758c2293b470125e4434`),
the committed matching `result.json` metadata, and the packed canonical corpus.
The runner verified the 47 test and 28 pinned validation programs before every
calculation. These are offline recalculations; no model was retrained.

| recipe | verified result |
|---|---:|
| O2/O3, 7-wide ensemble, 512-byte target stride | 75.10% sequence; 91.49% binary |
| O2/O3, same ensemble, radius 16 | 81.10% sequence |
| O2/O3, pretrained-wide three-seed mean | 71.98% (SD 0.68 pp) |
| opt4, 6-run ensemble, radius 16 | 87.62% sequence; 94.41% binary |
| opt4, 3-wide ensemble | 84.42% sequence; 95.21% binary |

The exact 19-member ensemble behind the separately reported O2/O3 binary score
of 93.09% is still not identified by the archive, so it is not represented here
as a reproducible recipe.
