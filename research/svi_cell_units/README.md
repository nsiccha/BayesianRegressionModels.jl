# Cell units in connected SVI: measurements

What `cell_units=:components` does against `:first_unit` on the synthetic model in
`toy.jl`: sites with one vector-valued cell each, and sensors whose gain and noise
effects are plain vectors read by elementwise rows that read no subject effect.
The logs beside this file are the two runs below, unedited.

```
julia --project=test research/svi_cell_units/measure.jl 12 24 4 2000 20000 100 1 3
julia --project=test research/svi_cell_units/measure.jl 32 64 4 2000 20000 100 1 3
```

Batch 4, `unit_sampling=:work`, `separator=:none`, one thread, 3 seeds, Julia 1.10.12,
RK `7dced1e5`, on a loaded 32-CPU host. Times vary by about a factor of two between
seeds, so read them as ranges. The ELBO is one estimator for both layouts:
`E_q[log p] + H(q)`, with the full density at 128 draws from q.

| `:first_unit` → `:components` | 12 sites, 24 sensors (dense first unit) | 32 sites, 64 sensors (low-rank first unit) |
|---|---|---|
| units | 12 → 36 | 32 → 96 |
| largest unit's dimension | 50 → 2 | 130 → 2 |
| largest unit's share of work | 0.972 → 0.115 | 0.971 → 0.039 |
| largest unit's draw probability | 0.528 → 0.071 | 0.501 → 0.024 |
| optimizer seconds, 20,000 updates | 73 → 10 | 248–308 → 15–16 |
| ELBO after 2,000 updates, mean | −2038.1 → −2035.7 | −5857.9 → −5864.4 |
| ELBO after 20,000 updates, mean of the last 5 of 10 windows | −2038.7 → −2033.3 | −5865.8 → −5853.8 |
| gradient variance per update at a fitted point | 1.05e5 → 2.67e5 | 6.94e5 → 1.94e6 |
| expected gradient time per update at a fitted point | 2.31 → 0.29 ms | 11.6 → 0.34 ms |
| variance × time at a fitted point | 243 → 77 | 8055 → 661 |

Variance is exact for the sampling law, `(Σ‖gᵢ‖²/pᵢ − ‖Σgᵢ‖²)/B`, at one point for
both layouts, and 4,000 sampled batches agree with it. Unit terms sum to the full
density and gradient to within 3e-11.

**What this shows.** Subsampling these rows adds gradient variance per update, as
expected, and costs far less per update, so the variance per unit of time falls.
At equal updates the larger toy's ELBO is behind at first and ahead by 20,000.

**What it does not show.**
- The sensors here are nearly independent given the globals, so this toy cannot
  show the cost of dropping a real correlation between them, which a larger
  shared factor could have kept. A model with such a correlation can lose ELBO.
- These rows cost far less than rows that compute per-row quantities, so the time
  ratios are direction only and not a forecast for any other model.
- `work` is not a time predictor here: its correlation with measured unit time is
  −0.54 and −0.73 over all units, because a site row is one BLAS product.

## The row weight

`_BRM_SVI_ROW_WORK = 16` is chosen on the sampler's variance, not on time, because
data-rich units have larger gradients whatever a row costs. At the same gradients:

| row weight | variance × time at start, small / large | at a fitted point, small / large |
|---|---|---|
| 0 | 1176 / 27989 | 117 / 1062 |
| 1 | 787 / 18171 | 91 / 825 |
| 4 | 448 / 10269 | 69 / 640 |
| 16 | 242 / 5775 | 60 / 567 |
| 64 | 172 / 4285 | 61 / 577 |
| 256 | 154 / 3887 | 63 / 592 |

Counting rows at all helps by about 2× at a fitted point and 5× at the start. Above
4 the fitted-point figures barely change, so this is a plateau and 16 is not a
precisely tuned value. Larger weights favour the start and lose slightly later.
