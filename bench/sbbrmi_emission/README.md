# SBBRMI emission / runtime performance harness

Evidence base for the SBBRMI emission inventory (snag `sbbrmi-emission-e84c1734`).
Every script is standalone and runs under the `test/` environment, matching
`test/README.md`:

    julia --startup-file=no --project=test bench/sbbrmi_emission/<script>.jl

Requires BridgeStan >= 2.9 (BridgeStan.jl self-resolves `$BRIDGESTAN`).

| script | what it answers |
|---|---|
| `emission_survey.jl` | Emits Stan for 27 representative model classes into `$SURVEY_OUT`. |
| `analyze_emitted.py` | Static anatomy of that corpus: per-block line counts, UDF reachability (density / GQ-only / dead), cross-cutting emission markers. |
| `strip.py` | Proper reachability-based `generated quantities` stripper (used by `compile_cost.py`). Seeds only from the non-`functions` body, then takes a transitive closure — a naive "is this name mentioned anywhere" filter keeps mutually-recursive dead clusters alive. |
| `compile_cost.py` | Compile time and `.so` size, full emission vs GQ-stripped. |
| `ladder.jl` | Isolation ladder for the varying-slope model: removes one piece of generic random-effect machinery at a time (V0 emitted → V6 minimal hand-written), with normalized density + gradient equivalence checks, across N ∈ {4, 100, 1000, 10000}. |
| `glm_matrix.jl` | Gaussian / Bernoulli / Poisson GLMs: BRM emission vs hand `X*beta` vs link-fused family vs Stan `*_glm_lpdf` primitive. |
| `bridgestan_floor.jl` | BridgeStan ABI floor (empty model) and the marginal cost of one transformed-parameter vector / one UDF call. |
| `compiler_flags.jl` | `STAN_CPP_OPTIMS` / `STAN_NO_RANGE_CHECKS` sweep. |

## Methodology

Correctness is checked as **normalized** density agreement — variants may differ
by an additive constant (`~` drops constants; a 1x1 LKJ contributes a constant),
so the invariant compared is `lp(q_i) - lp(q_1)` across positions, plus
elementwise gradients, at `propto=false, jacobian=true`. Variants keep BRM's
**parameter declaration order** so the unconstrained coordinate vector is
elementwise comparable.

Timing is `minimum` over repeats of a mean-over-inner-loop, at
`propto=true, jacobian=true` (the sampler's setting), one thread.

**Report ratios at more than one N.** Fixed per-call overhead is constant in N,
so a toy `N=4` benchmark reports it as a large multiple and an `N=1000`
benchmark reports it as noise. The varying-slope gradient ratio moves 3.70x ->
1.07x between those two sizes; the Gaussian GLM ratio moves the other way
(2.69x -> 3.38x) because that cost is per-row. A single-N measurement cannot
tell those two cases apart.
