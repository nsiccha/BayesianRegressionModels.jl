These links open the saved source text directly. No local execution is needed.
The generated Stan files are the targets used in the completed fits; the
counted variants include the zero-contribution gradient counter. Clean and
counted files target the same posterior. Our total-coefficient target is
implemented manually in Julia, so it has no generated Stan file.

### Test harness and analysis

| Inspect | Source files |
|---|---|
| Manual total target, priors and independent audits | [model.jl](<local path omitted> [audit.jl](<local path omitted> |
| Total NCP pilot, offline selection/refit and fixed CP | [run.jl](<local path omitted> [fixed_cp.jl](<local path omitted> |
| Ordinary brms + WHMC | [brms_baseline.jl](<local path omitted> [ordinary_cp.jl](<local path omitted> [stan_target.jl](<local path omitted> |
| Native Stan and actual gradient counting | [native_stan.R](<local path omitted> [native_tools.R](<local path omitted> [C++ counter](<local path omitted> |
| brms S2Z generation, native sampling and WHMC/audits | [s2z_native.R](<local path omitted> [capture_s2z.R](<local path omitted> [s2z_whmc.jl](<local path omitted> |
| Conditional recovery and the 46-QOI comparison | [recovery.jl](<local path omitted> [compare_qois.jl](<local path omitted> [matrix/cost assembly](<local path omitted> |
| Figures from saved draws | [total-coordinate preparation](<local path omitted> [S2Z-coordinate preparation](<local path omitted> [total-coordinate plot](<local path omitted> [comparison plots](<local path omitted> |

### Exact generated Stan targets and inputs

| Model | Generated Stan | Inputs |
|---|---|---|
| Ordinary NCP | [clean model](<local path omitted> [counted native model](<local path omitted> | [resolved data](<local path omitted> [initialization](<local path omitted> |
| Ordinary CP (WHMC) | [clean model](<local path omitted> | [resolved data](<local path omitted> |
| S2Z CP | [clean model](<local path omitted> [counted native model](<local path omitted> | [resolved data](<local path omitted> [initialization](<local path omitted> |
| S2Z NCP | [clean model](<local path omitted> [counted native model](<local path omitted> | [resolved data](<local path omitted> [initialization](<local path omitted> |
| S2Z auto | [clean model](<local path omitted> [counted native model](<local path omitted> | [resolved data](<local path omitted> [initialization](<local path omitted> [auto weights](<local path omitted> |

### Data and numerical results

- [Original-order pupil data (CSV)](<local path omitted>
- [Complete 46-QOI efficiency table (TSV)](<local path omitted>
- [Per-quantity ESS and MCSE (TSV)](<local path omitted>
- [S2Z scatter selection](<local path omitted> and [coordinate audit](<local path omitted>
- [Source snapshot manifest with byte counts and SHA-256](<local path omitted>
