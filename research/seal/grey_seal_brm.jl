# A faithful-STRUCTURE grey-seal IPM on the `@brm` FORMULA surface.
#
# The verbatim SlicTranspiler research model lives in `grey_seal_ipm.jl` (a
# `compile_slic_bundle` `@slic` workspace: Leslie matrix, a within-year hunting ODE,
# multinomial-allocation, 8 observation streams). THIS file is the demonstration that
# the seal's REGRESSION / RANDOM-EFFECT / TIME-SERIES / MULTI-STREAM structure is a
# `@brm` formula model — every such piece is a formula-level statement — with the
# mechanistic demographic dynamics in a `@deffun` scan (exactly as CDC's renewal is).
# Illustrative dynamics, the same way the CDC port uses illustrative PMFs.
#
# How each seam maps (cf. the `@slic` seal / `compute_baseline_birth_rate` /
# `run_state_process`):
#   - birth-rate COVARIATE REGRESSION .. `eta_birth ~ 1 + herring_index` (the
#                                        logistic-on-herring birth rate);
#   - per-year RANDOM EFFECT .......... `re_birth ~ 0 + (1 | year)` (the `epsilon_*`
#                                        non-centered per-year process innovations);
#   - mechanistic age-structured SCAN . `state = seal_state(...)`, a `@deffun` returning
#                                        a NamedTuple of carriers, CONSUMED VIA FIELD
#                                        ACCESS (`state.population_total`), one scan;
#   - standard-family OBS ............. `NegativeBinomial2` aerial pup counts +
#                                        `Binomial` pregnancy;
#   - per-row COMPOSITION OBS ......... `Multinomial(comp_sample, state.composition)`,
#                                        a per-YEAR age-composition simplex (the seal's
#                                        hunting/reproductive-signs composition shape).
#
# Verified: transpile + stanc + finite BridgeStan density/gradient; see
# `test/seal_brm.jl`. NOT sampled — the deliverable is the model.
using BayesianRegressionModels
using StanBlocks
using Distributions: Uniform, Binomial, Multinomial   # BRM re-exports Normal/Exponential/NegativeBinomial2

StanBlocks.@deffun begin
    # Age-structured (pup/juvenile/adult) projection over T years, returning a NamedTuple
    # of carriers: population total, a per-year pregnancy rate, and a per-year 3-class
    # age-composition simplex. Faithful in STRUCTURE to `run_state_process` (minus the
    # within-year hunting ODE and multinomial-allocation). Consumes the formula-built
    # `eta_birth` (birth-rate linear predictor) and `re_birth` (per-year RE).
    seal_state(eta_birth::vector[T], re_birth::vector[T],
               phi_pup::real, phi_juv::real, phi_adult::real,
               pop_init::real, log_K::real) = begin
        total::vector[T]
        preg_rate::vector[T]
        composition::matrix[T, 3]
        Npup   = pop_init * 0.3
        Njuv   = pop_init * 0.3
        Nadult = pop_init * 0.4
        for t in 1:T
            pop = Npup + Njuv + Nadult
            # bounded logistic birth rate: herring-covariate LP minus density dependence
            br  = inv_logit(eta_birth[t] - pop / exp(log_K))
            births = Nadult * br + re_birth[t]          # + per-year process innovation
            new_adult = phi_juv * Njuv + phi_adult * Nadult
            new_juv   = phi_pup * Npup
            new_pup   = births
            Npup   = new_pup
            Njuv   = new_juv
            Nadult = new_adult
            tot = Npup + Njuv + Nadult
            total[t]     = tot
            preg_rate[t] = inv_logit(eta_birth[t])
            composition[t, 1] = (Npup + 1.0) / (tot + 3.0)   # additive-smoothed simplex
            composition[t, 2] = (Njuv + 1.0) / (tot + 3.0)
            composition[t, 3] = (Nadult + 1.0) / (tot + 3.0)
        end
        (; total, preg_rate, composition)
    end
end

"""
    grey_seal_brm_fixture(; T = 40)

Dataset for [`grey_seal_brm_model`](@ref): `T` years, a herring-index covariate, aerial
pup counts, pregnancy counts + sample sizes, and a per-year 3-class age-composition
count matrix (`T × 3`) + its per-year totals.
"""
function grey_seal_brm_fixture(; T = 40)
    herring = [0.3 * sin(t / 5) + 0.1 * (t / T) for t in 1:T]
    pup = [max(1, round(Int, 800 + 300 * exp(-((t - 22.0) / 10)^2))) for t in 1:T]
    preg = [round(Int, 60 * (0.3 + 0.2 * sin(t / 5))) for t in 1:T]
    comp = permutedims(reduce(hcat, [[8, 6, 6] for _ in 1:T]))
    (;
        year = collect(1:T),
        herring_index = herring,
        pup_count = pup,
        preg_count = preg, preg_sample = fill(60, T),
        comp_count = comp, comp_sample = [sum(comp[t, :]) for t in 1:T],
        pop_init = 3000.0,
    )
end

"""
    grey_seal_brm_model([df]) -> BRMI

The faithful-structure grey-seal IPM on the `@brm` formula surface (see the block
comment above for the seam-by-seam mapping), returned as a `BRMI`; lower with
`SBBRMI(grey_seal_brm_model())`. Re-bind data with
`grey_seal_brm_model(grey_seal_brm_fixture(; T = 60))`.
"""
function grey_seal_brm_model(df = grey_seal_brm_fixture())
    @brm df begin
        # demographic / survival scalars
        phi_pup   ~ Uniform(0.0, 1.0)
        phi_juv   ~ Uniform(0.0, 1.0)
        phi_adult ~ Uniform(0.0, 1.0)
        log_K     ~ Normal(8.0, 1.0)          # log carrying capacity
        disp      ~ Exponential(1.0)          # pup-count NB dispersion
        # birth-rate covariate regression + per-year random effect
        eta_birth ~ 1 + herring_index
        re_birth  ~ 0 + (1 | year)
        # mechanistic age-structured scan (one call; carriers read via field access)
        state = seal_state(eta_birth, re_birth, phi_pup, phi_juv, phi_adult, pop_init, log_K)
        # observation streams
        pup_count  ~ NegativeBinomial2(state.total, disp)          # aerial pup counts
        preg_count ~ Binomial(preg_sample, state.preg_rate)        # pregnancy counts
        comp_count ~ Multinomial(comp_sample, state.composition)   # per-year age composition
    end
end
