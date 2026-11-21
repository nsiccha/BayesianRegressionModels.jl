# test/nlme_svi_trend.jl's ridge (8 units of 100 rows, data seed 101) at
# unit_reoptimize = 1 with the ELBO-only stop (`stationarity = false`, centering off),
# over fit seeds, with the default LinearCoupling's noise-scaled step (BRM 0.2.337),
# with the step before it (one-sd bound, unscaled: `_brm_svi_coupling_scaled`
# overridden to false), and with `coupling = nothing` (todo 12maq21). Each fit prints
# its stop, its gain median and spread, the last window's ELBO trace and ‖V‖.
# Before a copy's step had to raise its unit's objective, 3 (noise-scaled), 3
# (unscaled) and 5 (no coupling) of seeds 1-6 ended with exp(m) more than 0.1 off
# (0.006-1.00, last-window ELBO -1,642 to 560), all reported converged; after it,
# all 18 fits read exp(m) 1.163-1.170 (the 400-window fit's 1.166).
#
#     [SEEDS=1,2,3,4] julia --project=test research/svi_reoptimize/trend_ridge_coupling.jl
const ROOT = get(ENV, "BRM_ROOT", pwd())
include(joinpath(@__DIR__, "..", "..", "test", "fitting_packages.jl"))
include(joinpath(ROOT, "test", "nlme_fixtures.jl"))
using LinearAlgebra, Random, Statistics, Printf
const BRM = BayesianRegressionModels

include(joinpath(ROOT, "test", "nlme_svi_ridge_fixture.jl"))
const OPT = (; subject_group=:unit, batch_size=4, ndraws=200, stationarity=false,
    centering=nothing, unit_reoptimize=1)
const SEEDS = parse.(Int, split(get(ENV, "SEEDS", "1,2,3,4"), ","))

normV(d) = d.coupling === nothing ? 0.0 :
    sqrt(sum(sum(abs2, u.V) for u in d.coupling.units; init=0.0))

rk = RKBRMI(svi_ridge_model(svi_ridge_data(Xoshiro(101), 8)))
gain(f) = median(exp.(rk_restore_draws(rk, f.draws).gain_mean))
spread(f) = mean(rk_restore_draws(rk, f.draws).gain_spread)

for variant in (:scaled, :unscaled, :nothing)
    BRM.eval(:(_brm_svi_coupling_scaled(coupling) = $(variant !== :unscaled) &&
        hasproperty(coupling, :noise)))
    for seed in SEEDS
        kw = variant === :nothing ? (; coupling=nothing) : (;)
        # The override is a newer method than this loop's world.
        f = Base.invokelatest(brm_svi_fit, rk; rng=Xoshiro(seed), OPT..., kw...)
        d = f.diagnostics
        W = 100
        e = d.elbo
        @printf("%-9s seed %d | exploring %5d iterations %5d converged %-5s | exp(m) %.3f spread %.4f | last-window ELBO trace %9.2f | ‖V‖ %.3f\n",
            variant, seed, d.stages.exploring, d.iterations, d.converged, gain(f), spread(f),
            mean(e[end-W+1:end]), normV(d))
        flush(stdout)
    end
end
