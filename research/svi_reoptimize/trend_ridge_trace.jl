# Window-by-window trace of test/nlme_svi_trend.jl's ridge (8 units of 100 rows, data
# seed 101) with the ELBO-only stop (`stationarity = false`, centering off,
# `coupling = nothing`), for the given unit_reoptimize values and fit seeds (todo
# 12maq21). At k = 1 most seeds stop exploring at the 40-window floor and then end
# far from the posterior (exp(m) 0.2-0.3 against 1.17) while reporting convergence;
# this prints, per window, the phase, every global's mean and sd, each unit
# coordinate's mean over units and the units' mean sd, and the window's ELBO.
#
#     [KS=0,1] [SEEDS=1] julia --project=test research/svi_reoptimize/trend_ridge_trace.jl
const ROOT = get(ENV, "BRM_ROOT", pwd())
include(joinpath(@__DIR__, "..", "..", "test", "fitting_packages.jl"))
include(joinpath(ROOT, "test", "nlme_fixtures.jl"))
using LinearAlgebra, Random, Statistics, Printf
const BRM = BayesianRegressionModels

include(joinpath(ROOT, "test", "nlme_svi_ridge_fixture.jl"))
const OPT = (; subject_group=:unit, batch_size=4, ndraws=200, stationarity=false,
    centering=nothing, coupling=nothing)
const KS = parse.(Int, split(get(ENV, "KS", "0,1"), ","))
const SEEDS = parse.(Int, split(get(ENV, "SEEDS", "1"), ","))

rk = RKBRMI(svi_ridge_model(svi_ridge_data(Xoshiro(101), 8)))
for k in KS, seed in SEEDS
    rows = String[]
    save = state -> begin
        run = state.run
        q = run.q
        M = reduce(hcat, q.M)
        usd = mean(mean(sqrt.(BRM._brm_svi_unit_variances(U))) for U in q.U)
        push!(rows, @sprintf("  %6d %-10s μ %s | sd %s | unit M means %s | unit sd %.4f | elbo %9.2f",
            run.iteration, first(split(BRM._brm_svi_phase_text(run, state.controls), ",")),
            join((@sprintf("%7.3f", v) for v in q.μ), " "),
            join((@sprintf("%.4f", v) for v in exp.(q.ρ)), " "),
            join((@sprintf("%7.3f", v) for v in vec(mean(M; dims=2))), " "), usd,
            isempty(run.windows) ? NaN : last(run.windows).center))
    end
    fit = brm_svi_fit(save, rk; rng=Xoshiro(seed), OPT..., unit_reoptimize=k)
    d = fit.diagnostics
    r = rk_restore_draws(rk, fit.draws)
    println("k=$k seed=$seed globals=$(fit.coordinates[d.globals]) unit coordinates=" *
        "$(fit.coordinates[first(d.units)]) stages=$(d.stages) converged=$(d.converged) " *
        "exp(m) median=$(round(median(exp.(r.gain_mean)); digits=3))")
    for (n, row) in enumerate(rows)
        (n <= 50 || n % 5 == 0 || n == length(rows)) && println(row)
    end
    flush(stdout)
end
