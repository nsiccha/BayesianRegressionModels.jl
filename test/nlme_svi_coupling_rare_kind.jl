# Public regression of snag default-linearco-49897cce: on the 32-unit event/Kalman panel
# with one event kind that a single unit reads (nlme_svi_rare_kind_fixture.jl), the
# default LinearCoupling's noise on that unit's funnel rows (the kind's log spreads and
# its non-centred steps) widened them to 1.4-2.2 times their conditional sd during
# settling, and the over-dispersed unit pulled the population AR(1) to 0.95 (truth 0.6;
# `coupling = nothing` 0.674, data seed 1, SVI seed 1, 100,000 iterations). With each
# coupled row's step divided by its updates' mean square where that exceeds one, and
# each step's move of the conditional mean bounded by a quarter sd, the coupled fit
# stays where the independent one is.
# RUN: julia --project=test test/nlme_svi_coupling_rare_kind.jl
using Test, Treebars
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
import Random, Statistics

include("nlme_svi_rare_kind_fixture.jl")

@testset "a single-reader event kind leaves the coupled AR(1) in place" begin
    data = rare_data(; S=32, seed=1)
    rk = RKBRMI(TOY_RARE(data))
    fit = Treebars.with_progress(:state; description="rare-kind panel: default coupling") do root
        path = get(ENV, "KB_PROGRESS_FILE", tempname())
        Treebars.snapshot_progress(root, path) do
            brm_svi_fit(rk; subject_group=:unit, batch_size=4, rng=Random.Xoshiro(1),
                ndraws=1000, max_iterations=100_000, progress=root)
        end
    end
    draws = rk_restore_draws(rk, fit.draws)
    q(x) = Statistics.quantile(x, [0.05, 0.5, 0.95])
    phi = q(tanh.(collect(Float64, vec(draws.phi_Intercept))))
    ratio = q(exp.(draws.gain_mean[1, :]))
    c = fit.diagnostics.coupling
    rare = findfirst(==(data.rare_unit), fit.diagnostics.unit_levels)
    u = c.units[rare]
    # coupling sd / conditional sd of each coupled row of the rare unit
    widening = [sqrt(max(u.marginal_sd[r]^2 / u.conditional_sd[r]^2 - 1, 0.0)) for r in u.rows]
    println("RARE_KIND phi=", round.(phi; digits=3), " ratio=", round.(ratio; digits=3),
        " rare_rows_coupling_over_conditional=", round.(widening; digits=3),
        " converged=", fit.diagnostics.converged, " iterations=", fit.diagnostics.iterations)
    @test 0.5 < phi[2] < 0.8          # truth 0.6; coupling = nothing gives 0.674
    @test 1.13 < ratio[2] < 1.23      # truth 1.18
    @test maximum(widening) < 1       # was 1.46 (polished) on 0.2.330
    @test any(V -> any(!iszero, V), fit.diagnostics.q.coupling.V)
    @test fit.diagnostics.finished
end
