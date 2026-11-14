# Public 32-unit event/Kalman reproduction of snag brm-svi-fit-defa-afacf161.
# The model and data generator (nlme_svi_events_fixture.jl) are copied unchanged
# from the reporter's toy_events.jl, SHA256 05f376543fbc89a2e0058445374384c94107dbea00016b67066d6f51ae4234d3.
# Both policies get the same data, fit seed, batch size and total iteration cap.
# RUN: julia --project=test test/nlme_svi_coupling_events.jl
using Test, Treebars
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
import Random, Statistics

include("nlme_svi_events_fixture.jl")

function event_coupling_fit(rk, label; options...)
    Treebars.with_progress(:state; description=label) do root
        path = get(ENV, "KB_PROGRESS_FILE", tempname())
        Treebars.snapshot_progress(root, path) do
            brm_svi_fit(rk; subject_group=:unit, batch_size=4,
                rng=Random.Xoshiro(1), ndraws=1000, max_iterations=100_000,
                progress=root, options...)
        end
    end
end

function event_coupling_summary(rk, fit)
    draws = rk_restore_draws(rk, fit.draws)
    ratio = Statistics.quantile(exp.(draws.gain_mean[1, :]), [0.05, 0.5, 0.95])
    phi = Statistics.quantile(tanh.(collect(Float64, vec(draws.phi_Intercept))), [0.05, 0.5, 0.95])
    (; ratio, phi, elbo=Statistics.mean(fit.diagnostics.elbo[end-999:end]))
end

@testset "default coupling retains the event posterior region" begin
    rk = RKBRMI(TOY3(toy3_data(; S=32, seed=1)))
    coupled = event_coupling_fit(rk, "event model: default coupling")
    independent = event_coupling_fit(rk, "event model: independent factors"; coupling=nothing)
    c, p = event_coupling_summary(rk, coupled), event_coupling_summary(rk, independent)
    println("EVENT_COUPLING coupled=", c, " independent=", p,
        " coupled_converged=", coupled.diagnostics.converged,
        " independent_converged=", independent.diagnostics.converged)
    # Truth is gain ratio 1.18 and phi 0.6. Immediate coupling previously sent
    # the gain ratio far below 1 or phi toward 1; the independent fit recovered
    # the intended region with a much narrower interval. This regression does
    # not establish interval calibration for the event model.
    @test 1.13 < c.ratio[2] < 1.23
    @test 0.4 < c.phi[2] < 0.85
    @test abs(c.ratio[2] - p.ratio[2]) < 0.03
    @test abs(c.phi[2] - p.phi[2]) < 0.1
    @test c.ratio[3] - c.ratio[1] > 1.5 * (p.ratio[3] - p.ratio[1])
    @test c.elbo > p.elbo - 50
    @test any(V -> any(!iszero, V), coupled.diagnostics.q.coupling.V)
    @test coupled.diagnostics.finished && independent.diagnostics.finished
    @test !coupled.diagnostics.converged || coupled.diagnostics.stationarity.current.passed
end
