# test/nlme_offaxis_observations.jl — `brm_nlme_model`'s attribution of
# observations on rows other than the subject column's: a fragment's published
# counts read through an index (`exp(log_f[f_obs])`), its suppressed counts'
# residuals as a plate over its own groups, and a count that reads no random
# effect. Their densities' subjects are measured on the pointwise query by
# moving halves of the subjects' η (snag nlme-model-obser-dd665aca).
#
# Covered: each density's measured region against the region of its rows in
# the data; the per-region likelihoods against the pointwise densities summed
# by those regions; lockstep, per-subject cells and unit problems against the
# full RK density; a residual whose rows lie in two regions refused, naming
# them; and per-unit Laplace and SVI on the panel.
#
# Run: julia --project=test test/nlme_offaxis_observations.jl

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LogDensityProblems, LinearAlgebra, Random
include(joinpath(@__DIR__, "nlme_unit_checks.jl"))
include(joinpath(@__DIR__, "nlme_offaxis_fixture.jl"))

message(f) = try f(); "" catch e; sprint(showerror, e) end
attributed(m, name) = only(last(a) for a in m.attribution if first(a) === name)

@stestset "densities on rows of their own belong to the regions they read" begin
    data = offaxis_panel()
    rk = RKBRMI(offaxis_model(data))
    m = brm_nlme_model(rk; ad_backend=NLME_AD, offaxis_groups(data)...)
    @test length(m.view.levels) == 5
    region(i) = m.view.levels[i]
    owners = offaxis_owners(data)
    @test length(data.f_n) != length(data.y) != length(data.f_resid)
    @test region.(attributed(m, :f_n)) == owners.f_n
    @test region.(attributed(m, :f_resid)) == owners.f_resid
    # Every region owns some of each.
    @test all(i -> i in attributed(m, :f_n) && i in attributed(m, :f_resid), 1:5)
    # No random effect: the first region.
    @test attributed(m, :c_n) == fill(1, length(data.c_n))
    @test region.(attributed(m, :y)) == data.region

    # Each region's likelihood is its densities by the data's regions.
    θ, σ, H, values = check_lockstep(m, Xoshiro(11))
    pointwise = Base.invokelatest(pointwise_query(rk), BRM._brm_nlme_point(m, θ, σ, H))
    expected = [sum(pointwise.y[data.region .== r]) +
        sum(pointwise.f_n[owners.f_n .== r]) + sum(pointwise.f_resid[owners.f_resid .== r]) +
        (i == 1 ? sum(pointwise.c_n) : 0.0) for (i, r) in enumerate(m.view.levels)]
    @test values ≈ expected rtol=1e-12

    # One cell group per measured observation, each density a cell.
    @test Set(c.names for c in m.cell_query) == Set([(:y,), (:f_n,), (:f_resid,), (:c_n,)])
    for i in eachindex(m.view.levels)
        @test brm_nlme_loglikelihood(m, i, θ, σ, H[i]) ≈ values[i] rtol=1e-10
    end
    check_unit_problems(m, Xoshiro(12))
end

@stestset "a residual whose rows lie in two regions is refused by name" begin
    data = offaxis_panel(; span=true)
    @test offaxis_owners(data).f_resid[1] == [1, 2]
    rk = RKBRMI(offaxis_model(data))
    # refused: the density reads two regions' random effects, so the regions
    # are not conditionally independent (snag nlme-model-obser-dd665aca).
    e = message(() -> brm_nlme_model(rk; ad_backend=NLME_AD, offaxis_groups(data)...))
    @test occursin("density 1 of observation `f_resid` reads the random effects of " *
        "2 subjects of `region` (1, 2)", e)
end

@stestset "per-unit Laplace and SVI take the fragment" begin
    data = offaxis_panel()
    rk = RKBRMI(offaxis_model(data))
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, offaxis_groups(data)...)
    x = zeros(LogDensityProblems.dimension(p))
    value, gradient = LogDensityProblems.logdensity_and_gradient(p, x)
    @test isfinite(value) && all(isfinite, gradient)
    fit = brm_svi_fit(rk; rng=Xoshiro(13), ad_backend=NLME_AD, batch_size=2,
        max_iterations=200, ndraws=20, min_windows=1, offaxis_groups(data)...)
    @test fit.coordinates == p.model.view.coordinates
    @test size(fit.draws, 2) == 20 && all(isfinite, fit.draws)
end

println("nlme_offaxis_observations.jl: all testsets passed")
