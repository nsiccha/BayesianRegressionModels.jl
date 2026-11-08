# How wide minibatch SVI's global factor may get (src/nlme_svi_sd_bound.jl). Adam
# steps a coordinate by about its step size whatever its gradient noise, so where
# the noise dwarfs the prior's pull the factor's scale and rank row random-walk;
# where the gradient grows with the draw, as a log length's does inside a
# random-feature phase, a wider factor meets larger noise, and a `LowRankGlobal`
# fit's factor drifted to a marginal sd of 378 against a prior sd of 0.5 (snag
# lowrank-svi-scal-4c45adff). A global whose prior reads no other coordinate is
# held within `sd_bound` (2 by default) times the sd of its prior's Gaussian
# projection. The bound is a stopgap (decision 1vm7om5): it warns when a fit ends
# at it, and `sd_bound=nothing` turns it off.
#
# RUN: julia --project=test test/nlme_svi_sd_bound.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))

# A global factor of D coordinates under exploration, stepped as `_brm_svi_iteration!`
# steps it: BRM's global gradient, its exploration step and the clip, then `bound`.
# Coordinate 1 is a log length under a N(m0, τ²) prior whose likelihood gradient is
# noise of sd σw, independent of the draw (a random-feature phase of a short length is
# that, and larger). Coordinates 2:D have a N(0, 1) prior and a Gaussian likelihood
# of precision 4 around 0.3, so their posterior is N(0.24, 0.447²). Returns the factor
# and the log length's widest marginal sd over the run.
function svi_noise_walk(; σw=1e6, bound=nothing, r=8, n=20_000, seed=1, m0=-1.386, τ=0.5, D=6)
    rng = Xoshiro(seed)
    q = (; μ=vcat(m0, zeros(D - 1)), ρ=fill(log(0.1), D), Β=zeros(D, r))
    opt = BRM._BRMSVIAdam(2D + D * r, 0.01)
    widest = 0.0
    for _ in 1:n
        ε, η = randn(rng, D), randn(rng, r)
        z = BRM._brm_svi_draw_global(q, ε, η)
        w = [k == 1 ? -(z[1] - m0) / τ^2 + σw * randn(rng) :
             -z[k] - 4 * (z[k] - 0.3) + randn(rng) for k in 1:D]
        x = BRM._brm_svi_global_vector(q)
        BRM._brm_svi_exploration_global_step!(opt, q, x, BRM._brm_svi_global_gradient(q, w, ε, η),
            0.01; clip=10 * sqrt(5.0))
        BRM._brm_svi_set_global!(q, x)
        BRM._brm_svi_bound_sd!(q, bound)
        widest = max(widest, sqrt(exp(2q.ρ[1]) + sum(abs2, q.Β[1, :])))
    end
    (; q, widest)
end
svi_marginal(q) = sqrt.(exp.(2 .* q.ρ) .+ vec(sum(abs2, q.Β; dims=2)))

@stestset "a global whose gradient is mostly noise drifts wide of its prior, and the bound holds it" begin
    τ = 0.5
    # The control: the prior's pull is a millionth of the noise, Adam's step does
    # not see it, and the log length's factor walks past four prior sds (on BRM
    # 0.2.247: 16.9 mean-field, 3.34 with a rank part). Normalized, the walk is the
    # same at a hundredth of that noise.
    for r in (0, 8)
        free = svi_noise_walk(; r, seed=2)
        @test free.widest > 4τ
        @test svi_noise_walk(; r, seed=2, σw=1e4).widest > 4τ
        held = svi_noise_walk(; r, seed=2, bound=vcat(2τ, fill(Inf, 5)))
        @test held.widest <= 2τ * (1 + 1e-12)
        # The well-informed coordinates keep their posterior.
        @test all(abs.(held.q.μ[2:end] .- 0.24) .< 0.15)
        @test all(svi_marginal(held.q)[2:end] .< 0.7)
    end
end

@stestset "the bound shrinks a scale and its rank row together and leaves the rest alone" begin
    q = (; μ=[0.0, 1.0, 2.0], ρ=log.([3.0, 0.2, 1.0]), Β=[4.0 0.0; 0.1 0.1; 0.0 0.0])
    before = deepcopy(q)
    BRM._brm_svi_bound_sd!(q, [1.0, 1.0, Inf])
    # √(3² + 4²) = 5 → 1: both shrink by 5, so the row's ratio to the scale stays.
    @test svi_marginal(q)[1] ≈ 1.0
    @test exp(q.ρ[1]) ≈ 0.6 && q.Β[1, :] ≈ [0.8, 0.0]
    # Inside its bound, or without one, a global is untouched to the last bit.
    @test isequal(q.ρ[2:3], before.ρ[2:3]) && isequal(q.Β[2:3, :], before.Β[2:3, :])
    @test isequal(q.μ, before.μ)
    # A mean-field factor has no rank part.
    p = (; μ=[0.0], ρ=[log(5.0)], Β=zeros(1, 0))
    BRM._brm_svi_bound_sd!(p, [2.0])
    @test exp(p.ρ[1]) ≈ 2.0
    @test BRM._brm_svi_bound_sd!(p, nothing) === p
end

# Sites read a space-time `gmrf` field whose states are globals: each state's prior
# is a conditional on its neighbours under the field's scale and lengths.
function svi_sd_bound_field_data(; stations=5, times=[0.0, 1.0, 2.0], seed=3)
    rng = Xoshiro(seed)
    px, py = rand(rng, stations), rand(rng, stations)
    east, north, time, site = Float64[], Float64[], Float64[], String[]
    for s in 1:stations, t in times
        push!(east, px[s]); push!(north, py[s]); push!(time, t); push!(site, "s$s")
    end
    n = length(east)
    (; east, north, time, site, y=sin.(3 .* east) .+ 0.3 .* randn(rng, n), state=collect(1:n), nb=3)
end
svi_sd_bound_field_model(data) = @brm data begin
    sigma ~ Exponential(1)
    field ~ 0 + gmrf((east, north), time; neighbors=nb)
    sd(field, gmrf((east, north), time)) ~ Exponential(1)
    length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(0.3), 0.7)
    length_scale(field, gmrf((east, north), time), time) ~ LogNormal(log(3.0), 0.7)
    mu ~ 1 + (1 | site)
    @plate for s in eachindex(state)
        y[s] ~ Normal(mu[s] .+ field[state[s]], sigma)
    end
end

@stestset "only a global whose prior reads no other coordinate has a bound" begin
    m = brm_nlme_model(RKBRMI(svi_sd_bound_field_model(svi_sd_bound_field_data())); ad_backend=NLME_AD,
        subject_group=:site, unit_partition=:connected)
    t = BRM._brm_svi_target(m; separator=:none)
    names = string.(m.view.coordinates[t.globals])
    bound = Dict(zip(names, BRM._BRM_SVI_SD_BOUND .* t.sd_prior))
    field = "gmrf_field_east_north_time"
    states = filter(n -> startswith(n, "$field.field."), names)
    @test length(states) == 15
    # A state's conditional reads its neighbours and the field's lengths, and the
    # lengths' prior term reads every state: none of them has a bound.
    @test all(n -> bound[n] == Inf, states)
    @test bound["$field.spatial_length"] == Inf && bound["$field.temporal_length"] == Inf
    # The states are standardized in the field's sd, whose prior is its own, as are
    # the intercept's, the site sd's and the noise's. log σ under Exponential(1) has
    # the projection N(0, 1), so its bound is 2.
    @test bound["sigma"] ≈ 2.0 rtol=0.1
    @test all(n -> isfinite(bound[n]), ["$field.sigma", "mu_Intercept", "b_site.tau.1", "sigma"])
    # The bound is the model's: a second target agrees to the last bit.
    @test isequal(BRM._brm_svi_target(m; separator=:none).sd_prior, t.sd_prior)
    # With the states in the units' factors, the global term reads them through the
    # fields: the bound reads them at the target's point and leaves that reading as
    # it found it. The lengths are the states' scales here too.
    u = BRM._brm_svi_target(m; separator=:none, field_states=:units)
    unames = string.(m.view.coordinates[u.globals])
    ubound = Dict(zip(unames, u.sd_prior))
    @test ubound["$field.spatial_length"] == Inf && ubound["$field.temporal_length"] == Inf
    @test isfinite(ubound["sigma"]) && isfinite(ubound["$field.sigma"])
    @test isequal(u.fields.others, m.unit)
    @test all(f -> f.law[] === nothing, u.fields.fields)
end

@stestset "a fit inside its bounds reports none reached" begin
    rk = RKBRMI(svi_gauss_model(svi_gauss_data(Xoshiro(5), 40)))
    for family in (MeanFieldGlobal(), LowRankGlobal(2))
        fit = brm_svi_fit(rk; ad_backend=NLME_AD, rng=Xoshiro(6), family, batch_size=8,
            stop_after=2000, ndraws=10)
        s = fit.diagnostics.sd_bound
        @test s.factor == 2.0 && s.stopgap
        @test length(s.bound) == length(fit.diagnostics.globals) == length(s.coordinates)
        @test all(isfinite, s.bound)
        @test isempty(s.at_bound)
        @test all(svi_marginal(fit.diagnostics.q) .< s.bound)
    end
end

@stestset "sd_bound sets the factor, nothing turns it off, and a fit ending at it warns" begin
    rk = RKBRMI(svi_gauss_model(svi_gauss_data(Xoshiro(5), 40)))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    # Each fit gets a fresh stream of its own.
    opt = (; family=LowRankGlobal(2), batch_size=8, stop_after=500, ndraws=0)
    off = brm_svi_fit(m; opt..., rng=Xoshiro(6), sd_bound=nothing)
    @test off.diagnostics.sd_bound.factor === nothing
    @test all(==(Inf), off.diagnostics.sd_bound.bound) && isempty(off.diagnostics.sd_bound.at_bound)
    # A bound that never binds leaves the run as it is without one, to the last bit.
    loose = brm_svi_fit(m; opt..., rng=Xoshiro(6), sd_bound=1e6)
    @test all(n -> isequal(getfield(off.diagnostics.q, n), getfield(loose.diagnostics.q, n)), (:μ, :ρ, :Β))
    # One that binds holds every bounded global at it, and the fit says so.
    tight = @test_logs (:warn, r"at the SVI sd bound \(a stopgap") match_mode=:any brm_svi_fit(m; opt...,
        rng=Xoshiro(6), sd_bound=0.01)
    b = tight.diagnostics.sd_bound
    @test b.factor == 0.01 && !isempty(b.at_bound)
    @test all(svi_marginal(tight.diagnostics.q) .<= b.bound .* (1 + 1e-9))
    @test_throws ArgumentError brm_svi_fit(m; opt..., rng=Xoshiro(6), sd_bound=-1.0)
    @test_throws ArgumentError brm_svi_fit(m; opt..., rng=Xoshiro(6), sd_bound=Inf)
    # Controls written before the bound existed hold the default.
    @test BRM._brm_svi_sd_bound_factor((; settle_windows=0)) == 2.0
    @test BRM._brm_svi_sd_bound_factor((; sd_bound=nothing)) === nothing
end
