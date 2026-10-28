# test/grouped_hsgp_transport_readers.jl — draws of an RKBRMI fit read into the
# multigroup readers through the coordinate transport (snag
# `read-brm-svi-fit-bb7a9be0`). A panel of sites observed daily: per-site
# level, cycle and trend, a log-scale residual SD per site, a per-site daily
# field `hsgp(day; by = site, cor = multigroup((site_x, site_y); k))` whose
# levels correlate by their known coordinates, and AR(1) residuals in a
# `@plate for p in groups(site; order = day)` cell. `brm_coordinate_transport`
# pairs the grouped HSGP's coordinates (its length scale, amplitude, own
# spatial scale and inducing-grid weights), so a minibatch SVI fit's RK draws
# reach Stan's constrained names, and `brm_multigroup_coherence` /
# `brm_multigroup_field` read the same values RK evaluates.
#
#     julia --project=test test/grouped_hsgp_transport_readers.jl

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions
using LinearAlgebra
using Random
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
import BridgeStan

const BRM = BayesianRegressionModels
const BS = BridgeStan
const CACHE = joinpath(tempdir(), "brm-grouped-hsgp-transport-readers")
const S = 6            # sites
const D = 14           # days per site

function panel(; seed=11)
    rng = Random.MersenneTwister(seed)
    day = repeat(collect(1.0:D), S)
    (; site=repeat(1:S, inner=D), day, c1=cos.(2pi .* day ./ 7), s1=sin.(2pi .* day ./ 7),
       trend=(day .- (D + 1) / 2) ./ D, y=10 .+ randn(rng, S * D),
       site_x=100 .* rand(rng, S), site_y=80 .* rand(rng, S))
end

const panel_model = @brm begin
    mu ~ 1 + c1 + s1 + (1 | lv | site) + (0 + c1 | cc | site) + (0 + s1 | ss | site) +
         hsgp(day; k = 5, by = site, cor = multigroup((site_x, site_y); k = 4))
    tr ~ 0 + trend + (0 + trend | tt | site)
    log(sigma) ~ 1 + (1 | ns | site)
    effect(mu, Intercept) ~ Normal(10.0, 5.0)
    effect(mu, c1) ~ Normal(0.0, 2.0)
    effect(mu, s1) ~ Normal(0.0, 2.0)
    effect(tr, trend) ~ Normal(0.0, 2.0)
    effect(sigma, Intercept) ~ Normal(0.0, 1.0)
    sd(:, lv) ~ Exponential(2.0)
    sd(:, cc) ~ Exponential(1.0)
    sd(:, ss) ~ Exponential(1.0)
    sd(:, tt) ~ Exponential(1.0)
    sd(:, ns) ~ Exponential(0.4)
    phi ~ Uniform(-1.0, 1.0)
    @plate for p in groups(site; order = day)
        m = mu[p] .+ tr[p]
        y[p] ~ AR1Normal(m, sigma[p], phi; time = day[p])
    end
end

compiled(sb, tag) = (mkpath(CACHE);
    BRM.stan_instantiate(sb; path=joinpath(CACHE,
        tag * "_" * string(hash(BRM.stan_code(sb))) * ".stan")))
points(n) = (zeros(n), collect(range(-0.43, 0.51; length=n)), [0.37sin(i) for i in 1:n])

df = panel()
brmi = panel_model(df)
rk = RKBRMI(brmi)
sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
stan = compiled(sb, "panel")
transport = brm_coordinate_transport(rk, sb, BS.param_unc_names(stan.model))

@testset "the transport pairs every coordinate, the grouped HSGP's included" begin
    rk_names = BRM._rk_layout_coordinate_names(rk)
    @test length(transport) == length(rk_names)
    hsgp_pairs = [p for p in transport.pairs if p.address.kind === :hsgp]
    @test count(p -> p.address.parameter === :basis_weights, hsgp_pairs) ==
        4^2 * size(sb.data[:PHI_hsgp_day_by_site], 2)
    @test sort([p.address.parameter for p in hsgp_pairs if p.address.parameter !== :basis_weights]) ==
        [:length_scale, :sd, :spatial_scale]
    problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for u in points(length(transport))
        stan_u = brm_rk_point_to_stan(transport, u)
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        g = similar(stan_u)
        stan_value, _ = BS.log_density_gradient!(stan.model, stan_u, g;
                                                propto=false, jacobian=true)
        @test value ≈ brm_stan_logdensity_to_rk(transport, u, stan_value) atol=1e-10 rtol=1e-12
        @test gradient ≈ brm_stan_gradient_to_rk(transport, u, g) atol=1e-10 rtol=1e-10
        @test brm_check_coordinate_transport(transport, rk, stan.model, u).max_error <= 1e-10
    end
end

@testset "minibatch SVI draws read into the multigroup readers" begin
    fit = brm_svi_fit(rk; batch_size=3, rng=Random.Xoshiro(1), ndraws=6,
                      max_iterations=60)
    U = fit.draws
    @test size(U) == (length(transport), 6)
    names = BS.param_names(stan.model; include_tp=true, include_gq=true)
    C = permutedims(reduce(hcat, [BS.param_constrain(stan.model,
        brm_rk_point_to_stan(transport, U[:, k]); include_tp=true, include_gq=true,
        rng=BS.StanRNG(stan.model, k)) for k in axes(U, 2)]))
    d = brm_descriptor(sb)
    coherence = brm_multigroup_coherence(d, C, names; distances=[10.0, 50.0])
    # RK's own length scale and spatial scale at each draw.
    rk_value(u, name) = BRM._brm_rk_declaration_value(BRM._rk_constrained_values(rk, u), name)
    rho = [rk_value(U[:, k], Symbol("hsgp_day.rho")) for k in axes(U, 2)]
    lambda = [rk_value(U[:, k], Symbol("hsgp_day.lambda")) for k in axes(U, 2)]
    omega2 = vec(sb.data[:omega2_hsgp_day_by_site])
    @test coherence.range ≈ lambda ./ (rho .* sqrt.(omega2'))
    # The field at the sites' own coordinates and the fitted days is the term's
    # value on RK's rows.
    field = brm_multigroup_field(d, C, names; times=collect(1.0:D),
                                 coordinates=(df.site_x, df.site_y))
    @test size(field) == (6, D, S)
    term = rk_draw_values(rk, U, (:hsgp_day,)).hsgp_day
    for k in axes(U, 2)
        @test vec(field[k, :, :]) ≈ term[k] atol=1e-10
    end
end

println("grouped_hsgp_transport_readers.jl: all testsets passed")
