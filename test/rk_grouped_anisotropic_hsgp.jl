using Test, BayesianRegressionModels, StanBlocks, LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
import BridgeStan

const BRM = BayesianRegressionModels
const BS = BridgeStan
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "testset_filter.jl"))
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))

const DATA = (; x=[-0.7, 0.1, 0.5, -0.2, 0.6, 0.9],
    t=[0.0, 0.4, 0.9, 1.1, 1.5, 1.8],
    g=["b", "a", "b", "a", "b", "a"],
    y=[0.3, -0.2, 0.1, 0.5, -0.1, 0.2], C=[1.0 0.35; 0.35 1.0])

const sampled = @brm begin
    Omega ~ LKJCholesky(2, 2.0)
    mu ~ 0 + hsgp(x, t; iso=false, k=(2, 3),
        domain=((-2.0, 2.0), (-1.0, 3.0)), by=g, cor=Omega)
    sd(mu, hsgp(x, t)) ~ Exponential(0.7)
    length_scale(mu, hsgp(x, t)) ~ LogNormal(-0.2, 0.6)
    y ~ Normal(mu, 0.8)
end
const defaults = @brm begin
    mu ~ 0 + hsgp(x, t; iso=false, k=(2, 3), by=g)
    y ~ Normal(mu, 0.8)
end
const floor_prior = @brm begin
    mu ~ 0 + hsgp(x, t; iso=false, k=(2, 3), by=g, cor=C)
    length_scale(mu, hsgp(x, t)) ~ Uniform(length_scale_floor, 12.0)
    y ~ Normal(mu, 0.8)
end
const truncated_prior = @brm begin
    mu ~ 0 + hsgp(x, t; iso=false, k=(2, 3), by=g)
    length_scale(mu, hsgp(x, t)) ~ truncated(LogNormal(-0.2, 0.6);
        lower=length_scale_floor, upper=12.0)
    y ~ Normal(mu, 0.8)
end
const fixed = @brm begin
    mu ~ 0 + hsgp(x, t; iso=false, k=(2, 3), by=g, cor=C)
    length_scale(mu, hsgp(x, t)) = [0.7, 1.2]
    sd(mu, hsgp(x, t)) = 0.6
    y ~ Normal(mu, 0.8)
end
const multigroup_model = @brm begin
    Omega ~ LKJCholesky(2, 2.0)
    tau ~ Exponential(0.8)
    mu ~ 0 + hsgp(x, t; iso=false, k=(2, 3), by=g,
        cor=multigroup(Omega, tau))
    length_scale(mu, hsgp(x, t)) ~ LogNormal(-0.2, 0.6)
    y ~ Normal(mu, 0.8)
end
const one_axis = @brm begin
    mu ~ 0 + hsgp(x; iso=false, k=3, by=g)
    length_scale(mu, hsgp(x)) ~ Uniform(length_scale_floor, 12.0)
    y ~ Normal(mu, 0.8)
end

function check_pair(rk, sb, tag; finite_differences=true)
    stan = BRM.stan_instantiate(sb;
        path=joinpath(tempdir(), "grouped_anisotropic_$tag.stan"))
    transport = brm_coordinate_transport(rk, sb, BS.param_unc_names(stan.model))
    problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    n = length(transport)
    @test n == LogDensityProblems.dimension(problem)
    names = BS.param_names(stan.model; include_tp=true)
    indices = brm_output_coordinates(brm_descriptor(sb), :mu, names)
    for u in (zeros(n), [0.31sin(j) - 0.09cos(2j) for j in 1:n])
        su = brm_rk_point_to_stan(transport, u)
        @test brm_stan_point_to_rk(transport, su) ≈ u atol=1e-12
        value, raw_gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        gradient = copy(raw_gradient)
        sg = similar(su)
        sv, _ = BS.log_density_gradient!(stan.model, su, sg; propto=false, jacobian=true)
        @test value ≈ brm_stan_logdensity_to_rk(transport, u, sv) atol=2e-9 rtol=1e-11
        @test gradient ≈ brm_stan_gradient_to_rk(transport, u, sg) atol=2e-8 rtol=2e-8
        @test brm_check_coordinate_transport(transport, rk, stan.model, u).max_error < 1e-10
        actual = only(rk_draw_values(rk, reshape(u, :, 1), (:mu,)).mu)
        expected = BS.param_constrain(stan.model, su; include_tp=true)[indices]
        @test vec(actual) ≈ expected atol=1e-10 rtol=1e-10
        if finite_differences
            for j in eachindex(u)
                h = 1e-5
                plus, minus = copy(u), copy(u)
                plus[j] += h
                minus[j] -= h
                fd = (LogDensityProblems.logdensity(problem, plus) -
                      LogDensityProblems.logdensity(problem, minus)) / (2h)
                @test gradient[j] ≈ fd atol=3e-6 rtol=3e-6
            end
        end
    end
    transport, stan
end

for (label, builder) in (("sampled-correlation", sampled), ("defaults", defaults),
                         ("floor-prior", floor_prior), ("truncated-prior", truncated_prior),
                         ("known-scales", fixed),
                         ("multigroup", multigroup_model), ("one-axis", one_axis))
    @stestset "grouped anisotropic HSGP: $label" begin
        before = deepcopy(DATA)
        brmi = builder(DATA)
        rk = check_rk_source_roundtrip(RKBRMI(brmi))
        sb = SBBRMI(brmi; mod=@__MODULE__)
        transport, stan = check_pair(rk, sb, label)
        if label == "sampled-correlation"
            pairs = filter(p -> p.address.kind === :hsgp &&
                p.address.parameter === :length_scale, transport.pairs)
            @test length(pairs) == 2
            q = brm_start(rk, (;); term_inits=[
                :(length_scale(mu, hsgp(x, t))) => [0.8, 1.4],
                :(sd(mu, hsgp(x, t))) => 0.5])
            names = BS.param_names(stan.model; include_tp=true)
            values = BS.param_constrain(stan.model, brm_rk_point_to_stan(transport, q);
                include_tp=true)
            coordinates = brm_term_coordinates(brm_descriptor(sb), :mu, names;
                term=:hsgp_x_t_by_g, parameter=:length_scale)
            @test values[coordinates.coordinates] ≈ [0.8, 1.4]
            order = [6, 4, 2, 5, 3, 1]
            design = merge(DATA, (; x=DATA.x[order] .+ 0.12, t=DATA.t[order] .+ 0.2,
                g=DATA.g[order], y=DATA.y[order]))
            check_pair(reprocess(rk, design), reprocess(sb, design), "replay";
                finite_differences=false)
        end
        @test isequal(DATA, before)
    end
end
