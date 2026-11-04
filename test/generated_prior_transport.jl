using Test, BayesianRegressionModels, StanBlocks, BridgeStan

# Keep the declared joint prior authoritative: draw in the response-omitted
# program, then invert the observed program's constrained parameter frame.
# In particular, its empty sampler point is not the generated prior draw.
const BS = BridgeStan
const BRM = BayesianRegressionModels

data = (; east=[0.1, 0.9, 0.7, 0.3], north=[0.2, 0.4, 0.8, 0.5],
    time=[0.0, 0.3, 0.3, 1.7], channel=["b", "a", "b", "a"], y=zeros(4),
    xy=[0.0 0.0; 1 0; 1 1; 0 1; 0.5 0.5],
    triangles=[1 2 5; 2 3 5; 3 4 5; 4 1 5],
    x=[-1.0, 0.2, 0.5, 1.0], subject=["b", "a", "b", "a"],
    feature=[1, 2, 3, 1], category=[1, 3, 2, 1])

builder = @brm begin
    Omega ~ LKJCholesky(2, 2.0)
    sigma ~ Exponential(1.0)
    phi ~ Beta(2.0, 3.0)
    weights ~ Dirichlet([2.0, 3.0, 4.0])
    rff[1:3] .~ Normal.(0.0, 1.0)
    field ~ 0 + gmrf((east, north), time, channel;
        mesh=(vertices=xy, cells=triangles), cor=Omega)
    sd(field, gmrf((east, north), time, channel)) ~ LogNormal(0.0, 0.2)
    length_scale(field, gmrf((east, north), time, channel), space) ~ LogNormal(-0.5, 0.2)
    length_scale(field, gmrf((east, north), time, channel), time) ~ LogNormal(0.5, 0.2)
    location ~ 1 + x + (1 + x | p | subject)
    rff_part = rff[feature]
    mixture_part = weights[category]
    mu = location + field + rff_part + mixture_part + phi
    y ~ Normal(mu, sigma)
end

println("Generated-prior transport: build"); flush(stdout)
brmi = builder(data)
target = SBBRMI(brmi; mod=@__MODULE__)
prior_data = (; (k => v for (k, v) in pairs(data) if k !== :y)...)
source = SBBRMI(builder(prior_data); mod=@__MODULE__)
println("Generated-prior transport: compile source"); flush(stdout)
source_problem = BRM.stan_instantiate(source)
println("Generated-prior transport: compile target"); flush(stdout)
target_problem = BRM.stan_instantiate(target)

source_names = String.(BS.param_names(source_problem.model; include_tp=true, include_gq=true))
target_names = String.(BS.param_names(target_problem.model; include_tp=false, include_gq=false))
saved_draws = NamedTuple[]
@testset "Generated joint prior draw to observed Stan point" begin
    @test BS.param_unc_num(source_problem.model) == 0
    @test allunique(source_names)
    @test allunique(target_names)
    missing_names = setdiff(target_names, source_names)
    println("Target dimension=", BS.param_unc_num(target_problem.model),
        " constrained=", length(target_names), " missing=", repr(missing_names))
    flush(stdout)
    @test isempty(missing_names)
    source_positions = Dict(n => i for (i, n) in enumerate(source_names))
    selection = [source_positions[n] for n in target_names]
    rng = BS.StanRNG(source_problem.model, 20261104)
    for draw in 1:4
        generated = BS.param_constrain(source_problem.model, Float64[];
            include_tp=true, include_gq=true, rng)
        theta = generated[selection]
        stan_u = BS.param_unconstrain(target_problem.model, theta)
        @test BS.param_constrain(target_problem.model, stan_u) ≈ theta rtol=1e-10 atol=1e-10
        push!(saved_draws, (; generated, stan_u))
    end
end

using ReactiveKernels, ReactiveKernelsPPL
rk = RKBRMI(brmi)
@testset "Saved prior draws retain their physical values in RK" begin
    transport = brm_coordinate_transport(rk, target, BS.param_unc_names(target_problem.model))
    source_d = brm_descriptor(source; highlights=())
    target_d = brm_descriptor(target; highlights=())
    target_full_names = BS.param_names(target_problem.model; include_tp=true)
    for (; generated, stan_u) in saved_draws
        rk_u = brm_stan_point_to_rk(transport, stan_u)
        @test brm_rk_point_to_stan(transport, rk_u) ≈ stan_u rtol=1e-10 atol=1e-10
        @test brm_check_coordinate_transport(transport, rk, target_problem.model, rk_u).max_error < 1e-9
        values = rk_draw_values(rk, reshape(rk_u, :, 1), (:field, :location, :mu, :sigma, :phi, :rff))
        constrained = BS.param_constrain(target_problem.model, stan_u; include_tp=true)
        for name in (:field, :location, :mu, :sigma, :phi, :rff)
            expected = generated[brm_output_coordinates(source_d, name, source_names)]
            observed = constrained[brm_output_coordinates(target_d, name, target_full_names)]
            native = getproperty(values, name)[1]
            @test vec([native;]) ≈ vec([expected;]) rtol=1e-9 atol=1e-9
            @test vec([observed;]) ≈ vec([expected;]) rtol=1e-9 atol=1e-9
        end
    end
end
