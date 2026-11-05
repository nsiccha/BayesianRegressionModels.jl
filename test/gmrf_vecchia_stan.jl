# The footprint `gmrf` term on SBBRMI: valid Stan, density parity with the native core
# and RK through coordinate transport, replay, descriptions and refusals.
using Test, BayesianRegressionModels, LinearAlgebra, Random, Distributions
using StanBlocks, ReactiveKernels, ReactiveKernelsPPL, Enzyme
import BridgeStan as BS
import DifferentiationInterface as DI
import LogDensityProblems as LDP
const BRM = BayesianRegressionModels

function vecchia_stan_data(; channels=false, seed=21)
    rng = MersenneTwister(seed)
    npositions, times = 7, [0.0, 1.0, 2.0, 80.0]
    x, y = 10 .* rand(rng, npositions), 10 .* rand(rng, npositions)
    east, north = repeat(x; outer=length(times)), repeat(y; outer=length(times))
    time = repeat(times; inner=npositions)
    keep = [i for i in eachindex(time) if !(time[i] == 80.0 && isodd(i))]
    east, north, time = east[keep], north[keep], time[keep]
    channels || return (; east, north, time, y=0.3 .* randn(rng, length(time)))
    channel = repeat(["gauge", "radar"]; outer=length(keep))
    east, north, time = repeat(east; inner=2), repeat(north; inner=2), repeat(time; inner=2)
    (; east, north, time, channel, y=0.3 .* randn(rng, length(time)))
end

function vecchia_scalar_model(data)
    @brm data begin
        field ~ 0 + gmrf((east, north), time; neighbors=3)
        sd(field, gmrf((east, north), time)) ~ Exponential(0.7)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(3.0), 0.4)
        length_scale(field, gmrf((east, north), time), time) ~ LogNormal(log(1.7), 0.4)
        y ~ Normal(field, 0.2)
    end
end

function vecchia_channel_model(data)
    @brm data begin
        Omega ~ LKJCholesky(2, 2.0)
        field ~ 0 + gmrf((east, north), time, channel; neighbors=3, cor=Omega)
        sd(field, gmrf((east, north), time, channel)) = 1.1
        length_scale(field, gmrf((east, north), time, channel), space) = 3.0
        length_scale(field, gmrf((east, north), time, channel), time) = 1.7
        y ~ Normal(field, 0.2)
    end
end

@testset "Valid Stan and the native helper match the core" begin
    data = vecchia_stan_data()
    sb = SBBRMI(vecchia_scalar_model(data); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    check = StanBlocks.stanc_check(code; warn_pedantic=false)
    check.ok || println(check)
    @test check.ok
    @test occursin("brm_gmrf_vecchia", code)
    f = BRM._brm_vecchia_footprint(data.east, data.north, data.time, 2; neighbors=3)
    tables = BRM._brm_gmrf_footprint_data(f)
    L = [1.0 0.0; 0.5 sqrt(1 - 0.25)]
    x = 0.3 .* randn(MersenneTwister(1), f.ncoordinates)
    @test BRM.brm_gmrf_vecchia_logdensity(x, 2.1, 1.3, L, values(tables)...) ≈
        BRM._brm_vecchia_logdensity(x, 2.1, 1.3, L, f) atol=1e-10
end

@testset "SB and RK densities agree through coordinate transport" begin
    for (model, label) in ((vecchia_scalar_model(vecchia_stan_data()), "scalar"),
            (vecchia_channel_model(vecchia_stan_data(; channels=true)), "channels"))
        @testset "$label" begin
            artifact = BRM.emit_rk_artifact(model; case_id="gmrf-vecchia-transport-$label")
            bound = BRM.rk_translate_artifact(artifact)
            built = build_kernel(bound)
            rk = RKBRMI(model, artifact.plan, built)
            sb = SBBRMI(model; mod=@__MODULE__)
            stan = BRM.stan_instantiate(sb)
            transport = brm_coordinate_transport(rk, sb, BS.param_unc_names(stan.model))
            u = 0.1 .* sin.(1:built.layout.total)
            sampler = prepare_sampler(built, bound, u; backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
            for point in (zeros(length(u)), u)
                @test brm_check_coordinate_transport(transport, rk, stan.model, point).max_error < 1e-10
                value, gradient = sampler_value_and_gradient!(sampler, similar(u), point)
                sv, sg = BS.log_density_gradient(stan.model, brm_rk_to_stan(transport, point);
                    propto=false, jacobian=true)
                @test value ≈ sv atol=2e-8 rtol=2e-10
                @test gradient ≈ brm_stan_to_rk(transport, sg) atol=3e-7 rtol=3e-7
            end
        end
    end
end

@testset "Frozen and fresh replay" begin
    data = vecchia_stan_data()
    builder = @brm begin
        field ~ 0 + gmrf((east, north), time; neighbors=3)
        y ~ Normal(field, 0.2)
    end
    sb = SBBRMI(builder(data); mod=@__MODULE__)
    rows = [5, 1, 9]
    subset = (; east=data.east[rows], north=data.north[rows], time=data.time[rows], y=data.y[rows])
    replay = BRM.reprocess(sb, subset)
    keys_of(d, name) = only(k for k in keys(d) if endswith(string(k), name))
    @test replay.data[keys_of(replay.data, "_index")] ==
        sb.data[keys_of(sb.data, "_index")][rows]
    @test replay.data[keys_of(replay.data, "_n_field")] == sb.data[keys_of(sb.data, "_n_field")]
    moved = (; east=[data.east[1] + 0.01], north=[data.north[1]], time=[data.time[1]], y=[0.0])
    @test_throws ArgumentError BRM.reprocess(sb, moved)
    # A fresh refit takes its own rows' read pairs as its footprint.
    fresh = BRM.reprocess(sb, subset; freeze_constants=false)
    @test fresh.data[keys_of(fresh.data, "_n_field")] == 3
    @test sort(fresh.data[keys_of(fresh.data, "_index")]) == 1:3
    @test isfinite(BS.log_density(BRM.stan_instantiate(fresh).model,
        zeros(3 + fresh.data[keys_of(fresh.data, "_n_field")]); propto=false, jacobian=true))
end

@testset "Default priors transpile in a module that does not see BRM's internals" begin
    data = vecchia_stan_data()
    mesh = (; data..., xy=[-1.0 -1.0; 11 -1; 11 11; -1 11], triangles=[1 2 3; 1 3 4])
    footprint_field = @brm data begin
        field ~ 0 + gmrf((east, north), time; neighbors=3)
        y ~ Normal(field, 0.2)
    end
    mesh_field = @brm mesh begin
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        y ~ Normal(field, 0.2)
    end
    for model in (footprint_field, mesh_field)
        code = StanBlocks.stan_code(SBBRMI(model; mod=Main).model)
        @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    end
end

@testset "Descriptions name the representation in both languages" begin
    data = vecchia_stan_data()
    builder = @brm begin
        field ~ 0 + gmrf((east, north), time; neighbors=3)
        y ~ Normal(field, 0.2)
    end
    d = brm_descriptor(builder, data; mod=@__MODULE__, name=:footprint)
    english = string(brm_description(d))
    @test occursin("nearest-neighbour (Vecchia)", english)
    @test !occursin("finite-element", english)
    german = string(brm_description(d; texts=description_texts(:de)))
    @test occursin("Nächste-Nachbarn", german)
end
