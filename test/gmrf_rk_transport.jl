include("gmrf_stan.jl")
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, Serialization
import BridgeStan as BS
import DifferentiationInterface as DI
import LogDensityProblems as LDP

function gmrf_transport_model(known, sampled_correlation)
    sampled_correlation || return gmrf_stan_model(; known)
    input = (; (k => v for (k, v) in pairs(data) if k !== :Omega)...)
    @brm input begin
        Omega ~ LKJCholesky(2, 2.0)
        field ~ 0 + gmrf((east, north), time, channel;
            mesh=(vertices=xy, cells=triangles), cor=Omega)
        sd(field, gmrf((east, north), time, channel)) = 1.2
        length_scale(field, gmrf((east, north), time, channel), space) = 0.4
        length_scale(field, gmrf((east, north), time, channel), time) = 1.7
        y ~ Normal(field, 0.2)
    end
end

@testset "Native GMRF backend transport and exported fitted replay" begin
    for (known, sampled_correlation) in ((false, false), (true, false), (true, true))
        @testset "known=$known sampled_correlation=$sampled_correlation" begin
            model = gmrf_transport_model(known, sampled_correlation)
            artifact = BRM.emit_rk_artifact(model;
                case_id="gmrf-transport-$known-$sampled_correlation")
            bound = BRM.rk_translate_artifact(artifact)
            built = build_kernel(bound)
            rk = RKBRMI(model, artifact.plan, built)
            sb = SBBRMI(model; mod=@__MODULE__)
            stan = BRM.stan_instantiate(sb)
            transport = brm_coordinate_transport(rk, sb, BS.param_unc_names(stan.model))
            @test length(transport.pairs) == (known ? 30 : 33) + sampled_correlation
            field = [p for p in transport.pairs if get(p.address, :parameter, nothing) === :field]
            @test length(field) == 30
            @test Set(p.address.channel for p in field) == Set(["a", "b"])
            @test Set(p.address.time for p in field) == Set([0.0, 0.3, 1.7])
            @test Set(p.address.vertex for p in field) == Set(1:5)
            u = 0.1 .* sin.(1:built.layout.total)
            sampler = prepare_sampler(built, bound, u;
                backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
            for point in (zeros(length(u)), u)
                @test brm_check_coordinate_transport(transport, rk, stan.model, point).max_error < 1e-10
                value, gradient = sampler_value_and_gradient!(sampler, similar(u), point)
                sv, sg = BS.log_density_gradient(stan.model, brm_rk_to_stan(transport, point);
                    propto=false, jacobian=true)
                @test value ≈ sv atol=2e-8 rtol=2e-10
                @test gradient ≈ brm_stan_to_rk(transport, sg) atol=3e-7 rtol=3e-7
            end
            @testset "serialized fitted replay" begin
                state = brm_fitted_state(rk)
                io = IOBuffer()
                serialize(io, state)
                seekstart(io)
                state = deserialize(io)
                selected = [4, 2]
                fresh = merge(data, (; (k => getproperty(data, k)[selected]
                    for k in (:east, :north, :time, :channel, :y))...),
                    (; xy=data.xy .+ 10, Omega=[1.0 -0.2; -0.2 1.0]))
                sampled_correlation && (fresh = (; (k => v for (k, v) in pairs(fresh) if k !== :Omega)...))
                replay = reprocess(rk, fresh)
                from_state = RKBRMI(BRM._brm_rebind_brmi(model, fresh), state)
                @test BRM._rk_layout_coordinate_names(replay) == BRM._rk_layout_coordinate_names(rk)
                @test BRM._rk_layout_coordinate_names(from_state) == BRM._rk_layout_coordinate_names(rk)
                @test BRM._rk_replay_statements(BRM._rk_retained_program(replay), replay.plan) == state.program
                @test BRM._rk_replay_statements(BRM._rk_retained_program(from_state), from_state.plan) == state.program
                replay_lp = rk_logdensity_problem(replay; ad_backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
                state_lp = rk_logdensity_problem(from_state; ad_backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
                sb_replay = reprocess(sb, fresh)
                replay_stan = BRM.stan_instantiate(sb_replay)
                for point in (zeros(length(u)), u)
                    sv = BS.log_density(replay_stan.model, brm_rk_to_stan(transport, point);
                        propto=false, jacobian=true)
                    @test LDP.logdensity(replay_lp, point) ≈ sv atol=2e-8 rtol=2e-10
                    @test LDP.logdensity(state_lp, point) ≈ sv atol=2e-8 rtol=2e-10
                end
            end
        end
    end
end
