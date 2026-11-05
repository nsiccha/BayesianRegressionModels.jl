include("gmrf_stan.jl")
using ReactiveKernels, ReactiveKernelsPPL, Enzyme
import BridgeStan as BS
import DifferentiationInterface as DI

@testset "Public GMRF models retain short-range values and gradients" begin
    model = gmrf_stan_model()
    rk = RKBRMI(model)
    sb = SBBRMI(model; mod=@__MODULE__)
    stan = BRM.stan_instantiate(sb)
    transport = brm_coordinate_transport(rk, sb, BS.param_unc_names(stan.model))
    names = coordinate_names(rk.model.layout)
    coordinates(role) = [only(findall(==(p.rk), names)) for p in transport.pairs
        if get(p.address, :parameter, nothing) === role]
    spatial = only(coordinates(:spatial_length))
    field = coordinates(:field)
    @test length(field) == 30
    u = zeros(length(names))
    sampler = prepare_sampler(rk.model, rk_bound_plan(rk), u;
        backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
    query = prepare_query(rk.model, rk_bound_plan(rk), :sampler)
    for logell in (-2.0, -40.0, -180.0, -365.0)
        for amplitude in (0.0, exp(logell))
            u[spatial] = logell
            u[field] .= amplitude .* sin.(1:30)
            before = copy(u)
            value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
            sv, sg = BS.log_density_gradient(stan.model, brm_rk_to_stan(transport, u);
                propto=false, jacobian=true)
            @test isfinite(value) && isfinite(sv)
            @test all(isfinite, gradient) && all(isfinite, sg)
            @test value ≈ sv atol=2e-8 rtol=2e-10
            @test gradient ≈ brm_stan_to_rk(transport, sg) atol=3e-7 rtol=3e-7
            @test u == before
            if logell <= -40 && iszero(amplitude)
                @test gradient[spatial] ≈ -30 - (logell - 0.5) / 0.8^2 rtol=2e-12
            end
        end
    end
    # A genuine Float64 log-density overflow is -Inf, never an arbitrary
    # NaN caught and relabelled by BRM or by the sampler.
    u[spatial] = -365.0
    u[field] .= 0.1 .* sin.(1:30)
    @test Base.invokelatest(query, u) == -Inf
    @test BS.log_density(stan.model, brm_rk_to_stan(transport, u);
        propto=false, jacobian=true) == -Inf
end
