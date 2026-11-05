include("gmrf_stan_runtime.jl")

@testset "GMRF prior RNG reproduces finite-mesh space-time-channel covariance" begin
    input = (; (k => v for (k, v) in pairs(data) if k !== :y)...)
    model = @brm input begin
        field ~ 0 + gmrf((east, north), time, channel;
            mesh=(vertices=xy, cells=triangles), cor=Omega)
        sd(field, gmrf((east, north), time, channel)) = 1.2
        length_scale(field, gmrf((east, north), time, channel), space) = 0.4
        length_scale(field, gmrf((east, north), time, channel), time) = 1.7
        y ~ Normal(field, 0.2)
    end
    sb = SBBRMI(model; mod=@__MODULE__)
    problem = BRM.stan_instantiate(sb)
    @test BS.param_unc_num(problem.model) == 0
    descriptor = brm_descriptor(sb; highlights=())
    names = BS.param_names(problem.model; include_tp=true, include_gq=true)
    field = brm_term_coordinates(descriptor, :field, names;
        term=:gmrf_field_east_north_time_channel, parameter=:field).coordinates
    rng = BS.StanRNG(problem.model, 20261102)
    draws = reduce(hcat, (BS.param_constrain(problem.model, Float64[];
        include_tp=true, include_gq=true, rng)[field] for _ in 1:6000))
    expected = gmrf_dense_covariance(0.4, 1.7)
    scales = sqrt.(diag(expected))
    @test maximum(abs.(vec(mean(draws; dims=2)) ./ scales)) < 0.06
    empirical = cov(draws; dims=2)
    @test maximum(abs.(sqrt.(diag(empirical)) ./ scales .- 1)) < 0.05
    @test maximum(abs.((empirical - expected) ./ (scales * scales'))) < 0.07
end

