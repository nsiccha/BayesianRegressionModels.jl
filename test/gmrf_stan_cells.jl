include("gmrf_preparation.jl")
using StanBlocks
include("gmrf_stan_oracle.jl")

@testset "Grouped observation cells retain one joint GMRF field" begin
    input = merge(data, (; unit=[10, 20, 10, 10], units=[10, 20]))
    model = @brm input begin
        field ~ 0 + gmrf((east, north), time, channel;
            mesh=(vertices=xy, cells=triangles), cor=Omega)
        sd(field, gmrf((east, north), time, channel)) = 1.2
        length_scale(field, gmrf((east, north), time, channel), space) = 0.4
        length_scale(field, gmrf((east, north), time, channel), time) = 1.7
        rows = groups(unit => units)
        @plate for k in eachindex(units)
            y[rows[k]] ~ Normal(field[rows[k]], 0.2)
        end
    end
    sb = SBBRMI(model; mod=@__MODULE__)
    problem = BRM.stan_instantiate(sb)
    descriptor = brm_descriptor(sb; highlights=())
    names = BS.param_unc_names(problem.model)
    field = brm_term_coordinates(descriptor, :field, names;
        term=:gmrf_field_east_north_time_channel, parameter=:field).coordinates
    @test length(names) == length(field) == 30
    point = zeros(length(names))
    point[field] = 0.05 .* sin.(1:30)
    oracle = q -> first(gmrf_dense_oracle(sb, q, (; field); known=true))
    value, gradient = BS.log_density_gradient(problem.model, point; propto=false, jacobian=true)
    @test value ≈ oracle(point) atol=2e-8 rtol=2e-10
    for j in eachindex(point)
        h = 1e-5
        plus, minus = copy(point), copy(point)
        plus[j] += h
        minus[j] -= h
        @test gradient[j] ≈ (oracle(plus)-oracle(minus))/(2h) atol=3e-5 rtol=3e-6
    end
    @test brm_description(descriptor).complete
    order = [4, 1, 3, 2]
    fresh = (; (k => getproperty(input, k)[order]
        for k in (:east, :north, :time, :channel, :y, :unit))..., units=input.units)
    replay = reprocess(sb, fresh)
    stem = "gmrf_field_east_north_time_channel_"
    @test replay.data[Symbol(stem, "n_field")] == 30
    for j in 1:3
        key = Symbol(stem, "index", j)
        @test replay.data[key] == sb.data[key][order]
    end
end
