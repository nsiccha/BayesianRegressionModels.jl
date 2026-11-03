include("gmrf_rk_numerical.jl")

# Check the public lexical submodel route needed by the formula emitter.
# Its bare joint declaration obtains its shape from the fitted geometry.
@rkppl gmrf_cell_reader(spatial, temporal, L, geometry, index1, index2, index3,
        weight1, weight2, weight3) = begin
    z ~ BayesianRegressionModels._BRMGMRFLaw(spatial, temporal, L, geometry)
    return weight1 .* z[index1] + weight2 .* z[index2] + weight3 .* z[index3]
end

@testset "GMRF submodel cells keep joint fields and empty-cell priors" begin
    times = [[0.0, 0.3], [1.7]]
    geometries = [BRM.brm_gmrf_geometry(GMRF_XY, GMRF_TRIANGLES, t, 2) for t in times]
    L = [1.0 0.0; 0.45 sqrt(1-0.45^2)]
    data = (; geometries, spatial=0.7, temporal=1.4, L,
        index1=[[1, 11], Int[]], index2=[[2, 12], Int[]],
        index3=[[5, 15], Int[]], weight1=[[0.2, 0.3], Float64[]],
        weight2=[[0.3, 0.5], Float64[]], weight3=[[0.5, 0.2], Float64[]],
        y=[[0.1, -0.2], Float64[]])
    ast = quote
        @plate for i in eachindex(geometries)
            field[i] ~ gmrf_cell_reader(spatial, temporal, L, geometries[i],
                index1[i], index2[i], index3[i], weight1[i], weight2[i], weight3[i])
            y[i] .~ Normal.(field[i], 0.2)
        end
    end
    bound = bind_data(lower_rkppl(ast, data; mod=@__MODULE__, conditioned=(:y,)), data)
    built = build_kernel(bound)
    @test built.layout.total == 30
    u = 0.03 .* sin.(1:30)
    function expected(u)
        value = 0.0
        offset = 0
        for i in eachindex(geometries)
            n = geometries[i].ncoordinates
            z = @view u[offset+1:offset+n]
            value += gmrf_rk_oracle(z, data.spatial, data.temporal, L, times[i])
            mu = data.weight1[i] .* z[data.index1[i]] +
                data.weight2[i] .* z[data.index2[i]] +
                data.weight3[i] .* z[data.index3[i]]
            value += sum(logpdf.(Normal.(mu, 0.2), data.y[i]); init=0.0)
            offset += n
        end
        value
    end
    query = prepare_query(built, bound, :sampler)
    @test Base.invokelatest(query, u) ≈ expected(u) atol=2e-9
    priors = select_factors(density_factors(built, bound); kind=:prior)
    @test length(priors) == 2
    @test factor_coordinates(priors) == collect(1:30)
    cell = prepare_cell_query(built, bound, :y; locality=:error)
    footprint = ReactiveKernels.partition_footprint_kernel(cell.kernel)
    @test Base.invokelatest(cell, u, 2) == 0.0
    @test isempty(Base.invokelatest(footprint, u, 2).unconstrained)
    @test Base.invokelatest(footprint, u, 1).unconstrained == [1, 2, 5, 11, 12, 15]
    sampler = prepare_sampler(built, bound, u; backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
    value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ expected(u) atol=2e-9
    for j in eachindex(u)
        plus, minus = copy(u), copy(u)
        h = 2e-6
        plus[j] += h
        minus[j] -= h
        @test gradient[j] ≈ (expected(plus)-expected(minus))/(2h) atol=2e-5 rtol=2e-6
    end
end
