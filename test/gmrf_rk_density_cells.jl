include("gmrf_rk_numerical.jl")

@testset "Joint GMRF fields in ragged cells preserve priors with empty observations" begin
    times = [[0.0, 0.3], [1.7]]
    geometries = [BRM.brm_gmrf_geometry(GMRF_XY, GMRF_TRIANGLES, t, 2) for t in times]
    data = (; geometries, index1=[[1, 11], Int[]], index2=[[2, 12], Int[]],
        index3=[[5, 15], Int[]], weight1=[[0.2, 0.3], Float64[]],
        weight2=[[0.3, 0.5], Float64[]], weight3=[[0.5, 0.2], Float64[]],
        y=[[0.1, -0.2], Float64[]])
    ast = quote
        spatial ~ LogNormal(0.0, 1.0)
        temporal ~ LogNormal(0.0, 1.0)
        L ~ LKJCholesky(2, 2.0)
        @plate for i in eachindex(geometries)
            field[i] ~ BayesianRegressionModels._BRMGMRFLaw(spatial, temporal, L, geometries[i])
            mu[i] = weight1[i] .* field[i][index1[i]] +
                weight2[i] .* field[i][index2[i]] + weight3[i] .* field[i][index3[i]]
            y[i] .~ Normal.(mu[i], 0.2)
        end
    end
    bound = bind_data(lower_rkppl(ast, data; mod=@__MODULE__, conditioned=(:y,)), data)
    built = build_kernel(bound)
    point = (; spatial=0.7, temporal=1.4, L=[1.0 0.0; 0.45 sqrt(1-0.45^2)],
        field=[0.03 .* sin.(1:g.ncoordinates) for g in geometries])
    u = unconstrain(built.layout, point)
    @test length(u) == 33
    function expected(u)
        nt = constrain(built.layout, u)
        lp = logpdf(LogNormal(0, 1), nt.spatial) + log(nt.spatial) +
            logpdf(LogNormal(0, 1), nt.temporal) + log(nt.temporal) +
            2log1p(-nt.L[2, 1]^2) - BRM.logbeta(0.5, 2.0)
        for i in eachindex(geometries)
            lp += gmrf_rk_oracle(nt.field[i], nt.spatial, nt.temporal, nt.L, times[i])
            mu = data.weight1[i] .* nt.field[i][data.index1[i]] +
                data.weight2[i] .* nt.field[i][data.index2[i]] +
                data.weight3[i] .* nt.field[i][data.index3[i]]
            lp += sum(logpdf.(Normal.(mu, 0.2), data.y[i]); init=0.0)
        end
        lp
    end
    q = prepare_query(built, bound, :sampler)
    @test Base.invokelatest(q, u) ≈ expected(u) atol=2e-9
    selected = select_factors(density_factors(built, bound); kind=:prior, site=:field)
    @test length(selected) == 2
    @test factor_coordinates(selected) == collect(4:33)
    source = string(Base.remove_linenums!(kernel_expr(bound, built.layout)))
    @test !occursin("sampling_logdensity", source)
    replay = Core.eval(@__MODULE__, Expr(:macrocall,
        GlobalRef(ReactiveKernels, Symbol("@kernel")), LineNumberNode(0), Meta.parse(source)))
    replay_q = prepare_query((; spec=replay, layout=built.layout), bound, :sampler)
    @test Base.invokelatest(replay_q, u) ≈ expected(u) atol=2e-9
    cell = prepare_cell_query(built, bound, :y; locality=:error)
    footprint = ReactiveKernels.partition_footprint_kernel(cell.kernel)
    @test Base.invokelatest(cell, u, 2) == 0.0
    @test isempty(Base.invokelatest(footprint, u, 2).unconstrained)
    @test Base.invokelatest(footprint, u, 1).unconstrained == [4, 5, 8, 14, 15, 18]
    sampler = prepare_sampler(built, bound, u; backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
    value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ expected(u) atol=2e-9
    for j in eachindex(u)
        plus, minus = copy(u), copy(u)
        h = 2e-6 * max(1, abs(u[j]))
        plus[j] += h
        minus[j] -= h
        @test gradient[j] ≈ (expected(plus)-expected(minus))/(2h) atol=2e-5 rtol=2e-6
    end
end
