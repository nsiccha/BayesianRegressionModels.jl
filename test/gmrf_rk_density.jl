include("gmrf_rk_numerical.jl")

function gmrf_rk_fixture()
    times = [0.0, 0.3, 1.7]
    mesh = BRM._brm_gmrf_mesh(GMRF_XY, GMRF_TRIANGLES)
    rows = BRM._brm_gmrf_interpolation(mesh, [0.1, 0.9, 0.7], [0.2, 0.4, 0.8])
    indices = BRM._brm_gmrf_state_indices(rows.indices, times, [1, 2, 1], times, [1, 2], 5)
    data = (; vertices=GMRF_XY, cells=GMRF_TRIANGLES, times, n_field=30,
        index1=indices[:, 1], index2=indices[:, 2], index3=indices[:, 3],
        weight1=rows.weights[:, 1], weight2=rows.weights[:, 2], weight3=rows.weights[:, 3],
        y=[0.1, -0.2, 0.4])
    ast = quote
        geometry = BayesianRegressionModels.brm_gmrf_geometry(vertices, cells, times, 2)
        spatial ~ LogNormal(0.0, 1.0)
        temporal ~ LogNormal(0.0, 1.0)
        L ~ LKJCholesky(2, 2.0)
        field[1:n_field] ~ BayesianRegressionModels._BRMGMRFLaw(spatial, temporal, L, geometry)
        mu = weight1 .* field[index1] + weight2 .* field[index2] + weight3 .* field[index3]
        y .~ Normal.(mu, 0.2)
    end
    bound = bind_data(lower_rkppl(ast, data; mod=@__MODULE__, conditioned=(:y,)), data)
    built = build_kernel(bound)
    point = (; spatial=0.7, temporal=1.4, L=[1.0 0.0; 0.45 sqrt(1-0.45^2)],
        field=0.03 .* sin.(1:30))
    u = unconstrain(built.layout, point)
    (; data, bound, built, u)
end

function gmrf_rk_target(fx, u)
    nt = constrain(fx.built.layout, u)
    data = fx.data
    joint = gmrf_rk_oracle(nt.field, nt.spatial, nt.temporal, nt.L, data.times)
    hyper = logpdf(LogNormal(0, 1), nt.spatial) + log(nt.spatial) +
        logpdf(LogNormal(0, 1), nt.temporal) + log(nt.temporal)
    # K=2 LKJ(eta=2) plus the rho=tanh(u) Jacobian.
    hyper += 2log1p(-nt.L[2, 1]^2) - BRM.logbeta(0.5, 2.0)
    mu = data.weight1 .* nt.field[data.index1] + data.weight2 .* nt.field[data.index2] +
        data.weight3 .* nt.field[data.index3]
    joint + hyper + sum(logpdf.(Normal.(mu, 0.2), data.y))
end

@testset "Caller-owned GMRF graph, atomic prior, replay and native Reverse" begin
    fx = gmrf_rk_fixture()
    q = prepare_query(fx.built, fx.bound, :sampler)
    @test Base.invokelatest(q, fx.u) ≈ gmrf_rk_target(fx, fx.u) atol=2e-9 rtol=2e-10
    source = string(Base.remove_linenums!(kernel_expr(fx.bound, fx.built.layout)))
    @test !occursin("sampling_logdensity", source)
    replay = Core.eval(@__MODULE__, Expr(:macrocall,
        GlobalRef(ReactiveKernels, Symbol("@kernel")), LineNumberNode(0), Meta.parse(source)))
    replay_q = prepare_query((; spec=replay, layout=fx.built.layout), fx.bound, :sampler)
    @test Base.invokelatest(replay_q, fx.u) ≈ gmrf_rk_target(fx, fx.u) atol=2e-9
    catalogue = density_factors(fx.built, fx.bound)
    prior = select_factors(catalogue; kind=:prior, site=:field)
    @test length(prior) == 1
    @test factor_coordinates(prior) == collect(4:length(fx.u))
    # Ownership names the field's coordinates. Its density still reads the
    # sampled lengths and correlation as parents of the single joint factor.
    prior_q = prepare_query(fx.built, fx.bound, prior; u0=fx.u)
    nt = constrain(fx.built.layout, fx.u)
    @test Base.invokelatest(prior_q, fx.u) ≈
        gmrf_rk_oracle(nt.field, nt.spatial, nt.temporal, nt.L, fx.data.times) atol=2e-9
    cell = prepare_cell_query(fx.built, fx.bound, :y; locality=:error)
    footprint = ReactiveKernels.partition_footprint_kernel(cell.kernel)
    cells = Float64[]
    for i in eachindex(fx.data.y)
        expected = sort(unique([fx.data.index1[i], fx.data.index2[i], fx.data.index3[i]] .+ 3))
        @test Base.invokelatest(footprint, fx.u, i).unconstrained == expected
        # Structural support survives zero values; it is not a numerical
        # gradient threshold, and no range/correlation parent is read here.
        @test Base.invokelatest(footprint, zeros(length(fx.u)), i).unconstrained == expected
        push!(cells, Base.invokelatest(cell, fx.u, i))
    end
    likelihood = prepare_query(fx.built, fx.bound, :likelihood)
    @test sum(cells) ≈ Base.invokelatest(likelihood, fx.u) atol=1e-12
    native = string(code_expr(q))
    @test !occursin("sampling_logdensity", native)
    @test !occursin("brm_gmrf_logdensity", native)
    @test !occursin("_gmrf_density_graph(", native)
    sampler = prepare_sampler(fx.built, fx.bound, fx.u;
        backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
    value, gradient = sampler_value_and_gradient!(sampler, similar(fx.u), fx.u)
    @test value ≈ gmrf_rk_target(fx, fx.u) atol=2e-9
    for j in eachindex(fx.u)
        plus, minus = copy(fx.u), copy(fx.u)
        h = 2e-6 * max(1, abs(fx.u[j]))
        plus[j] += h
        minus[j] -= h
        @test gradient[j] ≈ (gmrf_rk_target(fx, plus) - gmrf_rk_target(fx, minus))/(2h) atol=2e-5 rtol=2e-6
    end
end
