# Public consumer-hook acceptance for BRM's joint completion laws. The
# ordinary Julia adapter supplies an independent normalization/support oracle.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

function owned_completion_density(F, args)
    names = ntuple(i -> Symbol(:argument_, i), length(args))
    spec = ReactiveKernelsPPL.sampling_distribution_graph(F, names)
    @test spec isa ReactiveKernels.KernelSpec
    spec, Base.invokelatest(ReactiveKernels.prepare, spec)
end

@stestset "joint completion graph preserves law arguments and normalization" begin
    for (F, args, values) in (
        (BRM._BRMShares{LogNormal}, (3, 6., [-0.2, 0.1, 0.3], 0.8), [1., 2., 3.]),
        (BRM._BRMShares{Gamma}, (3, 6., 2., [0.5, 1., 1.5]), [1., 2., 3.]),
        (BRM._BRMShares{Exponential}, (1, 2.), [2.]),
        (BRM._BRMGroupedShares{LogNormal}, ([2, 3], [3., 3.], 0.1, 0.7), [1., 2., 3.]),
        (BRM._BRMGroupedShares{Gamma}, (Int[], Float64[], 2., 0.5), Float64[]),
        (BRM._BRMLowerLaw{Normal,true}, (0.4, 0.1, 0.8), 1.2),
        (BRM._BRMLowerLaw{Normal,false}, (0.4, 0.1, 0.8), 1.2),
        (BRM._BRMLowerLaw{Normal,true}, ([0.4, 0.6], 0.1, [0.8, 1.2]), [1.2, 1.5]),
        (BRM._BRMLowerLaw{LogNormal,true}, (0., 0.1, 0.8), [0.7, 1.3]),
        (BRM._BRMLowerLaw{LogNormal,true}, (1., -3., 0.2), [1.01, 1.2]),
        (BRM._BRMLowerLaw{Exponential,true}, (0.4,), [0.8, 1.2]),
        (BRM._BRMLowerLaw{Gamma,true}, (0.4, 2., 0.8), [0.8, 1.2]),
        (BRM._BRMLowerLaw{Cauchy,true}, (0.4, 0.2, 0.8), [0.8, 1.2]),
        (BRM._BRMLowerLaw{Laplace,true}, (0.4, 0.2, 0.8), [0.8, 1.2]),
        (BRM._BRMLowerLaw{Logistic,true}, (0.4, 0.2, 0.8), [0.8, 1.2]),
        (BRM._BRMLowerShares{BRM._BRMLowerLaw{LogNormal,true}},
            (2, 3., [0.4, 0.6], 0.1, 0.8), [1., 2.]),
        (BRM._BRMLowerShares{BRM._BRMLowerLaw{Normal,false}},
            (1, 0.4, 0.4, 0.1, 0.8), [0.4]),
        (BRM._BRMLowerGroupedShares{BRM._BRMLowerLaw{LogNormal,true}},
            ([2, 3], [3., 2.], [0.4, 0.6, 2.], 0.1, 0.8), [1., 2., 2.]))
        original = deepcopy((values, args))
        spec, q = owned_completion_density(F, args)
        expected = ReactiveKernelsPPL.sampling_logdensity(F(args...), values)
        actual = Base.invokelatest(q, values, args...)
        @test actual ≈ expected rtol=2e-12 atol=2e-12
        @test Base.invokelatest(q, values, args...) == actual
        @test isequal((values, args), original)
        # The public inventory descends into plates/scans, not lazy branch
        # arms. Guarded share laws retain the plate in their local helper.
        if !(F <: Union{BRM._BRMLowerShares,BRM._BRMLowerGroupedShares})
            @test any(e -> e.kind === :plate, ReactiveKernels.recipe_inventory(spec))
        end
    end
end

@stestset "completion graphs retain support and dimension refusals" begin
    for normalized in (false, true)
        F = BRM._BRMLowerLaw{Normal,normalized}
        _, q = owned_completion_density(F, (0.5, 0., 1.))
        @test Base.invokelatest(q, 0.4, 0.5, 0., 1.) == -Inf
        @test Base.invokelatest(q, 0.8, Inf, 0., 1.) == -Inf
        @test Base.invokelatest(q, 0.8, NaN, 0., 1.) == -Inf
        @test_throws DimensionMismatch Base.invokelatest(q, [0.8, 0.9], [0.5], 0., 1.)
    end
    F = BRM._BRMLowerShares{BRM._BRMLowerLaw{Normal,true}}
    _, q = owned_completion_density(F, (2, 1., 0.5, 0., 1.))
    @test Base.invokelatest(q, [0.5, 0.5], 2, 1., 0.5, 0., 1.) == -Inf
    _, q = owned_completion_density(BRM._BRMShares{LogNormal}, (2, 3., 0., 1.))
    @test_throws DimensionMismatch Base.invokelatest(q, [1., 2.], 2, 3., [0.], 1.)
    # Weibull had no public mapping when this suite was written; at the test
    # pin RK maps it (and every positive family rkppl-use documents), so the
    # shares adapter now composes its graph.
    @test ReactiveKernelsPPL.sampling_distribution_graph(BRM._BRMShares{Weibull},
        (:n, :total, :shape, :scale)) isa ReactiveKernels.KernelSpec
end

@stestset "a newly mapped completion law (Weibull) keeps its density and native Reverse" begin
    data = (; x=Union{Missing,Float64}[missing, missing], total=[3.],
        g=[1, 1], cells=[1], y=[1.1, 1.8])
    model = @brm data begin
        rows = groups(g => cells)
        mi(x)::shares(total, rows) ~ Weibull(2., 3.)
        y ~ Normal(x, 0.5)
    end
    oracle(u) = begin
        x = BRM._brm_shares_values(u, 3.)
        sum(logpdf.(Weibull(2., 3.), x)) + sum(logpdf.(Normal.(x, 0.5), data.y)) +
            sum(log, x) - log(3.)
    end
    for mode in (:centered, :noncentered)
        rk = check_rk_source_roundtrip(RKBRMI(model; completions=mode))
        problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        @test LogDensityProblems.dimension(problem) == 1
        for point in ([0.], [0.3])
            check_consumer_point(problem, point, oracle)
        end
    end
end
