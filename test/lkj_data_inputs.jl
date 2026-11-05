include(joinpath(@__DIR__, "rk_consumer_support.jl"))

lkj_problem(backend) = rk_logdensity_problem(backend;
    ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))

@brm function data_correlation(K, eta)
    L ~ LKJCholesky(K, eta)
    return L
end

@brm function nested_correlation(K, eta)
    inner ~ data_correlation(K, eta)
    return inner
end

@stestset "data-supplied LKJ dimensions and concentrations" begin
    data_builder = @brm begin
        L ~ LKJCholesky(K, eta)
        y ~ Normal(L[2, 1], 0.3)
    end
    component_builder = @brm begin
        L ~ data_correlation(K, eta)
        y ~ Normal(L[2, 1], 0.3)
    end
    nested_builder = @brm begin
        L ~ nested_correlation(K, eta)
        y ~ Normal(L[2, 1], 0.3)
    end
    derived_builder = @brm begin
        K = length(channels)
        eta = exp(log_eta)
        L ~ LKJCholesky(K, eta)
        y ~ Normal(L[2, 1], 0.3)
    end
    constant_builder = @brm begin
        K = constant(length(channels))
        eta = constant(exp(log_eta))
        L ~ LKJCholesky(K, eta)
        y ~ Normal(L[2, 1], 0.3)
    end
    expression_builder = @brm begin
        L ~ LKJCholesky(length(channels), exp(log_eta))
        y ~ Normal(L[2, 1], 0.3)
    end
    emitted_sources = Dict{Symbol,Any}()
    for K in (2, 4)
        eta = K/2 + 0.3
        data = (; K, eta, y=0.1, channels=collect(1:K), log_eta=log(eta))
        literal = brm(quote
            L ~ LKJCholesky($K, $eta)
            y ~ Normal(L[2, 1], 0.3)
        end; mod=@__MODULE__)(data)
        literal_rk, literal_problem = consumer_problem(literal)
        for (kind, builder) in ((:derived, derived_builder), (:constant, constant_builder),
                (:expression, expression_builder), (:top, data_builder),
                (:component, component_builder), (:nested, nested_builder))
            println("LKJ_CASE=", kind, " K=", K); flush(stdout)
            inputs = kind in (:derived, :constant, :expression) ?
                (; data.channels, data.log_eta, data.y) : (; data.K, data.eta, data.y)
            brmi = builder(inputs)
            rk, problem = consumer_problem(brmi)
            if kind in (:top, :component, :nested)
                emitted = BRM._rk_emit_ast(rk.plan)
                source = sprint(Base.show_unquoted, Expr(:block, emitted.defs..., emitted.main))
                @test occursin("LKJCholesky(K, eta)", source)
                if haskey(emitted_sources, kind)
                    @test source == emitted_sources[kind]
                else
                    emitted_sources[kind] = source
                end
            end
            sb = SBBRMI(brmi; mod=@__MODULE__)
            stan = consumer_stan(brmi, "lkj_data_$(kind)")
            transport = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
            @test LogDensityProblems.dimension(problem) == K * (K - 1) ÷ 2
            @test length(transport) == LogDensityProblems.dimension(problem)
            for u in (fill(0.17, LogDensityProblems.dimension(problem)),
                    collect(range(-0.35; step=0.6/max(LogDensityProblems.dimension(problem)-1, 1),
                        length=LogDensityProblems.dimension(problem))))
                value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
                literal_value, literal_gradient = LogDensityProblems.logdensity_and_gradient(literal_problem, u)
                @test value ≈ literal_value atol=2e-11 rtol=2e-11
                @test gradient ≈ literal_gradient atol=2e-10 rtol=2e-10
                mapping = [pair.rk => pair.stan for pair in transport.pairs]
                check_consumer_stan(problem, stan, mapping, rk, u)
                @test brm_stan_point_to_rk(transport,
                    brm_rk_point_to_stan(transport, u)) ≈ u atol=1e-12 rtol=1e-12
                checked = brm_check_coordinate_transport(transport, rk, stan.model, u)
                @test checked.factors == 1
                @test checked.max_error <= 1e-12
                if K == 2
                    oracle(v) = logpdf(Beta(data.eta, data.eta), (tanh(v[1]) + 1)/2) -
                        log(2) + log1p(-tanh(v[1])^2) + logpdf(Normal(tanh(v[1]), 0.3), data.y)
                    check_consumer_point(problem, u, oracle)
                end
            end
            future = merge(inputs, (; y=-0.2))
            replay = reprocess(rk, future)
            fresh = RKBRMI(builder(future))
            u = fill(0.11, LogDensityProblems.dimension(problem))
            @test LogDensityProblems.logdensity(lkj_problem(replay), u) ≈
                LogDensityProblems.logdensity(lkj_problem(fresh), u)
            concentration_inputs = kind in (:derived, :constant, :expression) ?
                merge(inputs, (; log_eta=log(eta + 0.4))) : merge(inputs, (; eta=eta + 0.4))
            rebound = reprocess(rk, concentration_inputs; freeze_constants=false)
            fresh_concentration = RKBRMI(builder(concentration_inputs))
            @test LogDensityProblems.logdensity(lkj_problem(rebound), u) ≈
                LogDensityProblems.logdensity(lkj_problem(fresh_concentration), u)
        end
    end
end

@stestset "one correlation component serves two dimensions in the same model" begin
    builder = @brm begin
        small ~ data_correlation(small_K, small_eta)
        large ~ data_correlation(large_K, large_eta)
        y ~ Normal(small[2, 1] + large[2, 1], 0.3)
    end
    brmi = builder((; small_K=2, small_eta=1.3, large_K=4, large_eta=2.3, y=0.1))
    rk, problem = consumer_problem(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    stan = consumer_stan(brmi, "lkj_data_two_dimensions")
    transport = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
    @test LogDensityProblems.dimension(problem) == 7
    @test length(transport) == 7
    mapping = [pair.rk => pair.stan for pair in transport.pairs]
    for u in (fill(0.17, 7), collect(range(-0.35, 0.25; length=7)))
        check_consumer_stan(problem, stan, mapping, rk, u)
        checked = brm_check_coordinate_transport(transport, rk, stan.model, u)
        @test checked.factors == 2
        @test checked.max_error <= 1e-12
    end
end

@stestset "data-supplied LKJ dimension in a correlated grouping" begin
    builder = @brm begin
        Omega ~ LKJCholesky(K, eta)
        mu ~ 1 + t + (0 + t | gr(g; cor=Omega))
        y ~ Normal(mu, 1.0)
    end
    literal = @brm begin
        Omega ~ LKJCholesky(2, 1.3)
        mu ~ 1 + t + (0 + t | gr(g; cor=Omega))
        y ~ Normal(mu, 1.0)
    end
    data = (; K=2, eta=1.3, g=[1, 2, 1, 2],
        t=[-0.4, 0.2, 0.6, 1.0], y=[0.1, -0.2, 0.4, 0.3])
    brmi = builder(data)
    rk, problem = consumer_problem(brmi)
    _, reference = consumer_problem(literal(data))
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    stan = consumer_stan(brmi, "lkj_data_grouping")
    transport = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
    n = LogDensityProblems.dimension(problem)
    @test length(transport) == n
    for u in (fill(0.17, n), collect(range(-0.35, 0.25; length=n)))
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        reference_value, reference_gradient = LogDensityProblems.logdensity_and_gradient(reference, u)
        @test value ≈ reference_value atol=2e-11 rtol=2e-11
        @test gradient ≈ reference_gradient atol=2e-10 rtol=2e-10
        stan_u = brm_rk_point_to_stan(transport, u)
        stan_gradient = similar(stan_u)
        stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u, stan_gradient;
            propto=false, jacobian=true)
        @test value ≈ brm_stan_logdensity_to_rk(transport, u, stan_value) atol=2e-10 rtol=2e-10
        @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-10 rtol=2e-10
        @test brm_check_coordinate_transport(transport, rk, stan.model, u).max_error <= 1e-12
    end
end

@stestset "LKJ data validation and unit dimension" begin
    builder = @brm begin
        L ~ LKJCholesky(K, eta)
        y ~ Normal(L[1, 1], 0.3)
    end
    for backend in (RKBRMI, SBBRMI)
        for K in (0, -1, 2.5, true, [2])
            @test_throws "integer dimension" backend(builder((; K, eta=2.0, y=0.1)))
        end
        for eta in (0.0, -1.0, Inf, NaN, true, [2.0])
            @test_throws "finite, strictly positive" backend(builder((; K=2, eta, y=0.1)))
        end
        unit = backend(builder((; K=1, eta=2.0, y=0.1)))
        @test unit !== nothing
        if backend === RKBRMI
            problem = lkj_problem(unit)
            @test LogDensityProblems.dimension(problem) == 0
            @test LogDensityProblems.logdensity(problem, Float64[]) ≈
                logpdf(Normal(1.0, 0.3), 0.1)
        end
    end
end
