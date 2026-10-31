# Fixed-total count vectors are one likelihood event per plate cell.
# Run: julia --startup-file=no --project=test test/dirichlet_multinomial.jl
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

dm_shared(data) = @brm data begin
    phi ~ LogNormal(3, 1)
    @plate for k in eachindex(n)
        y[k] ~ DirichletMultinomial(n[k], phi .* p)
    end
end

dm_intensities(data) = @brm data begin
    phi ~ LogNormal(3, 1)
    beta ~ Normal(0, 1)
    @plate for k in eachindex(n)
        intensity = exp.(beta .* x[k])
        p = intensity ./ sum(intensity)
        y[k] ~ DirichletMultinomial(n[k], phi .* p)
    end
end

function dm_pointwise(backend)
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    prepare_query(backend.model, ext._rk_translated_plan(backend.plan), :pointwise)
end

@stestset "Dirichlet-multinomial shared concentration and probabilities" begin
    @test :DirichletMultinomial in names(BRM)
    @test BRM.DirichletMultinomial === Distributions.DirichletMultinomial
    data = (; y=[[2, 3, 1], [0, 4, 0]], n=[6, 4], p=[0.2, 0.5, 0.3])
    backend, problem = consumer_problem(dm_shared(data))
    @test coordinate_names(backend.model.layout) == [:phi]
    parts(u) = [logpdf(DirichletMultinomial(n, exp(u[1]) .* data.p), y)
        for (n, y) in zip(data.n, data.y)]
    oracle(u) = logpdf(LogNormal(3, 1), exp(u[1])) + u[1] + sum(parts(u))
    pointwise = dm_pointwise(backend)
    for u in ([0.0], [1.2], [-0.7])
        check_consumer_point(problem, u, oracle)
        @test Base.invokelatest(pointwise, u).y ≈ parts(u)
    end
end

@stestset "Joint multivariate plate observations preserve vector arguments" begin
    data = (; y=[[0.2, -0.4], [1.1, 0.7, -0.2]],
        x=[[0.5, -0.3], [0.2, 0.8, -0.6]],
        covariance=[[1.0 0.3; 0.3 0.8], [1.0 0.1 0.2; 0.1 0.9 0.0; 0.2 0.0 1.2]])
    model = @brm data begin
        beta ~ Normal(0, 1)
        @plate for k in eachindex(y)
            y[k] ~ MvNormal(beta .* x[k], covariance[k])
        end
    end
    backend, problem = consumer_problem(model)
    oracle(u) = logpdf(Normal(), only(u)) +
        sum(logpdf(MvNormal(only(u) .* x, covariance), y)
            for (x, covariance, y) in zip(data.x, data.covariance, data.y))
    for u in ([0.0], [0.4], [-0.8])
        check_consumer_point(problem, u, oracle)
    end
end

@stestset "Dirichlet-multinomial ragged model-derived probabilities" begin
    data = (; y=[[2, 3, 1], [1, 3], [4]], n=[6, 4, 4],
        x=[[-0.8, 0.3, 1.1], [0.7, -0.2], [0.5]])
    backend, problem = consumer_problem(dm_intensities(data))
    names = coordinate_names(backend.model.layout)
    iphi, ibeta = coordinate(names, :phi), coordinate(names, :beta)
    function parts(u)
        map(eachindex(data.n)) do k
            intensity = exp.(u[ibeta] .* data.x[k])
            alpha = exp(u[iphi]) .* intensity ./ sum(intensity)
            logpdf(DirichletMultinomial(data.n[k], alpha), data.y[k])
        end
    end
    oracle(u) = logpdf(LogNormal(3, 1), exp(u[iphi])) + u[iphi] +
        logpdf(Normal(), u[ibeta]) + sum(parts(u))
    pointwise = dm_pointwise(backend)
    for u in ([0.0, 0.0], [1.2, -0.4], [-0.7, 0.8])
        check_consumer_point(problem, u, oracle)
        @test Base.invokelatest(pointwise, u).y ≈ parts(u)
    end
end

@stestset "Dirichlet-multinomial input support boundary" begin
    # The explicit Distributions source pin in setup_env.jl owns support
    # checking; BRM and RKPPL delegate to that same public law.
    for (n, y) in ((5, [2, 3, 1]), (7, [2, 3, 1]),
            (6, [-1, 4, 3]), (6, [1.5, 3.5, 1.0]))
        data = (; y=[y], n=[n], p=[0.2, 0.5, 0.3])
        before = deepcopy(data)
        backend = RKBRMI(dm_shared(data))
        @test only(Base.invokelatest(dm_pointwise(backend), [0.0]).y) == -Inf
        problem = rk_logdensity_problem(backend;
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        @test LogDensityProblems.logdensity(problem, [0.0]) == -Inf
        @test data == before
    end
    # Distributions' constructor requires a strictly positive total.
    empty_data = (; y=[[0, 0, 0]], n=[0], p=[0.2, 0.5, 0.3])
    @test_throws DomainError Base.invokelatest(dm_pointwise(RKBRMI(dm_shared(empty_data))), [0.0])
end

@stestset "Dirichlet-multinomial NLME subject likelihoods and gradients" begin
    data = (; subject=["a", "b", "c"], y=[[2, 3, 1], [1, 3], [4, 1]],
        n=[6, 4, 5], x=[[-0.8, 0.3, 1.1], [0.7, -0.2], [-0.5, 0.5]])
    model = @brm data begin
        phi ~ LogNormal(3, 1)
        slope ~ 1 + (1 | subject)
        @plate for k in eachindex(n)
            intensity = exp.(slope[k] .* x[k])
            p = intensity ./ sum(intensity)
            y[k] ~ DirichletMultinomial(n[k], phi .* p)
        end
    end
    backend = RKBRMI(model)
    m = brm_nlme_model(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    @test length(m.theta) == length(m.sigma) == 1
    @test m.view.coordinates[only(m.sigma)] == :phi
    theta, sigma, eta = [0.3], [1.2], [0.2 -0.4 0.1]
    function subject_density(i, h)
        row = findfirst(==(string(m.view.levels[i])), data.subject)
        intensity = exp.((only(theta) + h) .* data.x[row])
        alpha = exp(only(sigma)) .* intensity ./ sum(intensity)
        logpdf(DirichletMultinomial(data.n[row], alpha), data.y[row])
    end
    expected = [subject_density(i, eta[1, i]) for i in eachindex(m.view.levels)]
    step = 1e-5
    gradient = [(subject_density(i, eta[1, i] + step) -
        subject_density(i, eta[1, i] - step)) / (2step) for i in eachindex(expected)]
    values, gradients = brm_nlme_loglikelihoods_and_gradients(m, theta, sigma, eta)
    @test values ≈ expected atol=2e-11 rtol=2e-11
    @test vec(gradients) ≈ gradient atol=2e-8 rtol=2e-8
    for i in eachindex(expected)
        value, grad = brm_nlme_loglikelihood_and_gradient(m, i, theta, sigma, eta[:, i])
        @test value ≈ expected[i] atol=2e-11 rtol=2e-11
        @test only(grad) ≈ gradient[i] atol=2e-8 rtol=2e-8
    end
    hessians = BRM._brm_nlme_hessian_blocks(m, theta, sigma, eta)
    second_step = 1e-3
    for i in eachindex(expected)
        curvature = (subject_density(i, eta[1, i] + second_step) - 2expected[i] +
            subject_density(i, eta[1, i] - second_step)) / second_step^2
        @test only(hessians[i]) ≈ curvature atol=2e-6 rtol=2e-6
    end
end

function check_dm_stan(model, data, label, points)
    backend, problem = consumer_problem(model)
    sb = SBBRMI(model; mod=@__MODULE__, total_groups=())
    stan = BRM.stan_instantiate(sb;
        path=joinpath(tempdir(), "brm-rk-consumer", label * ".stan"))
    @test Symbol.(BridgeStan.param_unc_names(stan.model)) ==
        coordinate_names(backend.model.layout)
    descriptor = brm_descriptor(sb)
    names = BridgeStan.param_names(stan.model; include_tp=true, include_gq=true)
    likelihood = brm_output(descriptor, :y; role=:pointwise_loglik)
    predictive = brm_output(descriptor, :y; role=:posterior_predictive)
    likelihood_coordinates = brm_output_coordinates(likelihood, names)
    predictive_coordinates = brm_output_coordinates(predictive, names)
    @test length(likelihood_coordinates) == length(data.n)
    @test predictive.segments == cumsum(length.(data.y))
    pointwise = dm_pointwise(backend)
    for (seed, u) in enumerate(points)
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(
            stan.model, u; propto=false, jacobian=true)
        @test stan_value ≈ value atol=2e-11 rtol=2e-11
        @test stan_gradient ≈ gradient atol=2e-10 rtol=2e-10
        values = BridgeStan.param_constrain(stan.model, u;
            include_tp=true, include_gq=true,
            rng=BridgeStan.StanRNG(stan.model, seed))
        @test values[likelihood_coordinates] ≈
            Base.invokelatest(pointwise, u).y atol=2e-11 rtol=2e-11
        draws = values[predictive_coordinates]
        @test all(isinteger, draws)
        @test all(>=(0), draws)
        offset = 0
        for k in eachindex(data.n)
            width = length(data.y[k])
            @test sum(draws[offset+1:offset+width]) == data.n[k]
            offset += width
        end
    end
    stan
end

@stestset "Dirichlet-multinomial Stan shared concentration and predictions" begin
    data = (; y=[[2, 3, 1], [0, 4, 0]], n=[6, 4], p=[0.2, 0.5, 0.3])
    check_dm_stan(dm_shared(data), data, "dm_shared",
        ([0.0], [1.2], [-0.7]))
end

@stestset "Dirichlet-multinomial Stan ragged live probabilities and predictions" begin
    data = (; y=[[2, 3, 1], [1, 3], [4]], n=[6, 4, 4],
        x=[[-0.8, 0.3, 1.1], [0.7, -0.2], [0.5]])
    check_dm_stan(dm_intensities(data), data, "dm_intensities",
        ([0.0, 0.0], [1.2, -0.4], [-0.7, 0.8]))
end

@stestset "Dirichlet-multinomial Stan rejects mismatched totals" begin
    for total in (5, 7)
        data = (; y=[[2, 3, 1], [0, 4, 0]], n=[total, 4], p=[0.2, 0.5, 0.3])
        stan = consumer_stan(dm_shared(data), "dm_shared")
        @test_throws ErrorException BridgeStan.log_density(stan.model, [0.0];
            propto=false, jacobian=true)
    end
end
