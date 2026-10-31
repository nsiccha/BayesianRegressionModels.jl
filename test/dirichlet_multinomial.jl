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
    data = (; y=[[2, 3, 1]], n=[5], p=[0.2, 0.5, 0.3])
    backend = RKBRMI(dm_shared(data))
    # Distributions 0.25.132's logpdf does not check insupport for this family.
    # Keep the missing support check visible until the upstream law is fixed
    # (ReactiveKernels snag dirichlet-multin-457e95cf).
    @test_broken only(Base.invokelatest(dm_pointwise(backend), [0.0]).y) == -Inf
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
