# Run: julia --project=test test/nlme_connected.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_connected_fixture.jl"))
using LinearAlgebra, LogDensityProblems, Random

@stestset "connected units preserve likelihoods, native gradients and subject priors" begin
    cases = [
        (connected_data(2, [1, 2], [1, 2]), [[1], [2]]),
        (connected_data(2, [1, 2], [2, 2]), [[1, 2]]),
        (connected_data(5, [1, 2, 4, 5, 1], [2, 3, 5, 5, 1]), [[1, 2, 3], [4, 5]]),
        (connected_data(2, [1, 2], [2, 2]; weight=[0.0, 1.0]), [[1, 2]]),
    ]
    for (data, members) in cases
        before = deepcopy(data)
        rk = RKBRMI(connected_model(data))
        original = brm_nlme_view(rk)
        m = brm_nlme_model(rk; ad_backend=NLME_AD, unit_partition=:connected)
        v = m.view
        @test v.unit_members == members
        @test v.subject_levels == original.levels == data.subject
        @test v.coordinate_subject == original.subject
        @test sum(length, v.subject_coordinates) == length(data.subject)
        @test sum(length, v.subject_families) == length(data.subject)
        @test v.families == original.families
        @test v.subject_unit == [findfirst(ms -> s in ms, members) for s in 1:length(data.subject)]
        @test all(c -> v.subject[c] == v.subject_unit[original.subject[c]],
            findall(!iszero, original.subject))
        @test data == before
        if any(length(ms) > 1 for ms in members)
            @test v.ragged
            @test isempty(v.mu_references)
            @test_throws BRMNLMEViewError brm_nlme_model(rk; ad_backend=NLME_AD)
        else
            strict = brm_nlme_model(rk; ad_backend=NLME_AD)
            @test v.subject_coordinates == strict.view.subject_coordinates
        end
        theta = fill(0.23, length(m.theta))
        sigma = Float64[]
        effects = collect(range(-0.2, 0.4; length=length(data.subject)))
        H = v.ragged ? [effects[ms] for ms in members] : reshape(copy(effects), 1, :)
        ll, gradients = brm_nlme_loglikelihoods_and_gradients(m, theta, sigma, H)
        expected = zeros(length(members))
        de = zeros(length(effects))
        for i in eachindex(data.y)
            a, b, w = data.left[i], data.right[i], data.weight[i]
            location = theta[1] + effects[a] + w * (theta[1] + effects[b])
            expected[v.subject_unit[a]] += logpdf(Normal(location, 0.5), data.y[i])
            residual = (data.y[i] - location) / 0.5^2
            de[a] += residual
            de[b] += w * residual
        end
        @test ll ≈ expected rtol=1e-12
        full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
        u = BRM._brm_nlme_point(m, theta, sigma, H)
        zero_H = v.ragged ? [zeros(length(ms)) for ms in members] : zeros(size(H))
        u0 = BRM._brm_nlme_point(m, theta, sigma, zero_H)
        joint_delta = 0.0
        for (i, ms) in enumerate(members)
            eta = BRM._brm_nlme_subject_eta(H, i)
            value, g = brm_nlme_loglikelihood_and_gradient(m, i, theta, sigma, eta)
            @test value ≈ expected[i] rtol=1e-12
            @test g ≈ de[ms] rtol=1e-10 atol=1e-11
            @test g ≈ BRM._brm_nlme_subject_eta(gradients, i) atol=1e-10
            @test brm_nlme_loglikelihood(m, i, theta, sigma, eta) ≈ value
            joint_delta += value - brm_nlme_loglikelihood(m, i, theta, sigma, zero(eta)) - sum(abs2, eta)/2
        end
        @test LogDensityProblems.logdensity(full, u) - LogDensityProblems.logdensity(full, u0) ≈ joint_delta atol=1e-10
        @test m.unit == m.cell_gradient.point
        @test BRM._brm_nlme_cell_state(m).plan.point == m.unit
        @test all(length(last(p)) == length(data.y) for p in m.attribution)
        if v.ragged
            patterns = BRM._brm_nlme_eta_sparsity(m)
            @test all(size(patterns[i]) == (length(ms), length(ms)) for (i, ms) in enumerate(members))
            # The transitive three-subject unit is a chain, not a dense clique.
            if length(first(members)) == 3
                @test !patterns[1][1, 3]
            end
        end
    end
end

@stestset "connected SVI counts authored factors once and samples whole units" begin
    data = connected_data(5, [1, 2, 4, 5, 1], [2, 3, 5, 5, 1])
    rk = RKBRMI(connected_model(data))
    m = brm_nlme_model(rk; ad_backend=NLME_AD, unit_partition=:connected)
    t = BRM._brm_svi_target(m)
    u = m.unit .+ 0.25 .* randn(Xoshiro(73), length(m.unit))
    g = u[t.globals]
    value, gradient = LogDensityProblems.logdensity_and_gradient(
        rk_logdensity_problem(rk; ad_backend=NLME_AD), u)
    P, gP = BRM._brm_svi_global_term(t, g)
    gg = zeros(length(g))
    terms = map(eachindex(t.units)) do i
        gz = zeros(length(t.units[i]))
        v = BRM._brm_svi_unit_term!(gg, gz, t, i, g, u[t.units[i]])
        (; value=v, gg=copy(gg), gz)
    end
    @test P + sum(x -> x.value, terms) ≈ value atol=1e-10
    @test gP + sum(x -> x.gg, terms) ≈ gradient[t.globals] atol=1e-10
    natural = BRM._brm_nlme_natural(rk, m.view, BRM._brm_nlme_families(rk, m.view), u)
    for (i, block) in enumerate(t.units)
        @test terms[i].gz ≈ gradient[block] atol=1e-10
        problem = brm_nlme_unit_problem(m, i, u[m.theta], u[m.sigma], natural.omega)
        @test LogDensityProblems.logdensity(problem, u[block]) ≈ terms[i].value atol=1e-10
    end
    @test sum(P + length(terms) * term.value for term in terms) / length(terms) ≈ value

    controls = (; batch_size=1, max_iterations=4, ndraws=4)
    full = brm_svi_fit(m; rng=Xoshiro(74), controls...)
    part = brm_svi_fit(m; rng=Xoshiro(74), controls..., stop_after=2)
    resumed = brm_svi_resume(m, part; ndraws=4)
    @test resumed.draws == full.draws
    @test resumed.diagnostics.elbo == full.diagnostics.elbo
    @test length(full.diagnostics.q.M) == 2
    @test length.(full.diagnostics.q.M) == [3, 2]

    carry = brm_svi_checkpoint(full)
    @test carry.unit_partition === :connected
    next_data = merge(data, (; y=data.y .+ 0.05))
    window = brm_svi_fit(carry, connected_model(next_data);
        rng=Xoshiro(75), max_iterations=1, ndraws=1)
    @test window.diagnostics.model.view.unit_partition === :connected
    @test brm_svi_checkpoint(window).updates == 1
    split_data = connected_data(5, collect(1:5), collect(1:5))
    @test_throws r"connected membership changed" brm_svi_fit(carry,
        connected_model(split_data); max_iterations=1, ndraws=0)
    @test_throws r"unit_partition changed" brm_svi_fit(carry,
        connected_model(split_data); unit_partition=:subjects, max_iterations=1, ndraws=0)
end

@stestset "dormant branches determine connected units structurally" begin
    data = (; subject=["a", "b", "c"], other=[2, 3, 1], y=[0.1, 0.2, 0.3])
    rk = RKBRMI(@brm data begin
        mu ~ 1 + (1 | subject)
        all_mu = mu
        @plate for i in eachindex(y)
            y[i] ~ Normal(ifelse(all_mu[other[i]] > 10, all_mu[other[i]], mu[i]), 1)
        end
    end)
    m = brm_nlme_model(rk; ad_backend=NLME_AD, unit_partition=:connected)
    @test m.view.unit_members == [[1, 2, 3]]
    for eta in ([0.1, 0.2, 0.3], [11.0, 0.2, 12.0])
        value, gradient = brm_nlme_loglikelihood_and_gradient(m, 1, [0.0], Float64[], eta)
        full, gradients = brm_nlme_loglikelihoods_and_gradients(m, [0.0], Float64[], [eta])
        @test value ≈ only(full)
        @test gradient ≈ only(gradients)
    end
end

@stestset "joint observation cells and shared-only factors remain indivisible" begin
    data = (; subject=["a", "b", "c"], left=[1, 3], right=[2, 3],
        y=[[0.1, 0.2], [-0.1, 0.4]], C=[1.0 0.2; 0.2 1.0], shared_y=[0.3])
    rk = RKBRMI(@brm data begin
        mu ~ 1 + (1 | subject)
        beta ~ Normal(0, 1)
        shared_y ~ Normal(beta, 1)
        @plate for i in eachindex(y)
            y[i] ~ MvNormal([mu[left[i]], mu[right[i]]], C)
        end
    end)
    m = brm_nlme_model(rk; ad_backend=NLME_AD, unit_partition=:connected)
    @test m.view.unit_members == [[1, 2], [3]]
    @test length(m.view.factor_supports[:y]) == 2
    attribution = Dict(m.attribution)
    @test attribution[:y] == [1, 2]
    @test attribution[:shared_y] == [1]
    values = brm_nlme_loglikelihoods(m, [0.2], [0.4], [[0.1, -0.2], [0.3]])
    expected = [logpdf(MvNormal([0.3, 0.0], data.C), data.y[1]) +
        logpdf(Normal(0.4, 1), only(data.shared_y)),
        logpdf(MvNormal([0.5, 0.5], data.C), data.y[2])]
    @test values ≈ expected atol=1e-10
    @test brm_nlme_loglikelihood(m, 2, [0.2], [0.4], [0.3]) ≈ expected[2]
end

@stestset "connected Laplace integrates the original independent Gaussian effects" begin
    data = connected_data(3, [1, 2, 3, 1], [2, 2, 3, 1])
    rk = RKBRMI(connected_model(data))
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, unit_partition=:connected,
        structured=false)
    m = p.model
    u = copy(m.unit)
    u[p.population] .= 0.17
    natural = BRM._brm_nlme_natural(rk, m.view, BRM._brm_nlme_families(rk, m.view), u)
    A = zeros(length(data.y), length(data.subject))
    for row in eachindex(data.y)
        A[row, data.left[row]] += 1
        A[row, data.right[row]] += data.weight[row]
    end
    covariance = only(only(natural.omega)) * A * A' + 0.5^2 * I
    likelihood = logpdf(MvNormal((1 .+ data.weight) .* only(u[m.theta]),
        Symmetric(covariance)), data.y)
    plan = rk_bound_plan(rk)
    prior = prepare_query(rk.model, plan, :prior)
    jacobian = prepare_query(rk.model, plan, :log_jacobian)
    local_coordinates = reduce(vcat, m.view.subject_coordinates)
    population_prior = Base.invokelatest(prior, u) + Base.invokelatest(jacobian, u) -
        sum(logpdf(Normal(), u[j]) for j in local_coordinates)
    value, gradient = LogDensityProblems.logdensity_and_gradient(p, u[p.population])
    @test value ≈ population_prior + likelihood atol=1e-8
    @test all(isfinite, gradient)
end
