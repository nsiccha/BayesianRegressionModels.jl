# A source provider replaces a law's implementation without changing its event.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
import StatsFuns

module JointObservationSources
using BayesianRegressionModels, Distributions
import ReactiveKernels, LogExpFunctions, StatsFuns
import BayesianRegressionModels: _rk_observation_source!, brm_distribution_type

function shared_location end
brm_distribution_type(::typeof(shared_location)) = ContinuousMultivariateDistribution

function _rk_observation_source!(definitions, bindings, entry, ::typeof(shared_location))
    row = Symbol(entry, :_row)
    lcdf = Symbol(entry, :_lcdf)
    lse = Symbol(entry, :_lse)
    push!(bindings, lcdf => StatsFuns.normlogcdf, lse => LogExpFunctions.logsumexp)
    push!(definitions, :(ReactiveKernels.@kernel $row(y, mu, sigma, limit) = begin
        density = if y < limit
            -Inf
        elseif y == limit
            $lcdf((limit - mu) / sigma)
        else
            -log(sigma) - log(2pi) / 2 - ((y - mu) / sigma)^2 / 2
        end
        return density
    end))
    push!(definitions, :(ReactiveKernels.@kernel $entry(y, means, sigma, limit, log_weights) = begin
        terms = ReactiveKernels.plate(axes(means, 2)) do k
            log_weights[k] + sum($row(y[r], means[r, k], sigma[r], limit[r])
                for r in eachindex(y); init=0.0)
        end
        density = $lse(terms) - $lse(log_weights)
        return density
    end))
    :done
end

function matrix_location end
brm_distribution_type(::typeof(matrix_location)) = ContinuousMatrixDistribution
function _rk_observation_source!(definitions, bindings, entry, ::typeof(matrix_location))
    push!(definitions, :(ReactiveKernels.@kernel $entry(y, shift) = begin
        residual = y .- shift
        density = -length(y) * log(2pi) / 2 - sum(abs2, residual) / 2
        return density
    end))
    :done
end
end

using .JointObservationSources: shared_location, matrix_location

function event_density(y, means, sigma, limit, log_weights, shift)
    terms = map(axes(means, 2)) do k
        log_weights[k] + sum(eachindex(y); init=0.0) do r
            law = Normal(means[r, k] + shift, sigma[r])
            y[r] < limit[r] ? -Inf :
                y[r] == limit[r] ? logcdf(law, limit[r]) : logpdf(law, y[r])
        end
    end
    LogExpFunctions.logsumexp(terms) - LogExpFunctions.logsumexp(log_weights)
end

event_model(data) = @brm data begin
    shift ~ Normal(0, 1)
    @plate for i in eachindex(y)
        y[i] ~ shared_location(means[i] .+ shift, sigma[i], limit[i], log_weights[i])
    end
end

@stestset "Source providers preserve joint vector events" begin
    data = (; y=[[0.5, -0.2], [1.3], Float64[]],
        means=[[0.3 2.5; -0.4 2.2], [0.8 -0.6], zeros(0, 2)],
        sigma=[[0.6, 0.8], [0.7], Float64[]],
        limit=[[-Inf, -0.2], [-Inf], Float64[]],
        log_weights=[log.([2.0, 3.0]), log.([4.0, 1.0]), log.([3.0, 7.0])])
    before = deepcopy(data)
    backend, problem = consumer_problem(event_model(data))
    parts(u) = [event_density(y, m, s, l, w, only(u))
        for (y, m, s, l, w) in zip(data.y, data.means, data.sigma, data.limit, data.log_weights)]
    oracle(u) = logpdf(Normal(), only(u)) + sum(parts(u))
    pointwise = prepare_query(backend.model, rk_bound_plan(backend), :pointwise)
    for u in ([-0.3], [0.0], [0.7], [50.0])
        check_consumer_point(problem, u, oracle)
        @test Base.invokelatest(pointwise, u).y ≈ parts(u)
        @test length(Base.invokelatest(pointwise, u).y) == length(data.y)
    end
    # Independent row-wise mixtures are a different statistical model.
    rowwise = sum(eachindex(first(data.y))) do r
        event_density(first(data.y)[r:r], first(data.means)[r:r, :],
            first(data.sigma)[r:r], first(data.limit)[r:r], first(data.log_weights), 0.7)
    end
    @test abs(rowwise - first(parts([0.7]))) > 0.01
    @test isequal(data, before)

    weights = [0.5, 2.0, 3.0]
    weighted_model = @brm merge(data, (; weights)) begin
        shift ~ Normal(0, 1)
        @plate for i in eachindex(y)
            y[i] ~ weighted(shared_location, weights[i], means[i] .+ shift,
                sigma[i], limit[i], log_weights[i])
        end
    end
    weighted_backend, weighted_problem = consumer_problem(weighted_model)
    weighted_oracle(u) = logpdf(Normal(), only(u)) + sum(weights .* parts(u))
    for u in ([-0.3], [0.7])
        check_consumer_point(weighted_problem, u, weighted_oracle)
    end
end

@stestset "Reported shared-location vector retains its joint law" begin
    data = (; y=[[0.2, 0.4]], means=[[0.0 3.0; 0.1 2.7]], sigma=[[0.4, 0.3]],
        limit=[[-Inf, 0.4]], log_weights=[log.([0.3, 0.7])])
    backend, problem = consumer_problem(event_model(data))
    oracle(u) = logpdf(Normal(), only(u)) + event_density(only(data.y),
        only(data.means), only(data.sigma), only(data.limit), only(data.log_weights), only(u))
    for u in ([-0.1], [0.0], [0.7], [50.0])
        value, gradient = check_consumer_point(problem, u, oracle)
        println("reported_case shift=", only(u), " density=", value,
            " gradient=", only(gradient), " oracle=", oracle(u))
    end
end

@stestset "Source provider whole-vector model-scope observation" begin
    data = (; y=[0.5, -0.2], means=[0.3 2.5; -0.4 2.2], sigma=[0.6, 0.8],
        limit=[-Inf, -0.2], log_weights=log.([2.0, 3.0]))
    model = @brm data begin
        shift ~ Normal(0, 1)
        means_shifted = means .+ shift
        y ~ shared_location(means_shifted, sigma, limit, log_weights)
    end
    backend, problem = consumer_problem(model)
    oracle(u) = logpdf(Normal(), only(u)) +
        event_density(data.y, data.means, data.sigma, data.limit, data.log_weights, only(u))
    for u in ([-0.3], [0.0], [0.7])
        check_consumer_point(problem, u, oracle)
    end

end

@stestset "Source providers preserve matrix events" begin
    data = (; y=[[0.5 -0.2; 0.8 0.1], reshape([1.3], 1, 1)])
    model = @brm data begin
        shift ~ Normal(0, 1)
        @plate for i in eachindex(y)
            y[i] ~ matrix_location(shift)
        end
    end
    backend, problem = consumer_problem(model)
    oracle(u) = logpdf(Normal(), only(u)) +
        sum(sum(logpdf.(Normal(only(u), 1), y)) for y in data.y)
    for u in ([-0.3], [0.0], [0.7])
        check_consumer_point(problem, u, oracle)
    end

    single = @brm (; y=first(data.y)) begin
        shift ~ Normal(0, 1)
        y ~ matrix_location(shift)
    end
    single_backend, single_problem = consumer_problem(single)
    single_oracle(u) = logpdf(Normal(), only(u)) +
        sum(logpdf.(Normal(only(u), 1), first(data.y)))
    for u in ([-0.3], [0.0], [0.7])
        check_consumer_point(single_problem, u, single_oracle)
    end
end

# An empty comprehension over a DataFrame column infers `Any[]`. Its cells are
# read as a typed `Vector{Float64}[]`'s are: a zero-group plate observes nothing
# and keeps the priors, and a populated `Any[...]` column of arrays reads as its
# typed twin (snag rkbrmi-on-a-brm-c7c1d415).
group_model(data) = @brm data begin
    shift ~ Normal(0, 1)
    @plate for g in eachindex(y)
        group_mean[g] = means[g] .+ shift
        y[g] ~ shared_location(group_mean[g], sigma[g], limit[g], log_weights[g])
    end
end

@stestset "Type-erased response columns read their cells" begin
    prior(u) = logpdf(Normal(), only(u))
    typed = (; y=Vector{Float64}[], means=Matrix{Float64}[], sigma=Vector{Float64}[],
        limit=Vector{Float64}[], log_weights=Vector{Float64}[])
    for data in (typed, merge(typed, (; y=Any[])), map(_ -> Any[], typed))
        before = deepcopy(data)
        backend, problem = consumer_problem(group_model(data))
        plan = rk_bound_plan(backend)
        likelihood, prior_query, pointwise = (prepare_query(backend.model, plan, preset)
            for preset in (:likelihood, :prior, :pointwise))
        for u in ([-0.3], [0.7])
            @test Base.invokelatest(likelihood, u) == 0.0
            @test Base.invokelatest(prior_query, u) ≈ prior(u)
            @test isempty(Base.invokelatest(pointwise, u).y)
            check_consumer_point(problem, u, prior)
        end
        @test isequal(data, before)
    end

    populated = (; y=[[0.5, -0.2], [1.3]], means=[[0.3 2.5; -0.4 2.2], [0.8 -0.6]],
        sigma=[[0.6, 0.8], [0.7]], limit=[[-Inf, -0.2], [-Inf]],
        log_weights=[log.([2.0, 3.0]), log.([4.0, 1.0])])
    parts(u) = [event_density(y, m, s, l, w, only(u)) for (y, m, s, l, w) in
        zip(populated.y, populated.means, populated.sigma, populated.limit,
            populated.log_weights)]
    for data in (populated, merge(populated, (; y=Any[populated.y...])),
            map(column -> Any[column...], populated))
        backend, problem = consumer_problem(group_model(data))
        pointwise = prepare_query(backend.model, rk_bound_plan(backend), :pointwise)
        for u in ([-0.3], [0.7])
            check_consumer_point(problem, u, u -> prior(u) + sum(parts(u)))
            @test Base.invokelatest(pointwise, u).y ≈ parts(u)
        end
    end
end
