# test/rk_rebuild_types.jl — a fresh build of one RKBRMI model reuses the first
# build's prepared sampler type (snag `rk-logdensity-pr-b8b1cc36`).
#
# A simulation study builds `RKBRMI(builder(data))` and `rk_logdensity_problem`
# once per data set. Before ReactiveKernels 0.1.307, a `const` child `@kernel`
# with its own data-bound `?:`, called inside another data-bound arm of an
# observation law, gave every build a new sampler type (RK snag
# `rk-0-1-206-every-58bdc046`), so every data set compiled its density and
# Enzyme gradient again, identical data included; and under
# `branch_arms = :all` side-by-side flag branches split into one plate per arm
# combination (RK snag `one-compiled-nat-bd7c7031`, fixed in 0.1.308). Here:
# three fresh builds (data 1, data 2, data 1 again) share one prepared type,
# by default when the data take the same arms, and under `branch_arms = :all`
# when they take different ones. Oracle for the values: the law written out by
# hand. Public synthetic models only.
#
# Run: julia --startup-file=no --project=test test/rk_rebuild_types.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module RebuildTypes
using BayesianRegressionModels, Distributions
using Distributions: ContinuousMultivariateDistribution
import ReactiveKernels
import BayesianRegressionModels: _rk_observation_source!

# A Bernoulli-logit term with a data-bound branch, as a consumer's case law
# writes it.
const BERN = ReactiveKernels.@kernel bern(y, eta) = begin
    value = y > 0 ? -log1pexp(-eta) : -log1pexp(eta)
    return value
end
# The branch kernel called directly: a control that never needed a rebuild.
const DIRECT = ReactiveKernels.@kernel direct_graph(y, eta) = begin
    value = BERN(y[1], eta) + BERN(y[2], eta)
    return value
end
# The branch kernel called inside another data-bound arm, beside a second
# flag branch: the shape that rebuilt.
const NESTED = ReactiveKernels.@kernel nested_graph(y, eta, d) = begin
    first = BERN(y[1], eta)
    second = d[1] > 0 ? BERN(y[2], eta) : 0.0
    third = d[2] > 0 ? -0.5 * eta * eta : 0.0
    value = first + second + third
    return value
end
direct(eta) = error("observed through its RK law only")
nested(eta, d) = error("observed through its RK law only")
BayesianRegressionModels.brm_distribution_type(::typeof(direct)) = ContinuousMultivariateDistribution
BayesianRegressionModels.brm_distribution_type(::typeof(nested)) = ContinuousMultivariateDistribution
function _rk_observation_source!(definitions, bindings, entry, ::typeof(direct))
    graph = Symbol(entry, :_graph)
    push!(bindings, graph => DIRECT)
    push!(definitions, :(ReactiveKernels.@kernel $entry(y, eta) = begin
        value = $graph(y, eta)
        return value
    end))
    :done
end
function _rk_observation_source!(definitions, bindings, entry, ::typeof(nested))
    graph = Symbol(entry, :_graph)
    push!(bindings, graph => NESTED)
    push!(definitions, :(ReactiveKernels.@kernel $entry(y, eta, d) = begin
        value = $graph(y, eta, d)
        return value
    end))
    :done
end

const DIRECT_MODEL = @brm begin
    a ~ Normal(0, 1)
    @plate for i in eachindex(y)
        y[i] ~ direct(a + x[i])
    end
end
const NESTED_MODEL = @brm begin
    a ~ Normal(0, 1)
    @plate for i in eachindex(y)
        y[i] ~ nested(a + x[i], d[i])
    end
end
# The nested law beside a vector cell latent, as a per-person effect.
const LATENT_MODEL = @brm begin
    a ~ Normal(0, 1)
    @plate for i in eachindex(y)
        z ~ Normal(zero[i], one)
        y[i] ~ nested(a + x[i] + z[1], d[i])
    end
end
end
const RT = RebuildTypes

# Every data set holds every flag value in every slot, so all three take the
# same arms; `other_arms` takes only some of them.
flags = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
switches = [[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
dataset(x, y, d) = (; y, x, d, zero=[zeros(2) for _ in y], one=1.0)
data1 = dataset([0.3, -0.2, 0.5, 0.1, -0.4, 0.2], flags, switches)
data2 = dataset([-0.1, 0.4, -0.3, 0.6, 0.0, -0.5], reverse(flags), reverse(switches))
other_arms = dataset([0.2, -0.3, 0.1, 0.4, -0.2, 0.3], fill([1.0, 1.0], 6), fill([1.0, 0.0], 6))

ad = AutoEnzyme(; mode=Enzyme.Reverse)
problem_of(model, data) = rk_logdensity_problem(RKBRMI(model(data)); ad_backend=ad)
query_type(problem) = typeof(problem.query)

bern(y, eta) = y > 0 ? -LogExpFunctions.log1pexp(-eta) : -LogExpFunctions.log1pexp(eta)
nested_oracle(data) = u -> logpdf(Normal(0, 1), u[1]) + sum(eachindex(data.y)) do i
    eta = u[1] + data.x[i]
    y, d = data.y[i], data.d[i]
    bern(y[1], eta) + (d[1] > 0 ? bern(y[2], eta) : 0.0) + (d[2] > 0 ? -0.5 * eta^2 : 0.0)
end

@stestset "fresh builds of one model share the prepared sampler type" begin
    for model in (RT.DIRECT_MODEL, RT.NESTED_MODEL, RT.LATENT_MODEL)
        first, second, again = (problem_of(model, d) for d in (data1, data2, data1))
        @test query_type(second) === query_type(first)
        @test query_type(again) === query_type(first)
    end
    # The shared type reads each build's own data.
    for data in (data1, data2)
        problem = problem_of(RT.NESTED_MODEL, data)
        for u in ([0.1], [-0.35])
            check_consumer_point(problem, u, nested_oracle(data))
        end
    end
end

@stestset "branch_arms = :all: data taking other arms share the type too" begin
    # ReactiveKernels' `[ReactiveKernels] branch_arms = "all"` preference, scoped
    # to this testset as ReactiveKernelsPPL's own tests scope it.
    ReactiveKernels._with_branch_arms(:all) do
        first = problem_of(RT.NESTED_MODEL, data1)
        other = problem_of(RT.NESTED_MODEL, other_arms)
        @test query_type(other) === query_type(first)
        check_consumer_point(other, [0.15], nested_oracle(other_arms))
        check_consumer_point(first, [0.15], nested_oracle(data1))
    end
end
