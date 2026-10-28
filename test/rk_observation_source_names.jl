include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# A provider's definitions are evaluated in the build's own module, not in the
# module that defines the provider, so a law body sees neither `logsumexp` nor
# any other function of the consumer's imports. The two documented routes call
# one anyway: bind it as a leaf named from `entry`, or define the law in the
# consumer's module, where its names resolve, and bind it as a child graph.
module ObservationSourceNames
using BayesianRegressionModels, Distributions
import ReactiveKernels
import BayesianRegressionModels: _rk_observation_source!

# An equal-weight Normal mixture with components at `location` and
# `location + 1`, observed only through its RK scalar law.
mixture_leaf(location, scale) = error("observed through its RK scalar law only")
mixture_child(location, scale) = error("observed through its RK scalar law only")

function _rk_observation_source!(definitions, bindings, entry, ::typeof(mixture_leaf))
    lse = Symbol(entry, :_logsumexp)
    push!(bindings, lse => logsumexp)
    push!(definitions, :(ReactiveKernels.@kernel $entry(value, location, scale) = begin
        near = (value - location) / scale
        far = (value - location - 1.0) / scale
        near_logdensity = -0.5 * log(2 * pi) - log(scale) - 0.5 * near * near
        far_logdensity = -0.5 * log(2 * pi) - log(scale) - 0.5 * far * far
        logmass = $lse((near_logdensity - log(2.0), far_logdensity - log(2.0)))
        return logmass
    end))
    :done
end

ReactiveKernels.@kernel mixture_law(value, location, scale) = begin
    near = (value - location) / scale
    far = (value - location - 1.0) / scale
    near_logdensity = -0.5 * log(2 * pi) - log(scale) - 0.5 * near * near
    far_logdensity = -0.5 * log(2 * pi) - log(scale) - 0.5 * far * far
    logmass = logsumexp((near_logdensity - log(2.0), far_logdensity - log(2.0)))
    return logmass
end

function _rk_observation_source!(definitions, bindings, entry, ::typeof(mixture_child))
    law = Symbol(entry, :_law)
    push!(bindings, law => mixture_law)
    push!(definitions, :(ReactiveKernels.@kernel $entry(value, location, scale) = begin
        density = $law(value, location, scale)
        return density
    end))
    :done
end

build(data, ::typeof(mixture_leaf)) = @brm data begin
    s ~ Normal(0, 1)
    sigma ~ Exponential(1)
    m = s * x
    y ~ mixture_leaf(m, sigma)
end
build(data, ::typeof(mixture_child)) = @brm data begin
    s ~ Normal(0, 1)
    sigma ~ Exponential(1)
    m = s * x
    y ~ mixture_child(m, sigma)
end
end

const NAMES_DATA = (; x=[0.2, 0.5, 0.7, -0.3, 1.1], y=[0.1, 1.4, 0.3, -1.0, 2.2])

@stestset "a law calls a function the build module does not bind" begin
    for (family, binding) in ((ObservationSourceNames.mixture_leaf,
                :y_scalar_logdensity_logsumexp),
            (ObservationSourceNames.mixture_child, :y_scalar_logdensity_law))
        saved = deepcopy(NAMES_DATA)
        brmi = ObservationSourceNames.build(NAMES_DATA, family)
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        @test sort(names) == [:s, :sigma]
        is, iσ = coordinate(names, :s), coordinate(names, :sigma)
        oracle(u) = begin
            s, sigma = u[is], exp(u[iσ])
            likelihood = sum(zip(NAMES_DATA.x, NAMES_DATA.y)) do (x, y)
                logpdf(MixtureModel([Normal(s * x, sigma), Normal(s * x + 1, sigma)]), y)
            end
            logpdf(Normal(), s) + logpdf(Exponential(), sigma) + u[iσ] + likelihood
        end
        for u in ([0.0, 0.0], [0.3, -0.2], [-0.4, 0.25])
            check_consumer_point(problem, u, oracle)
        end
        emitted = BRM._rk_emit_ast(backend.plan)
        @test binding in first.(emitted.bindings)
        @test !isdefined(
            Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)._rk_emit_module(
                emitted), :logsumexp)
        @test isequal(NAMES_DATA, saved)
    end
end
