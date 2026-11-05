# Exercise the actual optimizer source with a tiny analytic target. The three
# empty model types stand in only for unrelated public-method signatures, so
# this arithmetic test needs no model lowering or external dependency setup.
module SVIArithmetic
using Random, LinearAlgebra, Statistics
struct BRMI end
struct RKBRMI end
struct BRMNLMEModel end
include(joinpath(@__DIR__, "..", "src", "nlme_selection.jl"))
include(joinpath(@__DIR__, "..", "src", "nlme_svi.jl"))

struct QuadraticUnits
    globals::Vector{Int}
    units::Vector{Vector{Int}}
    plan::NamedTuple
    plans::Vector{NamedTuple}
    work::Vector{Nothing}
    calls::Vector{Int}
    precision::Vector{Float64}
    center::Vector{Float64}
end

QuadraticUnits(globals, units, plan, calls, precision, center) =
    QuadraticUnits(globals, units, plan, [plan], [nothing], calls, precision, center)
include("svi_arithmetic_support.jl")

_brm_svi_global_term(::QuadraticUnits, g) = (-sum(abs2, g) / 2, -g)
function _brm_svi_unit_term!(gg, gz, t::QuadraticUnits, i, g, z, plan, work)
    t.calls[i] += 1
    gg .= -t.precision[i] .* (g .- t.center[i])
    fill!(gz, 0.0)
    -t.precision[i] * sum(abs2, g .- t.center[i]) / 2
end
_brm_throw_if_interrupted(::Nothing) = nothing
_brm_update_progress!(::Nothing, message) = nothing
end

using Test, Random, LinearAlgebra
const S = SVIArithmetic

@testset "production SVI weighted iteration" begin
    t = S.QuadraticUnits([1], [[2], [3], [4]],
        (; footprints=[[1, i + 1] for i in 1:3]), zeros(Int, 3),
        [0.3, 4, 9], [-2.0, 0.5, 3])
    p = S._brm_svi_unit_sampler(:work, [1.0, 50, 4]).probabilities
    g = 0.7 + 0.2randn(Xoshiro(14))
    values = @. -t.precision * (g - t.center)^2 / 2
    gradients = @. -t.precision * (g - t.center)
    expected_value, expected_gradient = 0.0, 0.0
    for draws in Iterators.product(1:3, 1:3, 1:3)
        batch = S._brm_svi_weighted_batch(draws, p)
        q = S._brm_svi_params(t, [0.7], S.MeanFieldGlobal();
            global_scale=0.2, unit_scale=1.0)
        opt = S._BRMSVIAdam(2, 0.0)
        fill!(t.calls, 0)
        value = S._brm_svi_iteration!(Xoshiro(14), t, q, opt, batch.indices,
            0.0, 0.0; weights=batch.weights)
        @test value ≈ -g^2 / 2 + dot(batch.weights, values[batch.indices])
        @test opt.m[1] / (1 - opt.β1) ≈ -g +
            dot(batch.weights, gradients[batch.indices])
        @test t.calls == [i in batch.indices ? 2 : 0 for i in 1:3]
        probability = prod(p[i] for i in draws)
        expected_value += probability * value
        expected_gradient += probability * opt.m[1] / (1 - opt.β1)
    end
    @test expected_value ≈ -g^2 / 2 + sum(values)
    @test expected_gradient ≈ -g + sum(gradients)

    for method in (:uniform, :work), iterations in (0, 7)
        q = S._brm_svi_params(t, [0.7], S.MeanFieldGlobal();
            global_scale=0.2, unit_scale=1.0)
        fill!(t.calls, 0)
        fit = S.arithmetic_fit(Xoshiro(9), t, q; batch_size=3,
            max_iterations=iterations, lr_global=0.001, unit_step=0.1,
            polish_lr=0.01, window=100, patience=10, min_windows=40,
            polish_windows=20, unit_sampling=method)
        @test sum(fit.sampling.draws) == 3iterations
        @test fit.unit_gradient_calls == sum(t.calls) == 2sum(fit.sampling.visits)
        @test all(fit.sampling.visits .<= fit.sampling.draws)
        @test fit.sampling.method == method
        @test all(isfinite, fit.elbo)
        @test fit.iterations == iterations
    end

    # Zero learning rates make the two phases' optimizer state deterministic;
    # this exercises both final flushing paths of the production fit loop.
    for limit in (7, 20)
        q = S._brm_svi_params(t, [0.7], S.MeanFieldGlobal();
            global_scale=0.2, unit_scale=1.0)
        original = deepcopy(q)
        fit = S.arithmetic_fit(Xoshiro(9), t, q; batch_size=2,
            max_iterations=limit, lr_global=0.0, unit_step=0.0,
            polish_lr=0.0, window=2, patience=1, min_windows=2,
            polish_windows=2, unit_sampling=:work)
        @test fit.q.μ == original.μ
        @test all(isapprox.(fit.q.M, original.M))
        @test all(isapprox.(fit.q.Λ, original.Λ))
        @test all(isapprox.(fit.q.U, original.U))
        @test all(isfinite, fit.elbo)
        @test fit.converged == (limit == 20)
    end
end

@testset "lazy local averages equal averages of every iterate" begin
    rng = Xoshiro(82)
    q = (; M=[randn(rng, d) for d in 1:5],
        Λ=[Matrix{Float64}(I, d, d) for d in 1:5])
    lazy, eager = deepcopy(q), deepcopy(q)
    through = zeros(Int, 5)
    # Unit 5 is never visited; other units have different gaps between visits.
    for k in 1:31
        batch = k % 3 == 0 ? [1, 3] : [2, 4]
        S._brm_svi_average_units!(lazy, q, through, batch, k - 1)
        for i in batch
            q.M[i] .+= 0.1randn(rng, i)
            q.Λ[i] .+= 0.1Matrix{Float64}(I, i, i)
        end
        S._brm_svi_average!(eager.M, q.M, k)
        S._brm_svi_average!(eager.Λ, q.Λ, k)
    end
    S._brm_svi_average_units!(lazy, q, through, 1:5, 31)
    @test all(isapprox.(lazy.M, eager.M))
    @test all(isapprox.(lazy.Λ, eager.Λ))
    @test through == fill(31, 5)
    @test lazy.M[5] == q.M[5]
    before = deepcopy(lazy)
    S._brm_svi_average_units!(lazy, q, through, 1:5, 31)
    @test lazy == before
end
