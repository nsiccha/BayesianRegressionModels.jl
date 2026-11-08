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

# Unit normals centered at 1, away from the zero-mean start, whose last gradient
# entry can be made non-finite.
struct PoisonableUnits
    globals::Vector{Int}
    units::Vector{Vector{Int}}
    plan::NamedTuple
    plans::Vector{NamedTuple}
    work::Vector{Nothing}
    m::NamedTuple
    poisoned::Bool
end

_brm_svi_global_term(::PoisonableUnits, g) = (-sum(abs2, g) / 2, -g)
function _brm_svi_unit_term!(gg, gz, t::PoisonableUnits, i, g, z, plan, work)
    fill!(gg, 0.0)
    gz .= 1 .- z
    t.poisoned && (gz[end] = NaN)
    -sum(abs2, z .- 1) / 2
end
end

using Test, Random, LinearAlgebra
const S = SVIArithmetic

# BRMSVINonfiniteError's own test (test/nlme_svi.jl) reaches dense factors under
# uniform batches. This covers the other local family and sampler: a structured
# factor's optimizer and a work-weighted batch must also stay untouched.
@testset "a non-finite unit stops every local family and sampler before any update" begin
    names = ["g", "a1", "a2", "a3", "b1", "b2", "b3"]
    target(poisoned) = S.PoisonableUnits([1], [[2, 3, 4], [5, 6, 7]],
        (; footprints=[[1, 2, 3, 4], [1, 5, 6, 7]]),
        [(; footprints=[[1, 2, 3, 4], [1, 5, 6, 7]])], [nothing],
        (; view=(; coordinates=names), unit=fill(-1.0, 7)), poisoned)
    for dense_limit in (64, 0), method in (:uniform, :work)
        start() = S._brm_svi_params(target(false), [0.7], S.MeanFieldGlobal();
            global_scale=0.2, unit_scale=1.0, unit_dense_limit=dense_limit, unit_rank=1)
        controls = S._brm_svi_controls(2; batch_size=2, max_iterations=10,
            lr_global=0.01, unit_step=0.1, polish_lr=0.05, window=100, patience=10,
            min_windows=40, polish_windows=20, unit_sampling=method)
        t, q = target(true), start()
        before = deepcopy(q)
        run = S._brm_svi_run(Xoshiro(5), q, controls, t)
        run.iteration = 7
        e = try
            S._brm_svi_advance!(run, t, controls)
            nothing
        catch err
            err
        end
        @test e isa S.BRMSVINonfiniteError
        @test e.iteration == 8
        for p in e.terms
            @test p.term === :unit && p.draw_finite
            @test :unit_gradient_plus in p.quantities
            @test :unit_gradient_minus in p.quantities
            @test p.coordinates == [names[t.units[p.unit][end]]]
            # The evaluated pair, packed: the globals' draw, the unit's own draws
            # around its mean, the rest at the model's point.
            own = t.units[p.unit]
            @test (p.points.plus[own] .+ p.points.minus[own]) ./ 2 ≈ before.M[p.unit]
            @test p.points.plus[1] == p.points.minus[1] != before.μ[1]
            rest = setdiff(1:7, 1, own)
            @test p.points.plus[rest] == p.points.minus[rest] == fill(-1.0, length(rest))
            @test p.largest.coordinate in names[vcat(1, own)]
            @test abs(p.largest.value) == maximum(abs, vcat(p.points.plus[vcat(1, own)],
                p.points.minus[vcat(1, own)]))
        end
        @test !isempty(e.terms)
        @test run.adam.t == 0 && q.μ == before.μ && q.ρ == before.ρ && q.Β == before.Β
        @test q.M == before.M
        @test all(Matrix(q.Λ[i]) == Matrix(before.Λ[i]) for i in 1:2)
        @test all(o === nothing || o.t == 0 for o in q.unit_opt)
        @test isempty(run.trace) && run.iteration == 7

        # A dense unit on an orthogonal design reports the pair that failed, the first.
        if dense_limit > 0
            orthogonal = merge(controls, (; unit_curvature=:orthogonal))
            t, q = target(true), start()
            run = S._brm_svi_run(Xoshiro(5), q, orthogonal, t)
            e = try S._brm_svi_advance!(run, t, orthogonal); nothing catch err err end
            @test e isa S.BRMSVINonfiniteError
            for p in e.terms
                own = t.units[p.unit]
                @test (p.points.plus[own] .+ p.points.minus[own]) ./ 2 ≈ q.M[p.unit]
                @test norm(p.points.plus[own] .- q.M[p.unit]) ≈ sqrt(length(own))
            end
        end

        # The global term has one point, its draw.
        t, q = target(false), start()
        e = try S._brm_svi_check_finite(t, [2.5], NaN, [NaN], [nothing]); nothing catch err err end
        @test e isa S.BRMSVINonfiniteError
        p = only(e.terms)
        @test p.term === :global && keys(p.points) == (:draw,)
        @test p.points.draw == vcat(2.5, fill(-1.0, 6))
        @test p.largest == (; coordinate="g", value=2.5)
        @test occursin("largest coordinate: g = 2.5", sprint(showerror, e))

        # Positive control: the same run without the poison updates everything,
        # so the checks above can tell a guarded iteration from an inert one.
        t, q = target(false), start()
        run = S._brm_svi_run(Xoshiro(5), q, controls, t)
        @test S._brm_svi_advance!(run, t, controls) === :iteration
        @test run.iteration == 1 && length(run.trace) == 1 && run.adam.t == 1
        @test q.μ != before.μ
        @test any(q.M[i] != before.M[i] for i in 1:2)
        @test dense_limit > 0 || any(o !== nothing && o.t == 1 for o in q.unit_opt)
    end
end

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
        Λ=[Matrix{Float64}(I, d, d) for d in 1:5],
        U=[Matrix{Float64}(I, d, d) for d in 1:5])
    # A dense unit's slot of the running average holds its covariance.
    lazy, eager = S._brm_svi_average_start(q), deepcopy(q)
    @test all(isapprox.(lazy.Λ, q.Λ))  # the inverse of an identity precision
    through = zeros(Int, 5)
    # Unit 5 is never visited; other units have different gaps between visits.
    for k in 1:31
        batch = k % 3 == 0 ? [1, 3] : [2, 4]
        S._brm_svi_average_units!(lazy, q, through, batch, k - 1)
        for i in batch
            q.M[i] .+= 0.1randn(rng, i)
            A = randn(rng, i, i)
            q.Λ[i] .+= 0.1 .* (A * A' + I)
        end
        S._brm_svi_average!(eager.M, q.M, k)
        S._brm_svi_average!(eager.Λ, [inv(Symmetric(Λ)) for Λ in q.Λ], k)
    end
    S._brm_svi_average_units!(lazy, q, through, 1:5, 31)
    @test all(isapprox.(lazy.M, eager.M))
    @test all(isapprox.(lazy.Λ, eager.Λ))
    @test through == fill(31, 5)
    @test lazy.M[5] == q.M[5]
    before = deepcopy(lazy)
    S._brm_svi_average_units!(lazy, q, through, 1:5, 31)
    @test lazy == before
    # The result's precision is the inverse of the averaged covariance, with
    # its Cholesky factor.
    S._brm_svi_factor!(lazy)
    for i in 1:5
        @test lazy.Λ[i] ≈ inv(Symmetric(eager.Λ[i]))
        @test LowerTriangular(lazy.U[i]) * LowerTriangular(lazy.U[i])' ≈ lazy.Λ[i]
    end
end
