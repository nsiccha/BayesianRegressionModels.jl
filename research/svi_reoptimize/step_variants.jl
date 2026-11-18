# Variants of the re-optimization step (`_brm_svi_unit_step_newton!`) on the
# arithmetic scale toy of test/nlme_svi_reoptimize_step.jl (todo 12maq21): one row
# per unit, z_i ~ N(0, σ²), y_i ~ N(z_i, 1), N(0, 1) on log σ, 16 units, batch 4,
# against the exact structured optimum of q(log σ). Each variant redefines the step
# inside a module that includes BRM's SVI sources, then fits through the production
# run loop. The fit seeds and data seed vary with `seeds`.
#
#     julia --project=test research/svi_reoptimize/step_variants.jl [nseeds]
module StepVariants
using Random, LinearAlgebra, Statistics
const TEST = joinpath(dirname(dirname(@__DIR__)), "test")
include(joinpath(TEST, "svi_arithmetic_imports.jl"))
struct BRMI end
struct RKBRMI end
struct BRMNLMEModel end
include(joinpath(TEST, "..", "src", "nlme_selection.jl"))
include(joinpath(TEST, "..", "src", "nlme_svi.jl"))
include(joinpath(TEST, "..", "src", "nlme_svi_sd_bound.jl"))
include(joinpath(TEST, "..", "src", "nlme_svi_unit_streaming.jl"))
include(joinpath(TEST, "svi_scaled_units.jl"))
include(joinpath(TEST, "svi_arithmetic_support.jl"))
_brm_throw_if_interrupted(::Nothing) = nothing
_brm_update_progress!(::Nothing, message) = nothing

# The step with each direction's precision factor bounded to [FLOOR, CAP] (`CAP = 2`:
# a uniform β = min(1, 1 / r) instead, as the unit's own trust region scales it), and
# the mean's step bounded to MEANTRUST sds of the factor before it (MEANNEW: of the
# factor after it). BRM's step is FLOOR = 1/100, CAP = 100, MEANTRUST = 4 (old sds).
const CAP, FLOOR, MEANTRUST, MEANNEW = Ref(2.0), Ref(0.5), Ref(1.0), Ref(false)
function _brm_svi_unit_step_newton!(q, i, gz, S; positive=true)
    isempty(q.M[i]) && return q
    U = LowerTriangular(q.U[i])
    Ŝ = Symmetric((S .+ S') ./ 2)
    F = eigen(Symmetric((positive ? _brm_svi_positive_part(Ŝ) : Ŝ) - I))
    β = 1.0
    if CAP[] == 2.0
        r = maximum(abs, F.values)
        β = r > 0 ? min(1.0, 1 / r) : 1.0
    end
    f = clamp.(1 .+ β .* F.values, FLOOR[], CAP[])
    R = qr(Matrix((U * F.vectors * Diagonal(sqrt.(f)))')).R
    L = LowerTriangular(R' * Diagonal(sign.(diag(R))))
    Δ = β .* (L' \ (L \ gz))
    Δ .*= min(1.0, MEANTRUST[] / norm((MEANNEW[] ? L : U)' * Δ))
    q.U[i] .= L
    q.Λ[i] .= L * L'
    q.M[i] .+= Δ
    q
end
end

using Random, LinearAlgebra, Statistics, Printf
const V = StepVariants

function scale_elbo_optimum(y)
    marginal(ℓ) = -ℓ^2 / 2 + sum(yi -> -(log(exp(2ℓ) + 1) + yi[1]^2 / (exp(2ℓ) + 1)) / 2, y)
    J = SymTridiagonal(zeros(40), sqrt.(1.0:39.0))
    F = eigen(J)
    nodes, weights = F.values, F.vectors[1, :] .^ 2
    elbo(μ, s) = sum(weights .* marginal.(μ .+ s .* nodes)) + log(s)
    golden(f, a, b) = (φ = (sqrt(5) - 1) / 2; for _ in 1:80
        c, e = b - φ * (b - a), a + φ * (b - a); f(c) > f(e) ? (b = e) : (a = c) end; (a + b) / 2)
    μ, s = 0.0, 0.5
    for _ in 1:30
        μ = golden(m -> elbo(m, s), -6.0, 3.0)
        s = exp(golden(l -> elbo(μ, exp(l)), log(1e-3), log(5.0)))
    end
    (; μ, s, elbo)
end

nseeds = isempty(ARGS) ? 3 : parse(Int, ARGS[1])
variants = [("square root (before the Newton step), k = 1", 1, true, 2.0, 0.5, 1.0, false),
            ("square root, k = 3", 3, true, 2.0, 0.5, 1.0, false),
            ("first order, trust 2x / floor 1/2", 1, false, 2.0, 0.5, 1.0, false),
            ("[1/100, 100], mean trust 1 new sd", 1, false, 100.0, 0.01, 1.0, true),
            ("[1/100, 100], mean trust Inf", 1, false, 100.0, 0.01, Inf, false),
            ("Newton: [1/100, 100], mean trust 4", 1, false, 100.0, 0.01, 4.0, false),
            ("stock, k = 0", 0, false, 2.0, 0.5, 1.0, false)]
for seed in 1:nseeds
    rng = Xoshiro(22 + seed)
    y = [[0.4randn(rng) + randn(rng)] for _ in 1:16]
    t = V.ScaledUnits(fill(1.0, 1, 1), y)
    opt = scale_elbo_optimum(y)
    @printf("data seed %d: optimum log σ mean %.3f sd %.3f\n", seed, opt.μ, opt.s)
    for (label, k, old, cap, floor, meantrust, meannew) in variants
        V.CAP[], V.FLOOR[], V.MEANTRUST[], V.MEANNEW[] = cap, floor, meantrust, meannew
        kwargs = (; batch_size=4, max_iterations=40_000, lr_global=0.01, unit_step=1.0,
            polish_lr=0.05, window=nothing, patience=10, min_windows=40, polish_windows=20,
            unit_curvature=:orthogonal, unit_reoptimize=k)
        c = V._brm_svi_controls(16; kwargs...)
        old && (c = NamedTuple{filter(!=(:reoptimize_newton), keys(c))}(c))
        q = V._brm_svi_params(t, [0.0], V.MeanFieldGlobal(); global_scale=0.1, unit_scale=1.0)
        run = V._brm_svi_run(Xoshiro(23 + seed), q, c, t)
        V._brm_svi_fit!(run, t, c)
        r = V._brm_svi_result_q(run)
        s = exp(r.ρ[1])
        @printf("  %-42s mean %7.3f sd %.3f | err %6.2f opt sds, sd ratio %.2f, ELBO gap %.3f, %d it\n",
            label, r.μ[1], s, (r.μ[1] - opt.μ) / opt.s, s / opt.s,
            opt.elbo(opt.μ, opt.s) - opt.elbo(r.μ[1], s), run.iteration)
        flush(stdout)
    end
end
