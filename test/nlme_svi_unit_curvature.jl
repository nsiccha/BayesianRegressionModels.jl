# A dense unit's curvature on an orthogonal design (`unit_curvature`, snag
# brm-svi-fit-hype-47cb328e): pure arithmetic on analytic targets, through the
# production iteration and run loop. Run with
#
#     julia --project=test test/nlme_svi_unit_curvature.jl
module SVIUnitCurvature
using Random, LinearAlgebra, Statistics
include(joinpath(@__DIR__, "svi_arithmetic_imports.jl"))
struct BRMI end
struct RKBRMI end
struct BRMNLMEModel end
include(joinpath(@__DIR__, "..", "src", "nlme_selection.jl"))
include(joinpath(@__DIR__, "..", "src", "nlme_svi.jl"))
# The iteration reads the sd bound (a stopgap); a target without `sd_prior` has none.
include(joinpath(@__DIR__, "..", "src", "nlme_svi_sd_bound.jl"))
include(joinpath(@__DIR__, "..", "src", "nlme_svi_unit_streaming.jl"))

# Units whose terms are Gaussian in their coordinates: unit i is
# log N(z | 0, σ² C) + log N(y_i | z, I) with σ = exp(g[1]), the one global, under a
# N(0, 1) prior on log σ. C is an AR(1) correlation, so the scale's optimum reads
# the units' widths along C⁻¹'s rough, high-precision directions, as a chain's
# length scale and innovation sd do.
struct ScaledUnits
    globals::Vector{Int}
    units::Vector{Vector{Int}}
    plan::NamedTuple
    plans::Vector{NamedTuple}
    work::Vector{Nothing}
    Ci::Matrix{Float64}
    y::Vector{Vector{Float64}}
end
function ScaledUnits(Ci, y)
    d, N = size(Ci, 1), length(y)
    units = [collect(1+(i-1)*d+1:1+i*d) for i in 1:N]
    plan = (; footprints=[vcat(1, u) for u in units])
    ScaledUnits([1], units, plan, [plan], [nothing], Ci, y)
end
include("svi_arithmetic_support.jl")

_brm_svi_global_term(::ScaledUnits, g) = (-g[1]^2 / 2, [-g[1]])
function _brm_svi_unit_term!(gg, gz, t::ScaledUnits, i, g, z, plan, work)
    d, s2 = length(z), exp(2g[1])
    Cz = t.Ci * z
    quad = dot(z, Cz)
    r = t.y[i] .- z
    fill!(gg, 0.0)
    gg[1] = quad / s2 - d
    gz .= .-Cz ./ s2 .+ r
    -quad / (2s2) - d * g[1] - sum(abs2, r) / 2
end

# Units whose terms saturate: a unit prior on every coordinate, plus rows
# -(A tanh(aⱼ'z / s))² / 2 whose directions aⱼ span a subspace, so a pair's gradient
# difference is not proportional to its draw. The rows are flat away from z = 0, and
# stiff (A / s)² at it: outside span(a) the target's precision is exactly 1.
struct SaturatingUnits
    globals::Vector{Int}
    units::Vector{Vector{Int}}
    plan::NamedTuple
    plans::Vector{NamedTuple}
    work::Vector{Nothing}
    a::Matrix{Float64}
    s::Float64
    A::Float64
end
function SaturatingUnits(a, s, A, N)
    d = size(a, 1)
    units = [collect(1+(i-1)*d+1:1+i*d) for i in 1:N]
    plan = (; footprints=[vcat(1, u) for u in units])
    SaturatingUnits([1], units, plan, [plan], [nothing], a, s, A)
end
_brm_svi_global_term(::SaturatingUnits, g) = (-g[1]^2 / 2, [-g[1]])
function _brm_svi_unit_term!(gg, gz, t::SaturatingUnits, i, g, z, plan, work)
    r = t.a' * z ./ t.s
    h = t.A .* tanh.(r)
    fill!(gg, 0.0)
    gz .= .-z .- t.a * (h .* (t.A / t.s) .* sech.(r) .^ 2)
    -sum(abs2, z) / 2 - sum(abs2, h) / 2
end
_brm_throw_if_interrupted(::Nothing) = nothing
_brm_update_progress!(::Nothing, message) = nothing
end

using Test, Random, LinearAlgebra, Statistics
const UC = SVIUnitCurvature

ar1_precision(d, ρ) = inv([ρ^abs(i - j) for i in 1:d, j in 1:d])

# The profile objective of the scale: every unit at its exact optimum q*(σ), whose
# precision is C⁻¹/σ² + I, and log σ at a point (its factor's own spread is
# negligible at these sizes); the maximizer by golden section.
function profile_scale(Ci, y)
    d = size(Ci, 1)
    function F(s)
        σ2 = exp(2s)
        P = Symmetric(Ci ./ σ2 + I)
        Σ = inv(P)
        total = -s^2 / 2
        for yi in y
            m = P \ yi
            total += -(dot(m, Ci * m) + tr(Ci * Σ)) / (2σ2) - d * s -
                (sum(abs2, yi .- m) + tr(Σ)) / 2 + logdet(Σ) / 2
        end
        total
    end
    a, b = -3.0, 3.0
    φ = (sqrt(5) - 1) / 2
    for _ in 1:200
        c, e = b - φ * (b - a), a + φ * (b - a)
        F(c) > F(e) ? (b = e) : (a = c)
    end
    (a + b) / 2
end

@testset "an orthogonal design: d pairs at radius √d, Haar rotation" begin
    for d in (1, 3, 17, 64)
        E = UC._brm_svi_unit_design(Xoshiro(d), d)
        @test size(E) == (d, d)
        @test E * E' ≈ d * I
        @test all(c -> norm(c) ≈ sqrt(d), eachcol(E))
        @test E == UC._brm_svi_unit_design(Xoshiro(d), d)
    end
    # The rotation is uniform: its first column's coordinates have the moments of
    # a uniform point on the sphere of radius √d, E[x²] = 1 and E[x⁴] = 3d/(d+2).
    rng, d = Xoshiro(5), 6
    x = [UC._brm_svi_unit_design(rng, d)[1, 1] for _ in 1:40_000]
    @test abs(mean(x)) < 0.02
    @test abs(mean(abs2, x) - 1) < 0.03
    @test abs(mean(x .^ 4) - 3d / (d + 2)) < 0.1
end

@testset "orthogonal pairs estimate a quadratic unit's whole curvature exactly" begin
    rng, d = Xoshiro(11), 12
    Ci = ar1_precision(d, 0.9)
    t = UC.ScaledUnits(Ci, [randn(rng, d)])
    q = UC._brm_svi_params(t, [0.0], UC.MeanFieldGlobal(); global_scale=1.0, unit_scale=1.0)
    A = randn(rng, d, d)
    q.Λ[1] .= A * A' + I
    q.U[1] .= cholesky(Symmetric(q.Λ[1])).L
    q.M[1] .= randn(rng, d)
    g = [0.3]
    E = UC._brm_svi_unit_design(rng, d)
    r = UC._brm_svi_unit_design_terms(t, q, 1, g, E, t.plan, nothing, 1:1, 1)
    @test r.problem === nothing
    H = Ci ./ exp(2g[1]) + I                        # -∇² of the unit's term
    U = LowerTriangular(q.U[1])
    @test r.S ≈ U \ (U \ H)' rtol=1e-10             # whitened: U⁻¹ H U⁻ᵀ
    Σ = inv(Symmetric(q.Λ[1]))
    m = q.M[1]
    @test r.gz ≈ -(Ci * m) ./ exp(2g[1]) .+ (t.y[1] .- m) rtol=1e-10
    quad = dot(m, Ci * m) + tr(Ci * Σ)
    @test r.value ≈ -quad / (2exp(2g[1])) - d * g[1] - (sum(abs2, t.y[1] .- m) + tr(Σ)) / 2
    @test r.gg[1] / 2 ≈ quad / exp(2g[1]) - d rtol=1e-10
end

@testset "the full step reaches a quadratic unit's optimum within the trust regions" begin
    rng, d = Xoshiro(12), 30
    Ci = ar1_precision(d, 0.95)
    t = UC.ScaledUnits(Ci, [3randn(rng, d)])
    g = [0.0]
    H = Ci + I
    target = H \ t.y[1]
    q = UC._brm_svi_params(t, [0.0], UC.MeanFieldGlobal(); global_scale=1.0, unit_scale=1.0)
    for visit in 1:40
        E = UC._brm_svi_unit_design(rng, d)
        r = UC._brm_svi_unit_design_terms(t, q, 1, g, E, t.plan, nothing, 1:1, 1)
        UC._brm_svi_unit_step_full!(q, 1, r.gz, r.S, 1.0)
        # Each visit changes the precision at most 2.5-fold along any direction.
        @test isposdef(Symmetric(q.Λ[1]))
    end
    @test q.Λ[1] ≈ H rtol=1e-8
    @test q.M[1] ≈ target rtol=1e-8
end

# The snag's mechanism, exactly Gaussian: units of 48 coordinates whose scale's
# optimum reads their widths. The production loop under the default schedule
# (7,100 iterations, each unit visited about 890 times); the scale against the
# profile optimum. Measured: log σ 0.0345 against 0.0334 with the units' factors
# 0.002 nats from q*(σ) in all; the single pair at 0.1 (BRM 0.2.242) ends at log
# σ 0.507, its factors 100.6 nats from q*(σ): wide factors make the prior's
# scale look larger, as they made a chain's length scale look short.
@testset "a scale that reads dense units' widths reaches its optimum" begin
    rng, d, N = Xoshiro(13), 48, 32
    Ci = ar1_precision(d, 0.95)
    C = inv(Ci)
    L = cholesky(Symmetric(C)).L
    y = [L * randn(rng, d) .+ randn(rng, d) for _ in 1:N]
    t = UC.ScaledUnits(Ci, y)
    s_opt = profile_scale(Ci, y)
    fits = map((:orthogonal, :pair)) do curvature
        q = UC._brm_svi_params(t, [0.0], UC.MeanFieldGlobal(); global_scale=0.1, unit_scale=1.0)
        fit = UC.arithmetic_fit(Xoshiro(14), t, q; batch_size=4, max_iterations=40_000,
            lr_global=0.01, unit_step=UC._brm_svi_default_unit_step(curvature),
            polish_lr=0.05, window=nothing, patience=10, min_windows=40, polish_windows=20,
            unit_curvature=curvature)
        # Exact KL of each unit's factor from its optimum q*(σ) at the fitted scale.
        σ2 = exp(2fit.q.μ[1])
        P = Symmetric(Ci ./ σ2 + I)
        kl = map(1:N) do i
            Λ, m = Symmetric(fit.q.Λ[i]), fit.q.M[i]
            δ = m .- P \ y[i]
            (tr(P / Λ) + dot(δ, P * δ) - d - logdet(P) + logdet(Λ)) / 2
        end
        (; curvature, fit, s=fit.q.μ[1], kl=sum(kl))
    end
    o, p = fits
    @info "unit_curvature on 32 units of 48 coordinates" s_opt o.s p.s o.kl p.kl o.fit.iterations p.fit.iterations
    @test o.fit.converged
    @test abs(o.s - s_opt) < 0.01
    @test o.kl < 0.05
    @test o.fit.unit_gradient_calls == 2d * sum(o.fit.sampling.visits)
    @test p.fit.unit_gradient_calls == 2sum(p.fit.sampling.visits)
    # Control: the single pair (BRM 0.2.242's step) leaves the same units unsettled
    # and the scale off, so this testset fails on that step.
    @test p.kl > 10
    @test abs(p.s - s_opt) > 0.2
end

# On a saturating term the design's estimate is mostly noise with nearly symmetric
# eigenvalues until the stiff directions are resolved. The unit's precision off
# span(a) is exactly 1. Measured over 8 seeds, 80 visits: the whole estimate falls
# to 3e-4 and stays there with a top eigenvalue of 75; the positive part never
# leaves 1 and reaches the rows' stiffness, 1.65e8.
function saturating_rows(d, stiff, rows)
    G = randn(Xoshiro(8), stiff, rows)
    Matrix(qr(randn(Xoshiro(7), d, stiff)).Q) * (G ./ sqrt.(sum(abs2, G; dims=1)))
end
function saturating_visits(t, positive; visits=80, seed=1)
    rng, d = Xoshiro(seed), size(t.a, 1)
    q = UC._brm_svi_params(t, [0.0], UC.MeanFieldGlobal(); global_scale=1.0, unit_scale=1.0)
    lowest, ratios = Inf, Float64[]
    for visit in 1:visits
        E = UC._brm_svi_unit_design(rng, d)
        r = UC._brm_svi_unit_design_terms(t, q, 1, [0.0], E, t.plan, nothing, 1:1, 1)
        ev = eigvals(Symmetric((r.S .+ r.S') ./ 2))
        visit <= 10 && push!(ratios, abs(first(ev)) / last(ev))
        UC._brm_svi_unit_step_full!(q, 1, r.gz, r.S, 1.0; positive)
        lowest = min(lowest, eigvals(Symmetric(q.Λ[1]))[1])
    end
    ev = eigvals(Symmetric(q.Λ[1]))
    (; lowest, smallest=first(ev), largest=last(ev), ratio=median(ratios), mean=norm(q.M[1]))
end

@testset "a saturating term's noise does not collapse a dense unit's factor" begin
    t = UC.SaturatingUnits(saturating_rows(17, 5, 64), 0.01, 30.0, 1)
    for seed in 1:4
        whole = saturating_visits(t, false; seed)
        positive = saturating_visits(t, true; seed)
        # Control: the whole estimate shows the defect, so this testset fails on that
        # step. Its eigenvalues are nearly symmetric, which is the mechanism.
        @test whole.ratio > 0.8
        @test whole.lowest < 0.01
        @test whole.largest < 1e3
        # The target's precision off span(a) is 1, and the positive part keeps it.
        @test positive.lowest > 0.9
        @test positive.smallest ≈ 1 atol=0.01
        @test positive.largest > 1e7
        @test positive.mean < 1e-6
    end
end

@testset "the production step takes the positive part, and old controls keep the whole estimate" begin
    d, N = 17, 8
    t = UC.SaturatingUnits(saturating_rows(d, 5, 64), 0.01, 30.0, N)
    kwargs = (; batch_size=N, max_iterations=60, lr_global=0.01, unit_step=1.0, polish_lr=0.05,
        window=nothing, patience=10, min_windows=40, polish_windows=20, unit_curvature=:orthogonal)
    smallest(q) = minimum(i -> eigvals(Symmetric(q.Λ[i]))[1], 1:N)
    fresh() = UC._brm_svi_params(t, [0.0], UC.MeanFieldGlobal(); global_scale=0.1, unit_scale=1.0)
    @test smallest(UC.arithmetic_fit(Xoshiro(15), t, fresh(); kwargs...).q) > 0.9
    # A checkpoint written before the control has no field, so it resumes the run it was.
    c = UC._brm_svi_controls(N; kwargs...)
    @test UC._brm_svi_unit_positive_part(c) === true
    old = NamedTuple{filter(!=(:unit_positive_part), keys(c))}(c)
    @test UC._brm_svi_unit_positive_part(old) === false
    run = UC._brm_svi_run(Xoshiro(15), fresh(), old, t)
    UC._brm_svi_fit!(run, t, old)
    @test smallest(run.q) < 0.1
end

@testset "a direction whose curvature lies below the factor's still widens" begin
    d = 17
    # Exact on a Gaussian, beside a direction a million times stiffer than the factor.
    h = ones(d); h[1], h[2] = 0.01, 1e6
    q = (; M=[zeros(d)], U=[Matrix(1.0I, d, d)], Λ=[Matrix(1.0I, d, d)])
    rng = Xoshiro(16)
    for visit in 1:40
        E = UC._brm_svi_unit_design(rng, d)
        W = Diagonal(h) * (LowerTriangular(q.U[1])' \ E)
        W = LowerTriangular(q.U[1]) \ W
        UC._brm_svi_unit_step_full!(q, 1, zeros(d), (W * E' .+ E * W') ./ 2d, 1.0)
    end
    @test diag(q.Λ[1]) ≈ h rtol=1e-6
    # A negative estimate widens at the bound and no faster.
    q = (; M=[zeros(3)], U=[Matrix(1.0I, 3, 3)], Λ=[Matrix(1.0I, 3, 3)])
    UC._brm_svi_unit_step_full!(q, 1, zeros(3), Diagonal([-5.0, 1.0, 1.0]), 1.0)
    @test q.Λ[1] ≈ Diagonal([0.5, 1.0, 1.0])
end

@testset "a matrix's positive part" begin
    S = Symmetric([2.0 0.5; 0.5 1.0])
    # No negative eigenvalue: the matrix itself, so a quadratic unit's step is unchanged.
    @test UC._brm_svi_positive_part(S) === S
    F = eigen(Symmetric([1.0 2.0; 2.0 1.0]))
    P = UC._brm_svi_positive_part(Symmetric([1.0 2.0; 2.0 1.0]))
    @test P ≈ F.vectors[:, 2] * F.values[2] * F.vectors[:, 2]'
    @test minimum(eigvals(P)) >= -1e-12
    q = (; M=[zeros(2)], U=[Matrix(1.0I, 2, 2)], Λ=[Matrix(1.0I, 2, 2)])
    @test_throws ArgumentError UC._brm_svi_unit_step_full!(q, 1, zeros(2), [NaN 0.0; 0.0 1.0], 1.0)
end

@testset "controls without unit_curvature keep the single pair" begin
    @test UC._brm_svi_unit_curvature_mode((; unit_step=0.1)) === :pair
    c = UC._brm_svi_controls(10; batch_size=2, max_iterations=1, lr_global=0.01,
        unit_step=0.1, polish_lr=0.05, window=nothing, patience=1, min_windows=1,
        polish_windows=1)
    @test UC._brm_svi_unit_curvature_mode(c) === :pair
    @test UC._brm_svi_default_unit_step(:orthogonal) == 1.0
    @test UC._brm_svi_default_unit_step(:pair) == 0.1
end

# ── Re-optimization's step (`unit_reoptimize`, todo 12maq21) ─────────────────

# The step lands a quadratic unit's precision on its curvature and its mean on its
# optimum in one visit, wherever the trust region leaves β = 1; the unit's own
# square-root step lands above the curvature by (H - Λ) Λ⁻¹ (H - Λ) / 2.
@testset "re-optimization lands a quadratic unit on its optimum in one step" begin
    rng, d = Xoshiro(21), 6
    Ci = ar1_precision(d, 0.6)
    t = UC.ScaledUnits(Ci, [randn(rng, d)])
    g = [-0.4]
    H = Symmetric(Ci ./ exp(2g[1]) + I)
    target = H \ t.y[1]
    start(q) = (q.Λ[1] .= H ./ 1.6; q.U[1] .= cholesky(Symmetric(q.Λ[1])).L;
        q.M[1] .= target .+ 0.1 .* randn(Xoshiro(22), d); q)
    fresh() = UC._brm_svi_params(t, [0.0], UC.MeanFieldGlobal(); global_scale=1.0,
        unit_scale=1.0)
    q, full = start(fresh()), start(fresh())
    Λ0 = copy(q.Λ[1])
    E = UC._brm_svi_unit_design(rng, d)
    r = UC._brm_svi_unit_design_terms(t, q, 1, g, E, t.plan, nothing, 1:1, 1)
    UC._brm_svi_unit_step_conditional!(q, 1, r.gz, r.S)
    UC._brm_svi_unit_step_full!(full, 1, r.gz, r.S, 1.0)
    @test q.Λ[1] ≈ H rtol=1e-10
    @test q.M[1] ≈ target rtol=1e-10
    @test full.Λ[1] ≈ H + (H - Λ0) * (Λ0 \ (H - Λ0)) / 2 rtol=1e-10
    # Within the trust region a direction lands on its estimate, but never below
    # half the precision; beyond it every direction takes β = 1 / r of its change,
    # at most twofold, as the square-root step's trust region scales them.
    unit() = (; M=[zeros(3)], U=[Matrix(1.0I, 3, 3)], Λ=[Matrix(1.0I, 3, 3)])
    q = unit()
    UC._brm_svi_unit_step_conditional!(q, 1, zeros(3), Diagonal([0.0, 1.0, 1.5]))
    @test q.Λ[1] ≈ Diagonal([0.5, 1.0, 1.5])
    q = unit()
    UC._brm_svi_unit_step_conditional!(q, 1, zeros(3), Diagonal([0.0, 1.0, 9.0]))
    @test q.Λ[1] ≈ Diagonal([0.875, 1.0, 2.0])
    @test_throws ArgumentError UC._brm_svi_unit_step_conditional!(q, 1, zeros(3),
        [NaN 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
end

# The exact optimum of a Gaussian q(ℓ) over the scale's log when each unit sits at
# its optimum given every draw of ℓ: units Gaussian, so the profile is the marginal
# log N(y_i | 0, σ² C + I), and the ELBO is E_q[log p(y, ℓ)] + log s, here by
# 40-node Gauss-Hermite quadrature and coordinate golden sections.
function scale_elbo_optimum(Ci, y)
    C = inv(Ci)
    marginal(ℓ) = -ℓ^2 / 2 + sum(yi -> begin
            K = Symmetric(exp(2ℓ) .* C + I)
            -(logdet(K) + dot(yi, K \ yi)) / 2
        end, y)
    J = SymTridiagonal(zeros(40), sqrt.(1.0:39.0))
    F = eigen(J)
    nodes, weights = F.values, F.vectors[1, :] .^ 2
    elbo(μ, s) = sum(weights .* marginal.(μ .+ s .* nodes)) + log(s)
    golden(f, a, b) = begin
        φ = (sqrt(5) - 1) / 2
        for _ in 1:80
            c, e = b - φ * (b - a), a + φ * (b - a)
            f(c) > f(e) ? (b = e) : (a = c)
        end
        (a + b) / 2
    end
    μ, s = 0.0, 0.5
    for _ in 1:30
        μ = golden(m -> elbo(m, s), -6.0, 3.0)
        s = exp(golden(l -> elbo(μ, exp(l)), log(1e-3), log(5.0)))
    end
    (; μ, s)
end

# One row per unit beside a scale the units' spread pins only weakly: unit i is
# z_i ~ N(0, σ²), y_i ~ N(z_i, 1), so the unit's curvature 1/σ² + 1 moves with every
# draw of σ and q(log σ) is wide. Re-optimizing from the factor the unit's last visit
# left at another draw, the square-root step lands every unit too narrow and the
# scale's gradient reads too little of its spread, the joint mode's profile, which
# falls towards σ = 0. Measured (seed below): the old step's log σ ended at MEASURED,
# the first-order step's at MEASURED against the optimum's MEASURED.
@testset "re-optimized units leave a weakly pinned scale at its optimum" begin
    rng, N = Xoshiro(23), 16
    y = [[0.4randn(rng) + randn(rng)] for _ in 1:N]
    Ci = fill(1.0, 1, 1)
    t = UC.ScaledUnits(Ci, y)
    opt = scale_elbo_optimum(Ci, y)
    kwargs = (; batch_size=4, max_iterations=40_000, lr_global=0.01, unit_step=1.0,
        polish_lr=0.05, window=nothing, patience=10, min_windows=40, polish_windows=20,
        unit_curvature=:orthogonal, unit_reoptimize=1)
    fresh() = UC._brm_svi_params(t, [0.0], UC.MeanFieldGlobal(); global_scale=0.1,
        unit_scale=1.0)
    new = UC.arithmetic_fit(Xoshiro(24), t, fresh(); kwargs...)
    c = UC._brm_svi_controls(N; kwargs...)
    @test UC._brm_svi_reoptimize_first_order(c) === true
    # A checkpoint written before the control has no field, so it resumes the run
    # it was, on the square-root step.
    old_c = NamedTuple{filter(!=(:reoptimize_first_order), keys(c))}(c)
    @test UC._brm_svi_reoptimize_first_order(old_c) === false
    run = UC._brm_svi_run(Xoshiro(24), fresh(), old_c, t)
    UC._brm_svi_fit!(run, t, old_c)
    old = UC._brm_svi_result_q(run)
    stock = UC.arithmetic_fit(Xoshiro(24), t, fresh(); merge(kwargs, (; unit_reoptimize=0))...)
    sd(q) = exp(q.ρ[1])
    @info "re-optimization step on a weakly pinned scale" opt new.q.μ[1] sd(new.q) old.μ[1] sd(old) stock.q.μ[1] sd(stock.q) new.iterations run.iteration stock.iterations
    @test abs(new.q.μ[1] - opt.μ) < 0.2opt.s
    @test 0.8 < sd(new.q) / opt.s < 1.2
end
