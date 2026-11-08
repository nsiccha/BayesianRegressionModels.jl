# A dense unit's curvature on an orthogonal design (`unit_curvature`, snag
# brm-svi-fit-hype-47cb328e): pure arithmetic on analytic targets, through the
# production iteration and run loop. Run with
#
#     julia --project=test test/nlme_svi_unit_curvature.jl
module SVIUnitCurvature
using Random, LinearAlgebra, Statistics
struct BRMI end
struct RKBRMI end
struct BRMNLMEModel end
include(joinpath(@__DIR__, "..", "src", "nlme_selection.jl"))
include(joinpath(@__DIR__, "..", "src", "nlme_svi.jl"))
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

@testset "controls without unit_curvature keep the single pair" begin
    @test UC._brm_svi_unit_curvature_mode((; unit_step=0.1)) === :pair
    c = UC._brm_svi_controls(10; batch_size=2, max_iterations=1, lr_global=0.01,
        unit_step=0.1, polish_lr=0.05, window=nothing, patience=1, min_windows=1,
        polish_windows=1)
    @test UC._brm_svi_unit_curvature_mode(c) === :pair
    @test UC._brm_svi_default_unit_step(:orthogonal) == 1.0
    @test UC._brm_svi_default_unit_step(:pair) == 0.1
end
