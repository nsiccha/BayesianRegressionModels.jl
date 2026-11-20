# The re-optimization step of `brm_svi_fit(...; unit_reoptimize = k)` (todo 12maq21):
# pure arithmetic on Gaussian units whose curvature reads a global scale, through the
# production step, iteration and run loop. Run with
#
#     julia --project=test test/nlme_svi_reoptimize_step.jl
module SVIReoptimizeStep
include(joinpath(@__DIR__, "fitting_packages.jl"))
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
include(joinpath(@__DIR__, "svi_scaled_units.jl"))
include("svi_arithmetic_support.jl")
_brm_throw_if_interrupted(::Nothing) = nothing
_brm_update_progress!(::Nothing, message) = nothing
end

using Test, Random, LinearAlgebra, Statistics
const UC = SVIReoptimizeStep

ar1_precision(d, ρ) = inv([ρ^abs(i - j) for i in 1:d, j in 1:d])

# The Newton step lands a quadratic unit's precision on its curvature and its mean on
# its optimum in one visit, within its bounds; the unit's own square-root step lands
# above the curvature by (H - Λ) Λ⁻¹ (H - Λ) / 2.
@testset "re-optimization lands a quadratic unit on its optimum in one step" begin
    rng, d = Xoshiro(21), 6
    Ci = ar1_precision(d, 0.6)
    t = UC.ScaledUnits(Ci, [randn(rng, d)])
    g = [-0.4]
    H = Symmetric(Ci ./ exp(2g[1]) + I)
    target = H \ t.y[1]
    fresh() = UC._brm_svi_params(t, [0.0], UC.MeanFieldGlobal(); global_scale=1.0,
        unit_scale=1.0)
    function start(q, scale)
        q.Λ[1] .= H ./ scale
        q.U[1] .= cholesky(Symmetric(q.Λ[1])).L
        q.M[1] .= target .+ 0.05 .* randn(Xoshiro(22), d)
        q
    end
    # Within the square-root step's trust region and far beyond it (a 30-fold change).
    for scale in (1.6, 1 / 30)
        q, full = start(fresh(), scale), start(fresh(), scale)
        Λ0 = copy(q.Λ[1])
        E = UC._brm_svi_unit_design(rng, d)
        r = UC._brm_svi_unit_design_terms(t, q, 1, g, E, t.plan, nothing, 1:1, 1)
        UC._brm_svi_unit_step_newton!(q, 1, r.gz, r.S)
        UC._brm_svi_unit_step_full!(full, 1, r.gz, r.S, 1.0)
        @test q.Λ[1] ≈ H rtol=1e-10
        @test q.M[1] ≈ target rtol=1e-10
        scale == 1.6 && @test full.Λ[1] ≈ H + (H - Λ0) * (Λ0 \ (H - Λ0)) / 2 rtol=1e-10
    end
    # A curvature read as zero or as a 10,000-fold change moves the precision 100-fold
    # along its direction and no further; the mean moves at most four sds.
    unit() = (; M=[zeros(3)], U=[Matrix(1.0I, 3, 3)], Λ=[Matrix(1.0I, 3, 3)])
    q = unit()
    UC._brm_svi_unit_step_newton!(q, 1, zeros(3), Diagonal([0.0, 1.0, 1e4]))
    @test q.Λ[1] ≈ Diagonal([0.01, 1.0, 100.0])
    q = unit()
    UC._brm_svi_unit_step_newton!(q, 1, [100.0, 0.0, 0.0], Diagonal([1.0, 1.0, 1.0]))
    @test q.M[1] ≈ [4.0, 0.0, 0.0]
    @test_throws ArgumentError UC._brm_svi_unit_step_newton!(q, 1, zeros(3),
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
# left at another draw, a step that does not land on the conditional optimum biases
# the scale's gradient (src/nlme_svi.jl, `_brm_svi_reoptimized_terms`). Measured
# (this seed; research/svi_reoptimize/step_variants.jl has four): the optimum's log σ
# is -0.542 with sd 0.503, the re-optimized copies' -0.552 with 0.510, the square-root
# step's in place (before the Newton step) -0.540 with 0.304.
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
    @test UC._brm_svi_reoptimize_newton(c) === true
    # A checkpoint written before the control has no field, so it resumes the run
    # it was, on the square-root step: the negative control.
    old_c = NamedTuple{filter(!=(:reoptimize_newton), keys(c))}(c)
    @test UC._brm_svi_reoptimize_newton(old_c) === false
    run = UC._brm_svi_run(Xoshiro(24), fresh(), old_c, t)
    UC._brm_svi_fit!(run, t, old_c)
    old = UC._brm_svi_result_q(run)
    sd(q) = exp(q.ρ[1])
    @info "re-optimization step on a weakly pinned scale" opt new.q.μ[1] sd(new.q) old.μ[1] sd(old)
    @test new.converged
    @test abs(new.q.μ[1] - opt.μ) < 0.2opt.s
    @test 0.9 < sd(new.q) / opt.s < 1.1
    @test sd(old) / opt.s < 0.7
end
