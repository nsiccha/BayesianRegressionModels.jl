module SVIUnitArithmetic
using Random, LinearAlgebra, Statistics
struct BRMI end
struct RKBRMI end
struct BRMNLMEModel end
include(joinpath(@__DIR__, "..", "src", "nlme_selection.jl"))
include(joinpath(@__DIR__, "..", "src", "nlme_svi.jl"))
include(joinpath(@__DIR__, "..", "src", "nlme_svi_unit_streaming.jl"))
struct GaussianUnits
    globals::Vector{Int}
    units::Vector{Vector{Int}}
    plan::NamedTuple
    plans::Vector{NamedTuple}
    work::Vector{Nothing}
end
GaussianUnits(globals, units, plan) =
    GaussianUnits(globals, units, plan, [plan], [nothing])
include("svi_arithmetic_support.jl")

_brm_svi_global_term(::GaussianUnits, g) = (-sum(abs2, g) / 2, -g)
function _brm_svi_unit_term!(gg, gz, t::GaussianUnits, i, g, z, plan, work)
    fill!(gg, 0.0)
    gz .= -(z .- i) .* (i + 1)
    -(i + 1) * sum(abs2, z .- i) / 2
end
# One exact Gaussian unit with precisions `P` and centre `c`.
struct TightUnit
    globals::Vector{Int}
    units::Vector{Vector{Int}}
    plan::NamedTuple
    plans::Vector{NamedTuple}
    work::Vector{Nothing}
    P::Vector{Float64}
    c::Vector{Float64}
end
function TightUnit(P, c)
    plan = (; footprints=[collect(1:length(P) + 1)])
    TightUnit([1], [collect(2:length(P) + 1)], plan, [plan], [nothing], P, c)
end
_brm_svi_global_term(::TightUnit, g) = (-sum(abs2, g) / 2, -g)
function _brm_svi_unit_term!(gg, gz, t::TightUnit, i, g, z, plan, work)
    fill!(gg, 0.0)
    gz .= -t.P .* (z .- t.c)
    -sum(t.P .* (z .- t.c) .^ 2) / 2
end
# One exact Gaussian unit with precision matrix `H` and centre `c`.
struct CorrelatedUnit
    globals::Vector{Int}
    units::Vector{Vector{Int}}
    plan::NamedTuple
    plans::Vector{NamedTuple}
    work::Vector{Nothing}
    H::Matrix{Float64}
    c::Vector{Float64}
end
function CorrelatedUnit(H, c)
    plan = (; footprints=[collect(1:length(c) + 1)])
    CorrelatedUnit([1], [collect(2:length(c) + 1)], plan, [plan], [nothing], Matrix(H), c)
end
_brm_svi_global_term(::CorrelatedUnit, g) = (-sum(abs2, g) / 2, -g)
function _brm_svi_unit_term!(gg, gz, t::CorrelatedUnit, i, g, z, plan, work)
    fill!(gg, 0.0)
    r = z .- t.c
    gz .= -(t.H * r)
    -dot(r, t.H, r) / 2
end
_brm_throw_if_interrupted(::Nothing) = nothing
_brm_update_progress!(::Nothing, message) = nothing
end

using Test, Random, LinearAlgebra, Serialization
const SU = SVIUnitArithmetic

# Fits `SU.TightUnit` for `iterations` at a structured step: the factor's KL from the target
# and its largest marginal sd in target sds.
function tight_unit_fit(t, rank, step; iterations=5000, dense_step=1.0, rank_step=1.0)
    q = SU._brm_svi_params(t, [0.0], SU.MeanFieldGlobal();
        global_scale=0.1, unit_scale=1.0, unit_dense_limit=0, unit_rank=rank)
    c = SU._brm_svi_controls(1; batch_size=1, max_iterations=iterations, lr_global=0.01,
        unit_step=dense_step, polish_lr=0.05, window=100, patience=10, min_windows=1000,
        polish_windows=20, structured_unit_step=step, structured_rank_step=rank_step)
    run = SU._brm_svi_run(Xoshiro(1), q, c, t)
    worst = 0.0
    for _ in 1:iterations
        SU._brm_svi_advance!(run, t, c)
        f = q.Λ[1].factor
        worst = max(worst, maximum(sqrt.(exp.(2 .* f.ρ) .+ vec(sum(abs2, f.Β; dims=2))) .* sqrt.(t.P)))
    end
    f = q.Λ[1].factor
    C = Matrix(Diagonal(exp.(2 .* f.ρ)) + f.Β * f.Β')
    m = q.M[1] .- t.c
    kl = (tr(Diagonal(t.P) * C) + dot(m, t.P .* m) - length(m) - sum(log, t.P) - logdet(C)) / 2
    (; kl, worst, last=maximum(sqrt.(diag(C)) .* sqrt.(t.P)))
end

@testset "empty local factors after separator promotion" begin
    t = SU.GaussianUnits([1], [Int[], [2]], (; footprints=[Int[], [2]]))
    q = SU._brm_svi_params(t, [0.0], SU.MeanFieldGlobal();
        global_scale=0.1, unit_scale=1.0)
    fit = SU.arithmetic_fit(Xoshiro(52), t, q; batch_size=2, max_iterations=3,
        lr_global=0.01, unit_step=0.05, polish_lr=0.01, window=10,
        patience=2, min_windows=3, polish_windows=2)
    @test isempty(fit.q.M[1])
    @test size(fit.q.U[1]) == (0, 0)
    @test all(isfinite, fit.elbo)
end

@testset "bounded local Gaussian operators and exact path gradient" begin
    rng = Xoshiro(51)
    for d in (1, 5), rank in (0, 1, 2)
        f = SU._BRMSVILowRankUnit(0.2randn(rng, d), 0.3randn(rng, d, rank))
        precision, root = SU._BRMSVIUnitPrecision(f), SU._BRMSVIUnitRoot(f)
        C = Matrix(Diagonal(exp.(2f.ρ)) + f.Β * f.Β')
        P = inv(C)
        @test Matrix(precision) ≈ P
        U = Matrix(root)
        @test U * U' ≈ P
        x = randn(rng, d)
        @test precision * x ≈ P * x
        @test root' * x ≈ U' * x
        @test diag(precision) ≈ diag(P)
        @test diag(root) ≈ diag(U)
        @test SU._brm_svi_unit_logdet_root(root) ≈ -logdet(C) / 2
        # Exact integration of quadratic functions of standard normal noise.
        n = d + rank
        noise = sqrt(n) .* Matrix{Float64}(I, n, n)
        offsets = [SU._brm_svi_unit_offset(root, e) for e in eachcol(noise)]
        @test sum(δ * δ' for δ in offsets) / n ≈ C
        H = randn(rng, d, d); H = H * H' + I
        μ, target = randn(rng, d), randn(rng, d)
        gradients = map(zip(eachcol(noise), offsets)) do (ε, δ)
            SU._brm_svi_lowrank_gradient(f, -H * (μ - target), -H * δ, ε)
        end
        expected = vcat(-H * (μ - target),
            exp.(2f.ρ) .* diag(P - H), vec((P - H) * f.Β))
        @test sum(gradients) / n ≈ expected
        for (ε, δ) in zip(eachcol(noise), offsets)
            @test SU._brm_svi_lowrank_gradient(f, zeros(d), -P * δ, ε) ≈
                zeros(d * (2 + rank)) atol=2e-12
        end
    end
end

function lowrank_unit_allocation(d, rank)
    precision, root, opt = SU._brm_svi_unit_factors(d, 1.0, 0, rank)
    f, m = root.factor, zeros(d)
    ε = ones(d + rank)
    gz, gd = fill(0.1, d), fill(-0.2, d)
    SU._brm_svi_lowrank_step!(m, f, opt, gz, gd, ε, 0.01)
    bytes = @allocated SU._brm_svi_lowrank_step!(m, f, opt, gz, gd, ε, 0.01)
    io = IOBuffer(); serialize(io, (; m, f.ρ, f.Β))
    (; bytes, storage=Base.summarysize((precision, root, opt, m)), checkpoint=position(io))
end

@testset "low-rank local carry preserves density under reordering" begin
    t = (; globals=[1], units=[[2, 3, 4, 5]])
    q = SU._brm_svi_params(t, [0.0], SU.MeanFieldGlobal();
        global_scale=1.0, unit_scale=0.7, unit_dense_limit=0, unit_rank=2)
    q.M[1] .= [0.1, 0.2, 0.3, 0.4]
    f = q.Λ[1].factor
    f.ρ .= [-0.2, -0.3, 0.1, 0.5]
    f.Β .= [0.1 0.2; -0.3 0.1; 0.4 0.3; -0.2 0.5]
    old = SU._brm_svi_saved_unit([:a, :b, :c, :d], q.M[1], q.Λ[1], q.U[1])
    io = IOBuffer(); serialize(io, old); seekstart(io); old = deserialize(io)
    @test keys(old) == (:keys, :M, :family, :ρ, :Β)
    @test SU._brm_svi_state_format(Dict(:one => old)) == 2
    perm = [4, 2, 1, 3]
    restored = SU._brm_svi_params(t, [0.0], SU.MeanFieldGlobal();
        global_scale=1.0, unit_scale=1.0)
    SU._brm_svi_restore_unit!(restored, 1, old, perm)
    @test restored.U[1].factor === restored.Λ[1].factor
    @test restored.unit_opt[1].t == 0
    P = inv(Matrix(Diagonal(exp.(2f.ρ)) + f.Β * f.Β'))
    δ = [0.2, -0.4, 0.6, -0.8]
    @test restored.Λ[1] * δ[perm] ≈ (P * δ)[perm]
    @test SU._brm_svi_unit_logdet_root(restored.U[1]) ≈ logdet(P) / 2
    @test restored.M[1] == q.M[1][perm]
    f.ρ[1] += 1
    @test old.ρ[1] == -0.2

    # Legacy dense entries retain their exact family even when new units use
    # a smaller dense bound; no implicit projection of a previous posterior.
    dense = (; keys=[:a, :b, :c, :d], M=ones(4), Λ=2Matrix{Float64}(I, 4, 4),
        U=sqrt(2)Matrix{Float64}(I, 4, 4))
    SU._brm_svi_restore_unit!(restored, 1, dense, perm)
    @test restored.Λ[1] == dense.Λ
    @test restored.U[1] ≈ dense.U
    @test SU._brm_svi_state_format(Dict(:one => dense)) == 1
end

@testset "large local Gaussian storage and update work" begin
    lowrank_unit_allocation(20, 2)
    small, large = lowrank_unit_allocation(2000, 3), lowrank_unit_allocation(4000, 3)
    @test large.bytes < 2.2small.bytes
    @test large.storage < 2.1small.storage
    @test large.checkpoint < 2.1small.checkpoint
    @test large.bytes < 8_000_000
    Λ, U, opt = SU._brm_svi_unit_factors(8, 1.0, 64, 8)
    @test Λ == U == Matrix{Float64}(I, 8, 8)
    @test opt === nothing
end

@testset "low-rank local optimizer on an independent Gaussian oracle" begin
    rng = Xoshiro(511)
    d, rank = 5, 2
    B = [0.3 0.0; 0.5 0.1; -0.3 0.4; 0.2 -0.2; 0.0 0.3]
    C = Matrix(Diagonal([0.8, 0.5, 0.7, 1.1, 0.6]) + B * B')
    P, target = inv(C), [0.4, -0.8, 0.1, 0.7, -0.4]
    _, U, opt = SU._brm_svi_unit_factors(d, 1.0, 0, rank)
    f, m = U.factor, zeros(d)
    for k in 1:2000
        ε = randn(rng, d + rank)
        δ = SU._brm_svi_unit_offset(U, ε)
        SU._brm_svi_lowrank_step!(m, f, opt, -P * (m - target), -P * δ,
            ε, 0.04 / sqrt(1 + k / 200))
    end
    covariance = Matrix(Diagonal(exp.(2f.ρ)) + f.Β * f.Β')
    kl = (tr(P * covariance) + dot(m - target, P * (m - target)) - d +
        logdet(C) - logdet(covariance)) / 2
    @test norm(m - target) < 0.03
    @test kl < 0.03
    @test norm(covariance - C) / norm(C) < 0.15
end

@testset "a structured unit's step is its own, and does not follow unit_curvature" begin
    # `unit_step` defaulted to 1 under `:orthogonal` for every family, though only dense
    # units have an orthogonal design and a whole step to take (snag
    # svi-nonfinite-er-0923561d).
    steps(unit, structured) = (; unit_step=unit, structured_unit_step=structured,
        structured_rank_step=0.1)
    @test SU._brm_svi_unit_steps(nothing, :orthogonal) == steps(1.0, 0.1)
    @test SU._brm_svi_unit_steps(nothing, :pair) == steps(0.1, 0.1)
    @test SU._brm_svi_unit_steps(0.5, :orthogonal) == steps(0.5, 0.5)
    @test SU._brm_svi_unit_steps(0.5, :pair) == steps(0.5, 0.5)
    # Controls from before them keep what they ran with, so a resume stays exact.
    controls = SU._brm_svi_controls(1; batch_size=1, max_iterations=1, lr_global=0.01,
        unit_step=1.0, polish_lr=0.05, window=100, patience=10, min_windows=40, polish_windows=20)
    @test SU._brm_svi_structured_unit_step(controls) == 1.0
    @test SU._brm_svi_structured_rank_step(controls) == 1.0
    new = merge(controls, (; structured_unit_step=0.1, structured_rank_step=0.1))
    @test SU._brm_svi_structured_unit_step(new) == 0.1
    @test SU._brm_svi_structured_rank_step(new) == 0.1
    old = Base.structdiff(controls, NamedTuple{(:structured_unit_step, :structured_rank_step)})
    @test !haskey(old, :structured_unit_step) && !haskey(old, :structured_rank_step)
    @test SU._brm_svi_structured_unit_step(old) == 1.0
    @test SU._brm_svi_structured_rank_step(old) == 1.0

    # Each family takes its own: a zero step stops only the family it belongs to.
    t = SU.GaussianUnits([1], [[2], [3, 4, 5, 6, 7]],
        (; footprints=[[1, 2], [1, 3, 4, 5, 6, 7]]))
    for (dense, structured) in ((0.1, 0.0), (0.0, 0.1))
        q = SU._brm_svi_params(t, [0.0], SU.MeanFieldGlobal();
            global_scale=1.0, unit_scale=1.0, unit_dense_limit=2, unit_rank=2)
        @test q.U[1] isa Matrix && q.U[2] isa SU._BRMSVIUnitRoot
        c = SU._brm_svi_controls(2; batch_size=2, max_iterations=1, lr_global=0.0,
            unit_step=dense, polish_lr=0.0, window=2, patience=1, min_windows=2,
            polish_windows=2, structured_unit_step=structured)
        run = SU._brm_svi_run(Xoshiro(3), q, c, t)
        SU._brm_svi_advance!(run, t, c)
        @test (q.M[1] != zeros(1)) == (dense > 0)
        @test (q.M[2] != zeros(5)) == (structured > 0)
    end
end

@testset "a structured unit's default step stays on an exact Gaussian target" begin
    # A unit shaped like a station's statics and innovations: 15 coordinates pinned tight far
    # from the start, 285 left at their N(0, 1). The target is exact, so the factor's KL and
    # sds read off against the truth.
    P = vcat(fill(100.0, 15), ones(285))
    t = SU.TightUnit(P, vcat(5 .* randn(Xoshiro(9), 15), zeros(285)))
    default = SU._brm_svi_unit_steps(nothing, :orthogonal)
    fit = tight_unit_fit(t, 8, default.structured_unit_step; rank_step=default.structured_rank_step)
    @test fit.kl < 1e-3
    @test 0.9 < fit.last < 1.1
    @test fit.worst < 15
    # The controls: step 1, the default of every family for a time, sends Β's rows far off
    # the target and leaves them there.
    wide = tight_unit_fit(t, 8, 1.0)
    @test wide.worst > 100
    @test wide.kl > 1000
    # Rank 0 has no rows to send off, and 1 merely stays noisy there.
    @test tight_unit_fit(t, 0, default.structured_unit_step).kl < 1e-3
end

@testset "a structured unit's rank part beats the diagonal it contains" begin
    # One exact Gaussian unit whose covariance is a diagonal plus rank 2. Its family holds
    # the target, and holds the best diagonal, so anything above that diagonal's KL has
    # lost to doing nothing with the rank part. At the unit's whole step the rows walk off
    # and it does; at a tenth, the rate of the global factor's rows, they find the target
    # (snag svi-nonfinite-er-0923561d).
    rng = Xoshiro(1)
    d = 100
    B = randn(rng, d, 2)
    C = Matrix(Diagonal(exp.(randn(rng, d))) + B * B')
    H, c = inv(Symmetric(C)), randn(rng, d)
    t = SU.CorrelatedUnit(H, c)
    kl(Cq, m) = (tr(H * Cq) + dot(m, H * m) - d - logdet(Symmetric(H)) - logdet(Symmetric(Cq))) / 2
    best_diagonal = kl(Matrix(Diagonal(1 ./ diag(H))), zeros(d))
    @test best_diagonal > 3
    function fitted(rank_step; rank=8)
        q = SU._brm_svi_params(t, [0.0], SU.MeanFieldGlobal();
            global_scale=0.1, unit_scale=1.0, unit_dense_limit=0, unit_rank=rank)
        controls = SU._brm_svi_controls(1; batch_size=1, max_iterations=3000, lr_global=0.01,
            unit_step=1.0, polish_lr=0.05, window=100, patience=10, min_windows=1000,
            polish_windows=20, structured_unit_step=0.1, structured_rank_step=rank_step)
        run = SU._brm_svi_run(Xoshiro(1), q, controls, t)
        for _ in 1:3000
            SU._brm_svi_advance!(run, t, controls)
        end
        f = q.Λ[1].factor
        kl(Matrix(Diagonal(exp.(2 .* f.ρ)) + f.Β * f.Β'), q.M[1] .- c)
    end
    tenth = fitted(SU._brm_svi_unit_steps(nothing, :orthogonal).structured_rank_step)
    @test tenth < best_diagonal - 3
    # The control: the whole step loses to the diagonal it contains.
    @test fitted(1.0) > best_diagonal
    # A mean-field fit reaches that diagonal, so the two bounds are not vacuous.
    @test abs(fitted(1.0; rank=0) - best_diagonal) < 1
end

@testset "mixed local families in the production optimization loop" begin
    t = SU.GaussianUnits([1], [[2], [3, 4, 5, 6, 7]],
        (; footprints=[[1, 2], [1, 3, 4, 5, 6, 7]]))
    q = SU._brm_svi_params(t, [0.0], SU.MeanFieldGlobal();
        global_scale=1.0, unit_scale=1.0, unit_dense_limit=2, unit_rank=2)
    fit = SU.arithmetic_fit(Xoshiro(51), t, q; batch_size=2, max_iterations=40,
        lr_global=0.0, unit_step=0.05, polish_lr=0.0, window=2, patience=1,
        min_windows=2, polish_windows=2, unit_sampling=:work)
    @test all(isfinite, fit.elbo)
    @test fit.q.M[1][1] > 0
    @test all(fit.q.M[2] .> 0)
    @test fit.q.U[1] isa Matrix
    @test fit.q.U[2].factor === fit.q.Λ[2].factor
    @test q.unit_opt[2].t == fit.sampling.visits[2]
    @test all(isfinite, SU._brm_svi_draw_unit(fit.q, 2,
        SU._brm_svi_unit_noise(Xoshiro(52), fit.q.U[2])))
end

@testset "a dense factor's polishing average is not narrowed by the step's spread" begin
    # One exact Gaussian unit of 40 coordinates (precision 2I) under the
    # default schedule, the unit in every step. A visit changes the precision
    # by up to 2.5-fold along one drawn direction, so an average of precisions
    # is narrow: it returned sds of 0.75-0.77 times the truth here (snag
    # svi-dense-unit-f-71edf0f6). The covariance average is not.
    d = 40
    t = SU.GaussianUnits([1], [collect(2:d+1)], (; footprints=[collect(1:d+1)]))
    for seed in 1:3
        q = SU._brm_svi_params(t, [0.0], SU.MeanFieldGlobal();
            global_scale=1.0, unit_scale=1.0, unit_dense_limit=64)
        fit = SU.arithmetic_fit(Xoshiro(seed), t, q; batch_size=1,
            max_iterations=20_000, lr_global=0.01, unit_step=0.1, polish_lr=0.05,
            window=100, patience=10, min_windows=40, polish_windows=20)
        @test fit.converged
        Λ = fit.q.Λ[1]
        @test LowerTriangular(fit.q.U[1]) * LowerTriangular(fit.q.U[1])' ≈ Λ
        sd_ratio = sqrt(tr(2 .* inv(Symmetric(Λ))) / d)
        @test 0.93 < sd_ratio < 1.08
        @test maximum(abs, fit.q.M[1] .- 1) < 1e-8
    end
end
