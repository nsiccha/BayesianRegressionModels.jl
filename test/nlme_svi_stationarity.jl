# Statistical and storage controls for the iterate gate; no model compilation.
# RUN: julia --project=test test/nlme_svi_stationarity.jl [testset-filter...]
using Test, BayesianRegressionModels, Random, Statistics, Serialization, LinearAlgebra
include(joinpath(@__DIR__, "testset_filter.jl"))
const BRM = BayesianRegressionModels

function svi_iterate_history(xs)
    h = BRM._BRMSVIIterates(size(xs, 1))
    for x in eachcol(xs)
        slot = BRM._brm_svi_iterate_slot!(h)
        slot === nothing || (slot .= x)
    end
    h
end

@stestset "stationarity settling uses the run budget unless explicitly capped" begin
    options = (; batch_size=4, max_iterations=100_000, lr_global=0.01,
        unit_step=1.0, polish_lr=0.05, window=nothing, patience=10,
        min_windows=40, polish_windows=20)
    guarded = BRM._brm_svi_controls(1000; options...)
    @test guarded.window == 250
    @test guarded.settle_windows == 400
    @test BRM._brm_svi_controls(1000; options..., stationarity=false).settle_windows == 100
    @test BRM._brm_svi_controls(1000; options..., settle_windows=3).settle_windows == 3
    @test BRM._brm_svi_controls(1000; options..., settle_windows=0).settle_windows == 0
    @test BRM._brm_svi_budget_polish_due(95_000, guarded)
    @test !BRM._brm_svi_budget_polish_due(94_750, guarded)
    old = Base.structdiff(guarded, NamedTuple{(:budget_polish,)})
    @test !BRM._brm_svi_budget_polish_due(100_000, old)
    @test !BRM._brm_svi_budget_polish_due(100_000,
        BRM._brm_svi_controls(1000; options..., stationarity=false))
end

@stestset "stationarity history is bounded and has an exact update clock" begin
    h = svi_iterate_history(reshape(collect(1.0:10000), 1, :))
    @test h.updates == 10000
    @test h.count <= size(h.values, 2) == 512
    @test h.values[1, 1:h.count] == collect(Float64(h.stride):h.stride:10000)
    io = IOBuffer()
    serialize(io, h); seekstart(io)
    copy = deserialize(io)
    @test isequal(h, copy)
    for i in 10001:10500
        a, b = BRM._brm_svi_iterate_slot!(h), BRM._brm_svi_iterate_slot!(copy)
        @test (a === nothing) == (b === nothing)
        a === nothing || (a .= i; b .= i)
    end
    @test isequal(h, copy)
end

@stestset "stationarity diagnostics handle constants, missing history, and bad moments" begin
    @test !BRM._brm_svi_stationarity_check(svi_iterate_history(zeros(2, 199))).passed
    @test BRM._brm_svi_stationarity_check(svi_iterate_history(zeros(2, 200))).passed
    @test !BRM._brm_svi_stationarity_check(svi_iterate_history(fill(Inf, 1, 200))).passed
    @test !BRM._brm_svi_stationarity_check(svi_iterate_history(fill(NaN, 1, 200))).passed
    @test BRM._brm_svi_stationarity_check(BRM._BRMSVIIterates(0)).passed
end

function svi_stationary_ar(rng, d, n, rho)
    x = randn(rng, d, n)
    for j in 2:n
        x[:, j] .= rho .* view(x, :, j-1) .+ sqrt(1-rho^2) .* view(x, :, j)
    end
    x
end

@stestset "stationarity calibration covers autocorrelation and many coordinates" begin
    # Treat entire independent realizations as trials, not correlated time points.
    # The familywise maximum must be usable even at thousands of coordinates.
    for (d, n, rho) in ((1, 4096, 0.95), (32, 16384, 0.95), (1000, 512, 0.0))
        passed = map(vcat(1:20, 101:150)) do seed
            x = svi_stationary_ar(Xoshiro(seed), d, n, rho)
            BRM._brm_svi_stationarity_check(svi_iterate_history(x)).passed
        end
        @info "stationarity calibration" d n rho successes=sum(passed) trials=length(passed)
        @test mean(passed) >= 0.9
    end
    # A single moving coordinate cannot hide in a high quantile over stable ones.
    x = randn(Xoshiro(11), 1000, 512)
    x[end, :] .+= 0.05 .* (1:512)
    @test !BRM._brm_svi_stationarity_check(svi_iterate_history(x)).passed
    for seed in vcat(1:20, 101:150)
        x = svi_stationary_ar(Xoshiro(seed), 3, 4096, 0.9)
        x[2, :] .+= 0.003 .* (1:4096)
        @test !BRM._brm_svi_stationarity_check(svi_iterate_history(x)).passed
    end
end

# The drift gate of exploration's ELBO stop (src/nlme_svi_stationarity.jl, snag
# brm-svi-fit-unco-182711a4): `x` is the global history, D means then D log sds.
function svi_drift_gate(x; iteration=49_999, c=svi_drift_controls())
    D = size(x, 1) ÷ 2
    q = (; μ=zeros(D), ρ=zeros(D), Β=zeros(D, 0), M=[[0.0]], Λ=[ones(1, 1)])
    h = BRM._brm_svi_stationarity_state(q, c)
    h.globals = svi_iterate_history(x)
    run = (; iteration, windows=collect(1:iteration÷100), sampling=(; stationarity=h))
    BRM._brm_svi_exploration_drift(run, c), run
end
svi_drift_controls(; kw...) = merge((; stationarity=true, exploration_drift=true,
    window=100, min_windows=1, patience=1, settle_windows=10, max_iterations=100_000), kw)

@stestset "the drift gate holds exploration only while a global drifts steadily" begin
    # Stationary iterates, autocorrelated or many, almost never hold exploration
    # (0 of 70 realizations each, measured). A wander whose correlation time is a
    # fifth of the whole history (rho 0.99 at 512 samples) held 35 of 70: at that
    # length no window can tell it from a drift, and the averaging check's 1.1 fails
    # it too. The half-budget cap bounds what such a hold costs.
    for (d, n, rho) in ((2, 4096, 0.95), (64, 16384, 0.95), (2000, 512, 0.0))
        held = map(vcat(1:20, 101:150)) do seed
            first(svi_drift_gate(svi_stationary_ar(Xoshiro(seed), d, n, rho))).held
        end
        @info "drift gate calibration" d n rho holds=sum(held) trials=length(held)
        @test mean(held) <= 0.05
    end
    # A steady drift of one log sd, 25 noise sds over the history, holds it and is
    # named; the half-budget cap releases it.
    x = svi_stationary_ar(Xoshiro(7), 6, 512, 0.5)
    x[5, :] .+= 0.05 .* (1:512)
    drift, run = svi_drift_gate(x)
    @test drift.held && !drift.capped
    @test drift.rhat > BRM._BRM_SVI_DRIFT_RHAT
    @test first(drift.moving).index == 2 && first(drift.moving).moment === :log_sd
    capped, _ = svi_drift_gate(x; iteration=50_000)
    @test !capped.held && capped.capped
    # Each held window is counted on the current report.
    BRM._brm_svi_note_drift!(run, drift)
    BRM._brm_svi_note_drift!(run, drift)
    BRM._brm_svi_note_drift!(run, merge(drift, (; held=false)))
    @test run.sampling.stationarity.report.drift.held_windows == 2
    @test !run.sampling.stationarity.report.drift.held
    # A held reading is re-read only every tenth of the windows run so far.
    held, run = svi_drift_gate(x; iteration=20_000)
    BRM._brm_svi_note_drift!(run, held)
    at(w) = BRM._brm_svi_exploration_drift(merge(run, (; iteration=100w,
        windows=collect(1:w))), svi_drift_controls())
    @test at(219).window == 200 && at(219).held
    @test at(220).window == 220
    # Off: without the control (checkpoints written before it), without the iterate
    # check, and with no settling phase, where exploration waits for that check.
    for c in (Base.structdiff(svi_drift_controls(), NamedTuple{(:exploration_drift,)}),
            svi_drift_controls(; stationarity=false), svi_drift_controls(; settle_windows=0))
        @test first(svi_drift_gate(x; c)) === nothing
    end
    # Too short a history never holds exploration.
    @test !first(svi_drift_gate(x[:, 1:150])).held
    options = (; batch_size=4, max_iterations=100_000, lr_global=0.01,
        unit_step=1.0, polish_lr=0.05, window=nothing, patience=10,
        min_windows=40, polish_windows=20)
    @test BRM._brm_svi_controls(8; options...).exploration_drift
    @test !BRM._brm_svi_controls(8; options..., stationarity=false).exploration_drift
end

@stestset "stationarity reads marginal scales without dense low-rank covariance" begin
    rng = Xoshiro(4)
    q = (; μ=[0.2, -0.3], ρ=[-0.5, 400.0], Β=reshape([0.1, 0.0], 2, 1))
    h = BRM._BRMSVIIterates(4)
    BRM._brm_svi_record_global!(h, q)
    @test h.values[1:2, 1] == q.μ
    @test h.values[3, 1] ≈ log(sqrt(exp(-1) + 0.01))
    @test h.values[4, 1] == 400
    U = [2.0 0; 0.7 1.0]
    @test BRM._brm_svi_marginal_logsd(U*U') ≈ log.(diag(inv(U*U'))) ./ 2
    f = BRM._BRMSVILowRankUnit(randn(rng, 10000), randn(rng, 10000, 2))
    s = BRM._brm_svi_marginal_logsd(BRM._BRMSVIUnitPrecision(f))
    @test s ≈ log.(exp.(2f.ρ) .+ vec(sum(abs2, f.Β; dims=2))) ./ 2
end

@stestset "stationarity cannot certify an unvisited or drifting unit" begin
    c = (; stationarity=true, window=1, min_windows=1, patience=1, settle_windows=10)
    q = (; μ=[0.0], ρ=[0.0], Β=zeros(1,0), M=[[0.0]], Λ=[ones(1,1)])
    h = BRM._brm_svi_stationarity_state(q, c)
    h.globals = svi_iterate_history(randn(Xoshiro(3), 2, 512))
    run = (; iteration=512, sampling=(; stationarity=h))
    @test !BRM._brm_svi_stationary!(run, c)
    @test h.report.units[1].reason == :insufficient_history
    h.units[1] = svi_iterate_history(vcat(reshape(collect(1.0:512),1,:), zeros(1,512)))
    h.next_check = 512
    @test !BRM._brm_svi_stationary!(run, c)
    h.units[1] = svi_iterate_history(randn(Xoshiro(5), 2, 512))
    h.next_check = 512
    @test BRM._brm_svi_stationary!(run, c)
    @test h.report.passed
    # Mutating a delivered snapshot's diagnostic vector cannot reach the run.
    report = BRM._brm_svi_stationarity_report(run)
    report.current.units[1] = merge(report.current.units[1], (; passed=false))
    @test h.report.units[1].passed
    # New observations are required before another check can pass.
    @test !BRM._brm_svi_stationary!(run, c)
end

@stestset "stationarity detects changing coupling at constant marginal widths" begin
    c = (; stationarity=true, window=1, min_windows=1, patience=1)
    V = [1.0 0.0; 0.0 2.0]
    q = (; μ=zeros(2), ρ=zeros(2), Β=zeros(2, 0), M=[zeros(2)],
        Λ=[diagm([4.0, 9.0])], centering=(; x=[[0.1, 0.2]]),
        coupling=(; rows=[[1, 2]], V=[V]))
    h = BRM._brm_svi_stationarity_state(q, c).units[1]
    for _ in 1:512
        BRM._brm_svi_record_unit!(h, q, 1)
    end
    @test BRM._brm_svi_stationarity_check(h).passed
    @test size(h.values) == (10, 512)
    @test h.values[3:4, 1] ≈ log.([1/4 + 1, 1/9 + 4]) ./ 2
    @test h.values[5:6, 1] == BRM._brm_svi_unit_controls(q, 1)
    @test h.values[7:10, 1] == vec(V)

    moving = BRM._brm_svi_stationarity_state(q, c).units[1]
    for θ in range(0.0, pi/2; length=512)
        V .= [cos(θ) sin(θ); -2sin(θ) 2cos(θ)]
        BRM._brm_svi_record_unit!(moving, q, 1)
    end
    @test moving.values[3:4, :] ≈ repeat(h.values[3:4, 1:1], 1, 512)
    @test !BRM._brm_svi_stationarity_check(moving).passed

    uncoupled = merge(q, (; coupling=(; rows=[Int[]], V=[zeros(0, 2)])))
    empty = BRM._brm_svi_stationarity_state(uncoupled, c).units[1]
    BRM._brm_svi_record_unit!(empty, uncoupled, 1)
    @test empty.values[3:4, 1] ≈ log.([1/4, 1/9]) ./ 2
    @test size(empty.values, 1) == 6
end
