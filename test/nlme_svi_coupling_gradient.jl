# Actual minibatch iterations on a known Gaussian conditional. A unit's mean
# error must not shake a coupling already at its optimum. No model lowering.
# RUN: julia --project=test test/nlme_svi_coupling_gradient.jl
module CouplingGradientArithmetic
using Random, LinearAlgebra, Statistics
struct BRMI end
struct RKBRMI end
struct BRMNLMEModel end
include(joinpath(@__DIR__, "..", "src", "nlme_selection.jl"))
include(joinpath(@__DIR__, "..", "src", "nlme_svi.jl"))
include(joinpath(@__DIR__, "..", "src", "nlme_svi_sd_bound.jl"))

struct ConditionalUnit
    globals::Vector{Int}
    units::Vector{Vector{Int}}
    plan::NamedTuple
    plans::Vector{NamedTuple}
    work::Vector{NamedTuple}
    m::NamedTuple
    mode::Symbol
    calls::Vector{Int}
end
function ConditionalUnit(mode=:normal)
    plan = (; footprints=[[1, 2]])
    work = (; plus=zeros(1), minus=zeros(1))
    m = (; unit=zeros(2), view=(; coordinates=["g", "z"]))
    ConditionalUnit([1], [[2]], plan, [plan], [work], m, mode, [0])
end
_brm_svi_global_term(::ConditionalUnit, g) = (-sum(abs2, g) / 2, -g)
function _brm_svi_unit_local!(gg, gz, t::ConditionalUnit, i, g, z, plan, work)
    t.calls[i] += 1
    at_mean = g[1] == 0.0 && z[1] == 10.0
    if at_mean && t.mode in (:mean_nonfinite, :mean_domain)
        gg[1] += 123.0
        gz[1] = NaN
        t.mode === :mean_domain && throw(DomainError(z[1], "auxiliary point"))
        return NaN
    end
    t.mode === :draw_domain && !at_mean && throw(DomainError(z[1], "actual draw"))
    residual = 0.4g[1] - z[1]
    gg[1] -= 0.4residual
    gz[1] += residual
    t.mode === :draw_nonfinite && !at_mean && (gz[1] = NaN)
    -residual^2 / 2
end
function setup(mode=:normal)
    t = ConditionalUnit(mode)
    q = _brm_svi_params(t, [0.0], MeanFieldGlobal(); global_scale=1.0, unit_scale=1.0)
    t, q = _brm_svi_with_coupling(t, q, LinearCoupling())
    q.M[1] .= 10.0
    q.coupling.V[1] .= 0.4
    t, q, _BRMSVIAdam(2, 0.0)
end
end

using Test, Random, Statistics
const CG = CouplingGradientArithmetic

@testset "coupling at its optimum ignores a unit's unsettled mean" begin
    for curvature in (:pair, :orthogonal)
        plain, baseline = Float64[], Float64[]
        for seed in 1:100, enabled in (false, true)
            t, q, opt = CG.setup()
            CG._brm_svi_iteration!(Xoshiro(seed), t, q, opt, [1], 0.0, 0.0;
                curvature, coupling_baseline=enabled)
            push!(enabled ? baseline : plain, only(q.coupling.V[1]) - 0.4)
        end
        @test sqrt(mean(abs2, plain)) > 0.5
        @test maximum(abs, baseline) < 1e-12
    end
end

@testset "the baseline leaves the other gradients and random stream unchanged" begin
    for curvature in (:pair, :orthogonal), mode in (:normal, :mean_nonfinite, :mean_domain)
        outcomes = map((false, true)) do enabled
            t, q, opt = CG.setup(mode)
            rng = Xoshiro(7)
            value = CG._brm_svi_iteration!(rng, t, q, opt, [1], 0.01, 0.1;
                curvature, coupling_baseline=enabled, coupling_scale=0.0)
            @test t.calls[1] == 2 + enabled
            @test all(iszero, t.work[1].plus) && all(iszero, t.work[1].minus)
            (; value, μ=copy(q.μ), ρ=copy(q.ρ), M=copy(q.M[1]),
               U=copy(q.U[1]), m=copy(opt.m), v=copy(opt.v), next=rand(rng))
        end
        @test outcomes[1] == outcomes[2]
    end
end

@testset "actual draw failures remain failures before any parameter update" begin
    for (mode, error) in ((:draw_domain, DomainError), (:draw_nonfinite, CG.BRMSVINonfiniteError))
        t, q, opt = CG.setup(mode)
        before = deepcopy(q)
        @test_throws error CG._brm_svi_iteration!(Xoshiro(7), t, q, opt, [1], 0.01, 0.1)
        @test q.μ == before.μ && q.ρ == before.ρ && q.M == before.M
        @test q.coupling.V == before.coupling.V && opt.t == 0
    end
end

@testset "old controls retain the previous estimator and gradient count" begin
    for enabled in (false, true)
        t, q, opt = CG.setup()
        c = enabled ? (; coupling_baseline=true) : (;)
        @test CG._brm_svi_unit_calls(q, 1, c) == 2 + enabled
        CG._brm_svi_iteration!(Xoshiro(7), t, q, opt, [1], 0.0, 0.0;
            coupling_baseline=CG._brm_svi_coupling_baseline(c))
        @test t.calls[1] == CG._brm_svi_unit_calls(q, 1, c)
        @test (abs(only(q.coupling.V[1]) - 0.4) < 1e-12) == enabled
    end
end

@testset "independent exploration precedes coupling and survives continuation" begin
    c = CG._brm_svi_controls(1; batch_size=1, max_iterations=200,
        lr_global=0.0, unit_step=0.0, polish_lr=0.0, window=10,
        patience=1, min_windows=3, polish_windows=2, settle_windows=2,
        trend_windows=0, stationarity=false, sd_bound=nothing)
    t, q, _ = CG.setup()
    q.coupling.V[1] .= 0.0
    run = CG._brm_svi_run(Xoshiro(91), q, c, t)
    independent = CG._brm_svi_run(Xoshiro(91),
        Base.structdiff(deepcopy(q), NamedTuple{(:coupling,)}), c, t)
    phases = Symbol[]
    same_exploration = true
    resumed = nothing
    for it in 1:200
        phase = CG._brm_svi_phase(run, c)
        push!(phases, phase)
        status = CG._brm_svi_advance!(run, t, c)
        if phase === :exploring
            CG._brm_svi_advance!(independent, t, c)
            same_exploration &= all(iszero, only(run.q.coupling.V)) &&
                all(n -> getproperty(run.q, n) == getproperty(independent.q, n),
                    (:μ, :ρ, :Β, :M, :Λ, :U)) &&
                run.rng == independent.rng && run.trace == independent.trace
        elseif resumed === nothing
            # A detached run taken just after activation must continue the same
            # trajectory, including coupling and lazy polishing averages.
            resumed = deepcopy(run)
        else
            CG._brm_svi_advance!(resumed, t, c)
        end
        status === :converged && break
    end
    @test same_exploration
    @test unique(phases) == [:exploring, :settling, :polishing]
    @test any(!iszero, only(run.q.coupling.V))
    @test run.converged && resumed !== nothing
    @test run.q == resumed.q && run.average == resumed.average
    @test run.trace == resumed.trace && run.rng == resumed.rng
    @test run.unit_gradient_calls == resumed.unit_gradient_calls

    # A checkpoint lacking the new schedule control still learns dependence
    # during exploration, regardless of the current fresh-fit default.
    old = Base.structdiff(c, NamedTuple{(:coupling_warmup,)})
    t, q, _ = CG.setup()
    q.coupling.V[1] .= 0.0
    previous = CG._brm_svi_run(Xoshiro(91), q, old, t)
    CG._brm_svi_advance!(previous, t, old)
    @test CG._brm_svi_phase(previous, old) === :exploring
    @test any(!iszero, only(previous.q.coupling.V))
end
