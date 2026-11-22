# test/model_scope_dense_solve.jl — a model-scope dense linear solve
# `eta = A \ b` on both backends (todo 1s3nxxv).
#
# Each backend must give the analytic log density and gradient: SBBRMI through
# BridgeStan, and RKBRMI through its production gradient, plain Enzyme Reverse.
# Frozen replay on a second design must do the same, and the RK replay must
# reuse the fitted program. The RK gradient reads the declared `b` through a
# view of the packed point, an interior one when another coordinate comes
# first. Before Enzyme's `dense_solve_views` repair that view made the solve's
# reverse pass raise IllegalTypeAnalysisException in `QRPivoted.getproperty`
# (ReactiveKernels snag reverse-gradient-6b38cade, Enzyme snag
# use-ordinary-enz-fd0accd2). Public synthetic models only.
#
# Run: julia --startup-file=no --project=test test/model_scope_dense_solve.jl

using Test, Distributions, BayesianRegressionModels, StanBlocks, LinearAlgebra
import BridgeStan, ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme

Enzyme.require_dogfood_repairs(:dense_solve_views;
    requester="test/model_scope_dense_solve.jl (RKBRMI `A \\ b` gradients)")

const solve_builder = @brm begin
    b[1:K] .~ Normal.(0, 1)
    eta = A \ b
    y .~ Normal.(eta, 1)
end

# `c` comes first, so `b` is an interior range of the packed point.
const shifted_solve_builder = @brm begin
    c ~ Normal(0, 1)
    b[1:K] .~ Normal.(0, 1)
    eta = A \ b
    y .~ Normal.(eta .+ c, 1)
end

# Log density and gradient of the two models at the packed point `u`
# (unconstrained = constrained: every coordinate is a Normal).
function solve_oracle(rows, u; shifted::Bool)
    c, b = shifted ? (u[1], u[2:end]) : (0.0, u)
    residual = rows.y .- rows.A \ b .- c
    lp = sum(logpdf.(Normal(), u)) + sum(logpdf.(Normal(), residual))
    db = transpose(rows.A) \ residual - b
    lp, shifted ? [sum(residual) - c; db] : db
end

coordinate_names(K; shifted) = [shifted ? ["c"] : String[]; ["b.$j" for j in 1:K]]

function check_solve(sb, rk, rows, u, path; shifted)
    lp, gradient = solve_oracle(rows, u; shifted)
    names = coordinate_names(rows.K; shifted)
    stan = stan_instantiate(sb; path).model
    @test BridgeStan.param_unc_names(stan) == names
    value, grad = BridgeStan.log_density_gradient(stan, u; propto=false)
    @test value ≈ lp atol=1e-10
    @test grad ≈ gradient atol=1e-10
    @test ReactiveKernelsPPL.coordinate_names(rk.model.layout) == Symbol.(names)
    problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, grad = LogDensityProblems.logdensity_and_gradient(problem, u)
    @test value ≈ lp atol=1e-10
    @test grad ≈ gradient atol=1e-10
    # A second evaluation of the same prepared gradient at another point.
    v = u .* -0.7 .+ 0.1
    lp, gradient = solve_oracle(rows, v; shifted)
    value, grad = LogDensityProblems.logdensity_and_gradient(problem, v)
    @test value ≈ lp atol=1e-10
    @test grad ≈ gradient atol=1e-10
end

@testset "model-scope dense solve" begin
    data = (; K=2, A=[2.0 -0.4; 0.3 1.5], y=[0.6, -0.7])
    design = (; K=2, A=[1.2 0.5; -0.7 2.4], y=[-0.3, 1.1])
    wider = (; K=3, A=[2.0 -0.4 0.1; 0.3 1.5 -0.6; 0.2 0.7 1.9], y=[0.6, -0.7, 0.4])
    mktempdir() do dir
        for (builder, shifted) in ((solve_builder, false), (shifted_solve_builder, true))
            for make in (builder, brm(builder.source; mod=@__MODULE__))
                @testset "$(shifted ? "shifted" : "plain"), $(make === builder ? "macro" : "runtime source")" begin
                    path = joinpath(dir, shifted ? "shifted.stan" : "plain.stan")
                    point(K) = collect(range(-0.4, 0.55; length=K + shifted))
                    model = make(data)
                    @test occursin("A \\ b", string(model))
                    sb, rk = SBBRMI(model; mod=@__MODULE__), RKBRMI(model)
                    check_solve(sb, rk, data, point(2), path; shifted)
                    replay_sb, replay_rk = reprocess(sb, design), reprocess(rk, design)
                    @test replay_rk.model === rk.model
                    check_solve(replay_sb, replay_rk, design, point(2), path; shifted)
                    model = make(wider)
                    check_solve(SBBRMI(model; mod=@__MODULE__), RKBRMI(model), wider,
                        point(3), path; shifted)
                end
            end
        end
    end
end
