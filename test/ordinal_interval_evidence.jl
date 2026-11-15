# test/ordinal_interval_evidence.jl — coarsened ordinal evidence:
# `y_min ~ coarsened(Ordinal(...); upper = y_max)`, each row the inclusive
# category range `y_min..y_max` of one fixed, ordered level set (snag
# `ordinal-interval-0a6d9172`, design decision `1cxobqy`).
#
# Oracles:
# - the executable `Ordinal`'s range mass against `log(sum(probs))` and, in the
#   tails, against 256-bit MPFR normal tails;
# - RKBRMI (formula and value routes): the exact `Ordinal` model's own pointwise
#   log likelihood at the same coordinates, one dataset per category, so a range
#   row's log mass is the log-sum of its categories' exact masses; values and
#   native Enzyme Reverse gradients against central differences;
# - SBBRMI: the executable `Ordinal` at BridgeStan's constrained draw;
# - the RKBRMI JAX module against the native value and gradient (when
#   `RK_JAX_PYTHON` names a Python with jax).
#
# Run: julia --startup-file=no --project=test test/ordinal_interval_evidence.jl
using Test, Random
using BayesianRegressionModels, Distributions, CategoricalArrays
using ReactiveKernels, ReactiveKernelsPPL
using Enzyme, LogDensityProblems, SpecialFunctions
using DifferentiationInterface: AutoEnzyme
using LogExpFunctions: logsumexp
using StanBlocks
# RKPPL also exports the link tags.
using BayesianRegressionModels: Cumulative, StoppingRatio, LogitLink, ProbitLink, CloglogLink
const BRM = BayesianRegressionModels
const BridgeStan = StanBlocks.BridgeStan

include(joinpath(@__DIR__, "testset_filter.jl"))
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))

const LEVELS = 0:4
# Level 4 is never observed; rows 3, 6, 7 and 11 are ranges ({2,3}, {1,3},
# {0,1}, {2,3}), every other row an exact category.
const LO = [0, 1, 2, 2, 3, 1, 0, 2, 1, 3, 2, 0]
const HI = [0, 1, 3, 2, 3, 3, 1, 2, 1, 3, 3, 0]
const N = length(LO)
labels(v) = categorical(v; levels=LEVELS, ordered=true)
code(v) = v .+ 1

function design(lo=LO, hi=HI)
    (; x=collect(range(-1.1, 1.3; length=N)),
       z=[0.3, -0.2, 0.5, 0.1, -0.7, 0.9, 0.2, -0.4, 0.6, -0.1, 0.4, -0.5],
       g=repeat(["a", "b", "c"], N ÷ 3),
       y_min=labels(lo), y_max=labels(hi))
end

@testset "executable Ordinal range mass" begin
    cuts = [-1.0, -0.2, 0.4, 1.3]
    for structure in (Cumulative(), StoppingRatio()),
        link in (LogitLink(), ProbitLink(), CloglogLink())
        d = Ordinal(structure, link, 0.3, cuts; discrimination=1.4)
        p = probs(d)
        for lo in 1:5, hi in lo:5
            @test BRM._ordinal_range_logmass(d, lo, hi) ≈ log(sum(p[lo:hi])) atol=1e-12
        end
        @test BRM._ordinal_range_logmass(d, 3, 2) == -Inf
        @test BRM._ordinal_range_logmass(d, 0, 2) == -Inf
        @test BRM._ordinal_range_logmass(d, 2, 6) == -Inf
        @test logpdf.(Ref(d), 1:5) ≈ log.(p)
    end
    # Far below eta every interior mass is a difference of two cdfs that round
    # to one; it is the difference of the upper tails. 256-bit reference.
    q(z) = setprecision(BigFloat, 256) do
        erfc(BigFloat(z) / sqrt(big(2))) / 2
    end
    for eta in (-60.0, -400.0)
        d = Ordinal(Cumulative(), ProbitLink(), eta, cuts)
        z = cuts .- eta
        exact2 = Float64(log(q(z[1]) - q(z[2])))
        @test isfinite(logpdf(d, 2))
        @test logpdf(d, 2) ≈ exact2 rtol=1e-12
        range23 = Float64(log(q(z[1]) - q(z[3])))
        @test BRM._ordinal_range_logmass(d, 2, 3) ≈ range23 rtol=1e-12
    end
end

@testset "coarsened validation names the row" begin
    builder = @brm begin
        eta ~ 0 + x
        y_min ~ coarsened(Ordinal(Cumulative(), LogitLink(), eta); upper=y_max)
    end
    bad_order = design(LO, [0, 1, 1, 2, 3, 3, 1, 2, 1, 3, 3, 0])   # row 3: 2..1
    @test_throws "row 3: upper category `1` precedes the lower category `2`" SBBRMI(
        builder(bad_order); mod=@__MODULE__)
    @test_throws "row 3: upper category `1` precedes" RKBRMI(builder(bad_order))
    plain = merge(design(), (; y_min=LO, y_max=[0, 1, 4, 2, 3, 3, 1, 2, 1, 3, 3, 0]))
    @test_throws "upper label `4` is not one of the response's levels" SBBRMI(
        builder(plain); mod=@__MODULE__)
    @test_throws "upper label `4` is not one of the response's levels" RKBRMI(builder(plain))
    normal = @brm begin
        mu ~ 1 + x
        y_min ~ coarsened(Normal(mu, 1.0); upper=y_max)
    end
    @test_throws "`coarsened` supports an `Ordinal(structure, link, eta)` base" SBBRMI(
        normal((; x=randn(4), y_min=randn(4), y_max=randn(4))); mod=@__MODULE__)
end

# ---- RKBRMI ----------------------------------------------------------------

function rk_query(backend, name, u)
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    Base.invokelatest(prepare_query(backend.model, ext._rk_translated_plan(backend.plan), name), u)
end

function check_rk_gradient(backend, u)
    check_rk_source_roundtrip(backend)
    problem = rk_logdensity_problem(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
    @test isfinite(value)
    @test all(isfinite, gradient)
    h = 1e-5
    fd = map(eachindex(u)) do j
        plus, minus = copy(u), copy(u)
        plus[j] += h
        minus[j] -= h
        (LogDensityProblems.logdensity(problem, plus) -
         LogDensityProblems.logdensity(problem, minus)) / (2h)
    end
    @test gradient ≈ fd atol=2e-5 rtol=2e-5
end

layout_names(backend) = ReactiveKernelsPPL.coordinate_names(backend.model.layout)

# `exact(data)` builds the same model with `y_min ~ Ordinal(...)`.
function check_rk_coarsened(coarse, exact; value_route)
    data = design()
    backend = RKBRMI(coarse(data))
    @test (backend.plan isa BRM._RKValuePlan) == value_route
    references = [RKBRMI(exact(design(fill(level, N), fill(level, N)))) for level in LEVELS]
    @test all(r -> layout_names(r) == layout_names(backend), references)
    D = LogDensityProblems.dimension(rk_logdensity_problem(backend;
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse)))
    for u in (collect(range(-0.18, 0.22; length=D)), 0.4 .* randn(Xoshiro(7), D))
        exact_rows = [rk_query(r, :pointwise, u).y_min for r in references]
        expected = [logsumexp(exact_rows[k][i] for k in code(LO[i]):code(HI[i])) for i in 1:N]
        observed = rk_query(backend, :pointwise, u).y_min
        @test observed ≈ expected rtol=1e-12
        @test rk_query(backend, :likelihood, u) ≈ sum(expected) rtol=1e-12
        # Exact rows are the exact model's own masses.
        for i in findall(LO .== HI)
            @test observed[i] ≈ exact_rows[code(LO[i])][i] rtol=1e-12
        end
        check_rk_gradient(backend, u)
    end
    backend
end

@stestset "RKBRMI coarsened ordinal, formula route" begin
    cases = (
        ("cumulative probit", @brm(begin
            eta ~ 0 + x
            y_min ~ coarsened(Ordinal(Cumulative(), ProbitLink(), eta); upper=y_max)
        end), @brm(begin
            eta ~ 0 + x
            y_min ~ Ordinal(Cumulative(), ProbitLink(), eta)
        end)),
        ("cumulative logit, modelled discrimination", @brm(begin
            eta ~ 0 + x
            log(disc) ~ 0 + g
            y_min ~ coarsened(Ordinal(Cumulative(), LogitLink(), eta;
                                      discrimination=disc); upper=y_max)
        end), @brm(begin
            eta ~ 0 + x
            log(disc) ~ 0 + g
            y_min ~ Ordinal(Cumulative(), LogitLink(), eta; discrimination=disc)
        end)),
        ("stopping ratio cloglog, per_threshold", @brm(begin
            eta ~ 0 + x
            y_min ~ coarsened(Ordinal(StoppingRatio(), CloglogLink(), eta;
                                      per_threshold=(z,)); upper=y_max)
        end), @brm(begin
            eta ~ 0 + x
            y_min ~ Ordinal(StoppingRatio(), CloglogLink(), eta; per_threshold=(z,))
        end)))
    for (label, coarse, exact) in cases
        @testset "$label" begin
            backend = check_rk_coarsened(coarse, exact; value_route=false)
            # Four thresholds: the declared level 4 is kept.
            u = zeros(LogDensityProblems.dimension(rk_logdensity_problem(backend;
                ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))))
            @test length(ReactiveKernelsPPL.constrain(backend.model.layout, u).y_min_thresholds) == 4
        end
    end
end

@stestset "RKBRMI coarsened ordinal, authored location (value route)" begin
    coarse = @brm begin
        slope ~ 0 + x
        bias ~ Normal(0, 1)
        loc = slope + bias
        y_min ~ coarsened(Ordinal(Cumulative(), ProbitLink(), loc); upper=y_max)
    end
    exact = @brm begin
        slope ~ 0 + x
        bias ~ Normal(0, 1)
        loc = slope + bias
        y_min ~ Ordinal(Cumulative(), ProbitLink(), loc)
    end
    check_rk_coarsened(coarse, exact; value_route=true)
end

# ---- SBBRMI ----------------------------------------------------------------

const RUN_BRIDGESTAN = get(ENV, "BRM_ORDINAL_RUNTIME", "1") != "0"

function stan_problem(builder, data)
    descriptor = brm_descriptor(builder, data; mod=@__MODULE__)
    cache = joinpath(tempdir(), "brm-ordinal-coarsened")
    isdir(cache) || mkpath(cache)
    problem = brm_execute(descriptor, :instantiate;
        path=joinpath(cache, string(descriptor.id) * ".stan"))
    descriptor, problem
end

function stan_constrained(problem, q)
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=false)
    values = BridgeStan.param_constrain(problem.model, q; include_tp=true, include_gq=false)
    Dict(name => value for (name, value) in zip(names, values))
end

if RUN_BRIDGESTAN
    @testset "SBBRMI coarsened ordinal: density, pointwise, predictions" begin
        data = design()
        cases = (
            ("cumulative probit, modelled discrimination", Cumulative(), ProbitLink(),
             @brm(begin
                 eta ~ 0 + x
                 log(disc) ~ 0 + g
                 y_min ~ coarsened(Ordinal(Cumulative(), ProbitLink(), eta;
                                           discrimination=disc); upper=y_max)
             end)),
            ("stopping ratio cloglog, per_threshold", StoppingRatio(), CloglogLink(),
             @brm(begin
                 eta ~ 0 + x
                 y_min ~ coarsened(Ordinal(StoppingRatio(), CloglogLink(), eta;
                                           per_threshold=(z,)); upper=y_max)
             end)))
        for (label, structure, link, builder) in cases
            @testset "$label" begin
                descriptor, problem = stan_problem(builder, data)
                D = LogDensityProblems.dimension(problem)
                for q in (fill(0.05, D), 0.4 .* randn(Xoshiro(11), D))
                    lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
                    @test isfinite(lp)
                    @test all(isfinite, gradient)
                    pointwise = brm_execute(descriptor, :pointwise_loglik;
                        problem, draws=q, seed=20261115).y_min_likelihood
                    c = stan_constrained(problem, q)
                    cuts = [c["y_min_thresholds.$k"] for k in 1:4]
                    for i in 1:N
                        eta = c["eta.$i"]
                        d = structure isa Cumulative ?
                            Ordinal(structure, link, eta, cuts; discrimination=c["disc.$i"]) :
                            Ordinal(structure, link,
                                [eta + c["y_min_threshold_effect.$i.$k"] for k in 1:4], cuts)
                        @test pointwise[i] ≈ BRM._ordinal_range_logmass(
                            d, code(LO[i]), code(HI[i])) atol=1e-9
                    end
                    predictions = brm_execute(descriptor, :predict;
                        problem, draws=q, seed=20261115).y_min_gen
                    @test all(1 .<= predictions .<= 5)
                end
            end
        end
    end

    @testset "SBBRMI exact Ordinal keeps interior categories far below eta" begin
        # eta = beta * x with beta = 0.05 at q = 0.05: rows at x = -1200 put eta
        # at -60, sixty standard deviations above every threshold.
        tail = (; x=[-1200.0, -1200.0, 0.0, 0.5, -0.5, 1.0],
                  y=labels([1, 2, 0, 1, 2, 3]))
        builder = @brm begin
            eta ~ 0 + x
            y ~ Ordinal(Cumulative(), ProbitLink(), eta)
        end
        descriptor, problem = stan_problem(builder, tail)
        q = fill(0.05, LogDensityProblems.dimension(problem))
        pointwise = brm_execute(descriptor, :pointwise_loglik;
            problem, draws=q, seed=1).y_likelihood
        c = stan_constrained(problem, q)
        cuts = [c["y_thresholds.$k"] for k in 1:4]
        @test all(isfinite, pointwise)
        for i in eachindex(tail.x)
            d = Ordinal(Cumulative(), ProbitLink(), c["eta.$i"], cuts)
            @test pointwise[i] ≈ logpdf(d, levelcode(tail.y[i])) rtol=1e-9
        end
    end
else
    @info "Skipping SBBRMI coarsened ordinal execution (BRM_ORDINAL_RUNTIME=0)"
end

# ---- RKBRMI JAX module -------------------------------------------------------

include(joinpath(@__DIR__, "jax_module_driver.jl"))

@stestset "RKBRMI coarsened ordinal JAX module" begin
    builder = @brm begin
        eta ~ 0 + x
        log(disc) ~ 0 + g
        y_min ~ coarsened(Ordinal(Cumulative(), ProbitLink(), eta;
                                  discrimination=disc); upper=y_max)
    end
    rk = RKBRMI(builder(design()))
    src = jax_source(rk.model, rk_bound_plan(rk))
    @test src isa JAXSource
    if brm_jax_require_python()
        ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
        D = ext._rk_bound_layout(rk).total
        us = [0.3 .* randn(Xoshiro(100 + k), D) for k in 1:3]
        run = brm_jax_run(src, [brm_jax_inputs(rk, src)], [us])
        problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        for (j, u) in enumerate(us)
            point = run.datasets[1]["points"][j]
            lp, g = LogDensityProblems.logdensity_and_gradient(problem, u)
            @test point["log_density"]["eager"] ≈ lp rtol=1e-12
            @test point["log_density"]["jit"] ≈ lp rtol=1e-12
            @test point["log_likelihood"]["jit"] ≈ rk_query(rk, :likelihood, u) rtol=1e-12
            @test vec(collect(point["grad"])) ≈ g rtol=1e-10 atol=1e-10
        end
    end
end
