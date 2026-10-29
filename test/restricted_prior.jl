# test/restricted_prior.jl — `restricted(D, lower, upper)`: a prior law that is
# `D`'s density on the bounds, not renormalized (Stan's law for a bounded
# declaration), and the deprecation of `D(...; lower, upper)` keywords on a
# prior in its favour (decision 0q18o98, todo 2026-10-29T23-49-52-862-1e4wxe4).
#
# Covered: the spelling rewrites to the declaration-bound form the backends
# read; the keyword form warns, `restricted` and `truncated` do not; Turing's
# log joint is the plain kernel; RK emits RKPPL's `restricted`; malformed
# spellings are refused. Not covered here: SBBRMI, whose plain kernel waits on
# a StanBlocks spelling (todo 2026-10-29T23-49-53-197-18t54wh).
#
# Run: julia --project=test test/restricted_prior.jl

using Test
using BayesianRegressionModels, Distributions, Turing
using ReactiveKernels, ReactiveKernelsPPL
const BRM = BayesianRegressionModels

const DATA = (; y=[0.1, -0.2])
operations(brmi) = sprint(show, getfield(brmi, :operations))

@testset "restricted rewrites to the declaration-bound form, without a warning" begin
    restricted_form = @test_logs @brm begin
        location ~ Normal(0, 1)
        bounded ~ restricted(Normal(0.2, 0.7), location, location + 2)
        y ~ Normal(bounded, 1)
    end
    keyword_form = @brm begin
        location ~ Normal(0, 1)
        bounded ~ Normal(0.2, 0.7; lower=location, upper=location + 2)
        y ~ Normal(bounded, 1)
    end
    by_keywords = @brm begin
        location ~ Normal(0, 1)
        bounded ~ restricted(Normal(0.2, 0.7); lower=location, upper=location + 2)
        y ~ Normal(bounded, 1)
    end
    brmi = @test_logs restricted_form(DATA)
    deprecated = @test_logs (:warn, r"`lower`/`upper` keywords on a prior.*deprecated") keyword_form(DATA)
    @test operations(brmi) == operations(deprecated) == operations(by_keywords(DATA))
    # `truncated` is the normalized law and stays as written.
    normalized = @brm begin
        bounded ~ truncated(Normal(0.2, 0.7), 0.0, 1.0)
        y ~ Normal(bounded, 1)
    end
    @test_logs normalized(DATA)
end

@testset "Turing evaluates the plain kernel" begin
    model = @brm begin
        location ~ Normal(0, 1)
        bounded ~ restricted(Normal(0.2, 0.7), location, location + 2)
        y ~ Normal(bounded, 1)
    end
    backend = TuringBRMI(model(DATA))
    for (location, bounded) in ((0.3, 1.1), (-0.4, 0.0))
        expected = logpdf(Normal(0, 1), location) + logpdf(Normal(0.2, 0.7), bounded) +
            sum(logpdf.(Normal(bounded, 1), DATA.y))
        @test Turing.logjoint(backend.model, (; location, bounded)) ≈ expected
    end
end

@testset "RK emits RKPPL's restricted" begin
    model = @brm begin
        location ~ Normal(0, 1)
        bounded ~ restricted(Normal(location, 0.7), -1.0, 3.0)
        y ~ Normal(bounded, 1)
    end
    main = sprint(Base.show_unquoted, BRM._rk_emit_ast(BRM._brm_rk_plan(model(DATA))).main)
    @test occursin("bounded ~ restricted(Normal(location, 0.7), -1.0, 3.0)", main)
end

@testset "malformed restricted spellings are refused by name" begin
    message(f) = try f(); "" catch e; sprint(showerror, e) end
    no_bounds = @brm begin
        x ~ restricted(Normal(0, 1))
        y ~ Normal(x, 1)
    end
    @test occursin("`restricted` takes a distribution and its bounds",
        message(() -> no_bounds(DATA)))
    nested = @brm begin
        x ~ restricted(Normal(0, 1; lower=0), 0, 1)
        y ~ Normal(x, 1)
    end
    @test occursin("without `lower`/`upper` keywords of its own",
        message(() -> nested(DATA)))
end

println("restricted_prior.jl: all testsets passed")
