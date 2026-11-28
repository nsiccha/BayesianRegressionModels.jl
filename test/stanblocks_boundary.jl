# test/stanblocks_boundary.jl — the RK backend reaches the Stan backend's
# package only through src/stan_vocabulary.jl.
#
# StanBlocks (and BridgeStan, which it requires) is becoming a weak dependency
# of BRM (snag `stanblocks-and-b-0e9cf921`, user pick `079l8hv`): an
# environment that only uses RKBRMI should not carry it. The first step is that
# RK and shared numeric code never names StanBlocks, BridgeStan or SBBRMI. It
# asks the functions src/stan_vocabulary.jl declares, whose fallbacks answer
# for a session without StanBlocks, and the Stan backend adds the methods that
# answer with it (src/sb_vocabulary.jl). This file checks that rule on the
# parsed source (comments and strings do not count) and checks both answers
# of every vocabulary function.
#
# Run: julia --startup-file=no --project=test test/stanblocks_boundary.jl
using Test, BayesianRegressionModels, StanBlocks
using Distributions: Normal
const BRM = BayesianRegressionModels

const REPO = dirname(@__DIR__)
const STAN_NAMES = (:StanBlocks, :BridgeStan, :SBBRMI)

# Every unquoted identifier in `ex`, with its line.
function _identifiers!(out, ex, line=0)
    if ex isa Symbol
        push!(out, (ex, line))
    elseif ex isa LineNumberNode
        return ex.line
    elseif ex isa QuoteNode
        return line
    elseif ex isa Expr
        ex.head === :quote && return line
        for a in ex.args
            a isa LineNumberNode ? (line = a.line) : _identifiers!(out, a, line)
        end
    end
    line
end

function _named_stan(path)
    out = Tuple{Symbol,Int}[]
    _identifiers!(out, Meta.parseall(read(path, String); filename=path))
    [(name, line) for (name, line) in out if name in STAN_NAMES]
end

# The RK backend's files, and the shared files RK lowering and NLME/SVI fitting
# run. `rk_artifact.jl` is not one: it cross-checks RK against the Stan
# backend and moves with the Stan backend.
const RK_FILES = vcat(
    sort!([f for f in readdir(joinpath(REPO, "src")) if startswith(f, "rk_") &&
        endswith(f, ".jl") && f != "rk_artifact.jl"]),
    ["ode.jl", "nlme_view.jl", "nlme_svi_laplace_plan.jl", "preparation_priors.jl",
     "stan_vocabulary.jl"])

@testset "RK and shared numeric code never names StanBlocks, BridgeStan or SBBRMI" begin
    @test length(RK_FILES) > 30
    violations = String[]
    for file in RK_FILES
        for (name, line) in _named_stan(joinpath(REPO, "src", file))
            push!(violations, "src/$file:$line names `$name`; ask a function " *
                "src/stan_vocabulary.jl declares")
        end
    end
    isempty(violations) || foreach(v -> @error(v), violations)
    @test isempty(violations)
    # Positive control: the scan sees a name in code, not in a comment or a
    # string.
    probe = tempname() * ".jl"
    write(probe, "# StanBlocks in a comment\nf() = \"BridgeStan in a string\"\ng(x) = StanBlocks.normal(x)\n")
    @test _named_stan(probe) == [(:StanBlocks, 3)]
    rm(probe)
end

# `invoke` on the fallback's signature gives the answer for a session without
# StanBlocks: no value is a StanBlocks object there and no SLIC name exists.
fallback(f, args...) = invoke(f, Tuple{ntuple(_ -> Any, length(args))...}, args...)

@testset "each vocabulary function answers with and without StanBlocks" begin
    @test BRM._brm_stan_token(:normal) === StanBlocks.normal
    @test BRM._brm_stan_token(:weighted) === StanBlocks.weighted
    @test BRM._brm_stan_token(:no_such_slic_name) === nothing
    @test fallback(BRM._brm_stan_token, :normal) === nothing

    @test BRM._brm_stan_builtin(:ode_rk45) === StanBlocks.builtin.ode_rk45
    @test BRM._brm_stan_builtin(:no_such_builtin) === nothing
    @test fallback(BRM._brm_stan_builtin, :ode_rk45) === nothing
    @test BRM._brm_stan_builtin_ref(:ode_rk45) == GlobalRef(StanBlocks.builtin, :ode_rk45)
    @test_throws r"Stan's builtin spelling, which needs StanBlocks" fallback(BRM._brm_stan_builtin_ref, :ode_rk45)

    for name in (:normal, :student_t, :weighted, :censored, :truncated, :interval_censored)
        @test BRM._brm_stan_family(getfield(StanBlocks, name)) === name
        @test fallback(BRM._brm_stan_family, getfield(StanBlocks, name)) === nothing
    end
    # The tokens are told apart by identity, so a Julia value is never one.
    @test BRM._brm_stan_family(Normal) === nothing
    @test BRM._brm_stan_family(sum) === nothing
    @test BRM._brm_stan_family(StanBlocks.builtin.ode_rk45) === nothing

    @test BRM._brm_stan_value_family_label(sum) === nothing
    @test BRM._brm_stan_value_family_label(Normal) === nothing

    message(f, args...) = try fallback(f, args...); "" catch err sprint(showerror, err) end
    @test occursin("declared by StanBlocks", message(BRM._brm_stan_family_support, sum, 1.0))
    @test occursin("no bound observation", message(BRM._brm_stan_unconditioned_generative, nothing))
    @test occursin("omitted response", message(BRM._brm_stan_unbound_observations, nothing))
    @test occursin("sampled parameters", message(BRM._brm_stan_sampled_parameters, nothing))
end

@testset "a prior-only RK program still reads the Stan backend's declaration plan" begin
    model = @brm begin
        slope ~ Normal(0.0, 1.0)
        mu = slope * dose
        y ~ Normal(mu, 1.0)
    end
    brmi = model((; dose=[0.0, 1.0, 2.0]))
    generated = BRM._brm_stan_unconditioned_generative(brmi)
    @test generated isa BRM.GenerativePlan
    @test :y in Tuple(d.target for d in generated.declarations if d.role === :observation)
    bound = model((; dose=[0.0, 1.0, 2.0], y=[0.1, 0.2, 0.3]))
    @test BRM._brm_stan_unconditioned_generative(bound) === nothing
    @test BRM._brm_stan_unbound_observations(brmi) == [:y]
    @test isempty(BRM._brm_stan_unbound_observations(bound))
end

# The functions emitted RK source calls by name must run without StanBlocks, so
# each has a Julia method and none comes from the Stan backend's files (a SLIC
# `@deffun @juliacompat` method is generated by StanBlocks there). This also
# checks that a `@deffun` extending one of them added no Julia method.
_stan_file(m) = (f = basename(string(m.file)); f == "sbimpl.jl" || startswith(f, "sb_"))
@testset "every callable RK source imports from BRM is plain Julia" begin
    names = BRM._RK_SOURCE_IMPORTS.BayesianRegressionModels
    @test :brm_rw2_transform in names && :brm_rw2_basis_path in names
    for name in names
        ms = methods(getfield(BRM, name))
        @test !isempty(ms)
        stan = [m for m in ms if _stan_file(m)]
        isempty(stan) || @error "`$name` has methods from the Stan backend's files" stan
        @test isempty(stan)
    end
    @test !isempty(methods(BRM.brm_rw2_factors)) && !any(_stan_file, methods(BRM.brm_rw2_factors))
end

# The rw2 basis transform, checked against its definition rather than a
# recorded output: with D the second differences, AA' = DD' and
# CC' = A'A + sigma² I, and the transform returns r = A(C' \ z) and
# log|A| - log|C| (src/rw2_walk.jl).
@testset "the plain-Julia rw2 transform satisfies its defining identities" begin
    LA = BRM.LinearAlgebra
    for (m, sigma) in ((1, 0.3), (2, 0.7), (5, 0.05), (12, 1.3))
        n = m + 2
        b = BRM.brm_rw2_factors(sigma, m)
        A = zeros(m, m); C = zeros(m, m)
        for i in 1:m
            A[i, i] = b[i, 1]; C[i, i] = b[i, 4]
            i > 1 && (A[i, i - 1] = b[i, 2]; C[i, i - 1] = b[i, 5])
            i > 2 && (A[i, i - 2] = b[i, 3]; C[i, i - 2] = b[i, 6])
        end
        D = zeros(m, n)
        for i in 1:m
            D[i, i:i + 2] = [1.0, -2.0, 1.0]
        end
        z = collect(range(-1.0, 1.0; length=n)) .^ 3 .+ 0.1
        out = BRM.brm_rw2_transform(sigma, z)
        @test A * transpose(A) ≈ D * transpose(D) atol=1e-10
        @test C * transpose(C) ≈ transpose(A) * A + sigma^2 * LA.I atol=1e-10
        @test out[1:2] == z[1:2]
        @test out[3:n] ≈ A * (transpose(C) \ z[3:n]) atol=1e-10
        @test out[n + 1] ≈ LA.logdet(A) - LA.logdet(C) atol=1e-10
        path = BRM.brm_rw2_basis_path(sigma, z)
        @test length(path) == n && all(isfinite, path)
    end
end
