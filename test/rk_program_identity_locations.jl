# test/rk_program_identity_locations.jl — a model's program identity does not
# depend on where its macro calls were loaded from (snag
# `brm-svi-resume-r-d9940eed`).
#
# An emitted `ReactiveKernels.@kernel` definition keeps its source location:
# BRM's own file in this installation (a depot slug and line) or a source
# provider's file in the consumer's checkout. Every identity that compares the
# program (SVI optimizer checkpoints, SVI and SMC streaming carries, SMC and
# filter identities, fitted-state replay) read that place, so the same model
# loaded from another checkout, depot, host or BRM release was refused as a
# different program. Oracle: one model whose provider writes its kernel at two
# places has one program text and one identity, while the emitted program
# itself still names the place. Public synthetic model only.
#
# Run: julia --startup-file=no --project=test test/rk_program_identity_locations.jl

using Test, BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))

module LocatedKernels
using BayesianRegressionModels, Distributions, ReactiveKernels
import BayesianRegressionModels: _rk_callable_source!
scaled_mean(mu, s) = mu * s
# Where the provider's kernel reads as written: a checkout's file and line.
const PLACE = Ref(LineNumberNode(12, Symbol("/fleet/job-a/checkout/src/kernels.jl")))
function _rk_callable_source!(definitions, bindings, entry, ::typeof(scaled_mean))
    kernel = Expr(:., :ReactiveKernels, QuoteNode(Symbol("@kernel")))
    push!(definitions, Expr(:macrocall, kernel, PLACE[], :($entry(mu, s) = begin
        value = mu * s
        return value
    end)))
    :done
end
model(d) = @brm d begin
    mu ~ Normal(0, 1)
    sigma ~ Exponential(1)
    @plate for i in eachindex(y)
        m[i] = scaled_mean(mu, s[i])
        y[i] ~ Normal(m[i], sigma)
    end
end
end
const LK = LocatedKernels

const DATA = (; y=[0.3, -0.2, 1.1, 0.4], s=[1.0, 0.5, 2.0, 1.5])
const HERE = LineNumberNode(12, Symbol("/fleet/job-a/checkout/src/kernels.jl"))
const ELSEWHERE = LineNumberNode(240, Symbol("/home/user/.julia/packages/X/Ab3dE/src/kernels.jl"))

function build_at(place)
    LK.PLACE[] = place
    RKBRMI(LK.model(DATA))
end
statements(rk) = BRM._rk_program_statements(BRM._rk_retained_program(rk))

@stestset "a macro call's place is not part of the program identity" begin
    a, b = build_at(HERE), build_at(ELSEWHERE)
    # Positive control: the emitted program itself names each place.
    shown(rk) = join(map(d -> sprint(show, d), BRM._rk_retained_program(rk).defs), "\n")
    @test occursin("job-a/checkout/src/kernels.jl", shown(a))
    @test occursin("packages/X/Ab3dE/src/kernels.jl", shown(b))
    # The text before 0.2.225 read each place, so it told the two apart.
    located(rk) = map(d -> sprint(Base.show_unquoted,
        Base.remove_linenums!(BRM._brm_strip_docs(d))), BRM._rk_retained_program(rk).defs)
    @test located(a) != located(b)
    # Its identity text names neither, and reads the same.
    @test statements(a) == statements(b)
    @test !any(s -> occursin("kernels.jl", s), statements(a))
    @test any(s -> occursin("#= line 0 =# ReactiveKernels.@kernel", s), statements(a))
    @test BRM._rk_identity_statements(a) == BRM._rk_identity_statements(b)
    @test BRM._brm_rk_binding_parts(a) == BRM._brm_rk_binding_parts(b)
    @test BRM._brm_rk_binding_identity(a, "brm-svi-v1") ==
        BRM._brm_rk_binding_identity(b, "brm-svi-v1")
    @test BRM._brm_rk_binding_parts(a).format == 2
    # The text is rendered from a copy: the retained program keeps its place.
    @test occursin("job-a/checkout/src/kernels.jl", shown(a))
end

@stestset "SVI resume compares the program without places, and explains older parts" begin
    a, b = build_at(HERE), build_at(ELSEWHERE)
    parts = BRM._brm_rk_binding_parts(a)
    @test BRM._brm_svi_check_binding(b, (; binding_parts=parts)) === nothing
    # Parts written before format 2 hold the digest of the located text.
    located = (; program="digest of the program with its places",
        callables=parts.callables, data=parts.data, coordinates=parts.coordinates)
    refusal = try
        BRM._brm_svi_check_binding(b, (; binding_parts=located)); nothing
    catch e
        e
    end
    @test refusal isa ArgumentError
    @test occursin("in its emitted program (the checkpoint's program digest predates BRM's place-free program identity",
        refusal.msg)
    # A changed program in a current checkpoint is refused without that note.
    changed = merge(parts, (; program="another program"))
    refusal = try
        BRM._brm_svi_check_binding(b, (; binding_parts=changed)); nothing
    catch e
        e
    end
    @test refusal isa ArgumentError
    @test occursin("in its emitted program;", refusal.msg)
    @test !occursin("place-free", refusal.msg)
end

@stestset "a fitted state exported with places still replays" begin
    @test BRM._rk_unlocated_statement(
        "definition: #= /fleet/job-a/checkout/src/kernels.jl:12 =# ReactiveKernels.@kernel(f(x) = x)") ==
        "definition: #= line 0 =# ReactiveKernels.@kernel(f(x) = x)"
    @test BRM._rk_unlocated_statement("definition: #= line 7 =# @kernel(f(x) = x)") ==
        "definition: #= line 0 =# @kernel(f(x) = x)"
    @test BRM._rk_unlocated_statement("statement: y = x + 1") == "statement: y = x + 1"

    a = build_at(HERE)
    state = brm_fitted_state(a)
    # As an earlier version exported it: the provider's kernel with its place.
    program = map(s -> replace(s, "#= line 0 =# ReactiveKernels.@kernel" =>
        "#= /fleet/job-a/checkout/src/kernels.jl:12 =# ReactiveKernels.@kernel"), state.program)
    @test program != state.program
    fields = map(f -> f === :program ? program : getfield(state, f), fieldnames(typeof(state)))
    exported = typeof(state)(fields...)
    LK.PLACE[] = ELSEWHERE
    replay = RKBRMI(LK.model(DATA), exported)
    @test statements(replay) == statements(a)
end
