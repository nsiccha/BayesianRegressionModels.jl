# `rkppl_source(rk)` returns the program an `RKBRMI` runs as source text: each
# definition under its docstring, then the model. The text is the program:
# parsing it gives back the emitted definitions and main block. A sum of several
# terms is written infix (`a .+ b .+ c`, which parses nested to the left, where
# the emitter holds one call of three operands), with no `Core.@doc` call and no
# line marker.
#
#     julia --project=test test/rk_source_text.jl
using Test, BayesianRegressionModels, Distributions
using ReactiveKernels, ReactiveKernelsPPL
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))

const TEXT_DATA = (;
    subject=repeat(["s1", "s2", "s3", "s4"]; inner=4),
    x=[sin(1.3j) for j in 1:16], t=[0.25j for j in 1:16],
    east=[cos(0.9j) for j in 1:16], north=[sin(0.4j) for j in 1:16],
    y=[0.4 * sin(2.1j) + 1.5 for j in 1:16])

# Line numbers are no part of a program, and a documented statement is the same
# statement whether it is written as `Core.@doc text statement` (top level) or
# as a string literal before it (inside a block). Docstrings compare without
# trailing whitespace: a triple-quoted block ends with a newline.
# A chain of sums or products is the same program as one call of several
# operands and nested to the left: the emitter holds `.+(a, b, c)` where the
# typed text parses nested, and `(a * b) * c` where the typed text parses as
# one call. A non-finite number the emitter holds as a value (the upper bound
# `Inf` of a restricted prior) reads back as its name.
strip_program(x) = x
strip_program(x::AbstractFloat) = isfinite(x) ? x : Meta.parse(repr(x))
function strip_program(x::Expr)
    x = BRM._brm_normalize_docs(x)
    if BRM._brm_is_doc(x)
        return Expr(:documented, rstrip(String(x.args[3])), strip_program(x.args[4]))
    end
    args = Any[strip_program(a) for a in x.args if !(a isa LineNumberNode)]
    x.head === :call && length(args) > 3 && args[1] in (:.+, :.*, :+, :*) &&
        return foldl((left, right) -> Expr(:call, args[1], left, right), args[2:end])
    Expr(x.head, args...)
end
same_program(a, b) = isequal(strip_program(a), strip_program(b))

# The definitions of a printed program, each as the emitter spells it: a
# submodel bare, without the `@rkppl` it is typed with.
function parsed_definitions(text)
    parsed = Meta.parseall(text)
    [BRM._rk_canonical_source_definition(definition)
        for definition in parsed.args if !(definition isa LineNumberNode)]
end
parsed_main(text) = Meta.parse("begin\n" * text * "\nend")

function check_source_text(rk)
    program = BRM._rk_retained_program(rk)
    definitions = rkppl_source(rk; part=:definitions)
    main = rkppl_source(rk; part=:main)
    whole = rkppl_source(rk)
    @test length(parsed_definitions(definitions)) == length(program.defs)
    @test all(splat(same_program), zip(parsed_definitions(definitions), program.defs))
    @test same_program(parsed_main(main), program.main)
    # The whole program is the definitions, then the model under `@rkppl`.
    model = last(filter(x -> !(x isa LineNumberNode), Meta.parseall(whole).args))
    @test Meta.isexpr(model, :macrocall) && model.args[1] === Symbol("@rkppl")
    @test same_program(last(model.args), program.main)
    @test startswith(whole, definitions)
    for text in (definitions, main, whole)
        @test !occursin("#=", text)
        @test !occursin("@doc", text)
        @test !occursin(r"\.[+*]\(", text)
        @test !occursin('\t', text)
    end
    (; definitions, main, whole)
end

@stestset "a formula model prints as the program it runs" begin
    brmi = @brm TEXT_DATA begin
        "The mean of `y`: a slope, a smooth trend and a subject effect."
        mu ~ 1 + x + hsgp(t; k=4) + (1 | subject)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    rk = RKBRMI(brmi)
    text = check_source_text(rk)
    # Each definition sits under its docstring and says what kind it is.
    @test occursin("\"\"\"\nA Hilbert-space Gaussian-process (HSGP) effect.", text.definitions)
    @test occursin(r"^@rkppl brm_hsgp_effect\("m, text.definitions)
    @test occursin(r"^ReactiveKernels\.@kernel brm_hsgp_basis_graph\("m, text.definitions)
    # Four columns per level; a block ends in its own column.
    @test occursin(r"^    rho_iso ~ "m, text.definitions)
    @test occursin(r"^end$"m, text.definitions)
    @test !occursin(r"^ {5,7}\S"m, text.definitions)
    # The sum of a predictor's terms is written as one types it.
    @test occursin(r"^mu = \S+ \.\+ \S+ \.\+ \S+"m, text.main)
    # The authored docstring stands above the statement it documents.
    @test occursin("\"The mean of `y`: a slope, a smooth trend and a subject effect.\"\n", text.main)
    @test occursin("@rkppl begin\n", text.whole) && endswith(text.whole, "end\n")
    @test_throws ArgumentError rkppl_source(rk; part=:model)
end

@stestset "a value model with a two-axis field prints as the program it runs" begin
    brmi = @brm TEXT_DATA begin
        field ~ 0 + hsgp(east, north; k=(3, 2), domain=((-1.5, 1.5), (-1.5, 1.5)))
        level ~ 1 + (1 | subject)
        mu = level + field + 0.5 * x
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    text = check_source_text(RKBRMI(brmi))
    @test occursin(r"^mu = .* \.\+ .* \.\+ "m, text.main)
end

# The printer itself, on the constructs emitted programs use: each prints in
# the typed form and parses back to itself.
@stestset "statements print as typed and parse back" begin
    code(x) = sprint(io -> BRM._rk_print_statement(io, x, 0))
    roundtrip(x) = same_program(Meta.parse(code(x)), x)
    plate = :(weights = ReactiveKernels.plate(axes(omega2, 1)) do b
        return sigma * exp(-0.25 * rho^2 * omega2[b])
    end)
    @test code(plate) == """
        weights = ReactiveKernels.plate(axes(omega2, 1)) do b
            return sigma * exp(-0.25 * rho ^ 2 * omega2[b])
        end"""
    @test roundtrip(plate)
    nested = :(fit_basis = stack(ReactiveKernels.plate(axis_1, axis_2) do x_1, x_2
        values = ReactiveKernels.plate(1:B) do b
            return sin(x_1 * b) * sin(x_2 * b)
        end
        return values
    end; dims=1))
    @test code(nested) == """
        fit_basis = stack(ReactiveKernels.plate(axis_1, axis_2) do x_1, x_2
            values = ReactiveKernels.plate(1:B) do b
                return sin(x_1 * b) * sin(x_2 * b)
            end
            return values
        end; dims=1)"""
    @test roundtrip(nested)
    # A conditional value is written as it is typed, a statement `if` in full.
    ternary = :(floor = modes[1] == 1 ? 0.0 : 4 * width / pi)
    @test code(ternary) == "floor = modes[1] == 1 ? 0.0 : 4 * width / pi"
    @test roundtrip(ternary)
    chained = :(z = a ? 1 : b ? 2 : 3)
    @test code(chained) == "z = a ? 1 : b ? 2 : 3"
    @test roundtrip(chained)
    scaled = :(z = 2 * (a ? 1 : 3))
    @test code(scaled) == "z = 2 * (a ? 1 : 3)"
    @test roundtrip(scaled)
    conditional = quote
        if K <= 1
            floor = 0.0
        elseif K == 2
            floor = 1.0
        else
            floor = 2.0
        end
    end
    conditional = only(filter(x -> !(x isa LineNumberNode), conditional.args))
    @test code(conditional) == """
        if K <= 1
            floor = 0.0
        elseif K == 2
            floor = 1.0
        else
            floor = 2.0
        end"""
    @test roundtrip(conditional)
    definition = :(effect(axis, rho_floor) = begin
        rho ~ restricted(LogNormal(0.0, 1.0), rho_floor, Inf)
        z[1:4] .~ Normal.(0, 1)
        (PHI, omega2) = basis(axis)
        return PHI * (rho .* z)
    end)
    @test code(definition) == """
        effect(axis, rho_floor) = begin
            rho ~ restricted(LogNormal(0.0, 1.0), rho_floor, Inf)
            z[1:4] .~ Normal.(0, 1)
            (PHI, omega2) = basis(axis)
            return PHI * (rho .* z)
        end"""
    @test roundtrip(definition)
    scan = :(bounds = ReactiveKernels.scan(1:200; init=(0.0, 1.0)) do bracket, step
        mid = (bracket[1] + bracket[2]) / 2
        ((bracket[1], mid), mid)
    end)
    @test code(scan) == """
        bounds = ReactiveKernels.scan(1:200; init=(0.0, 1.0)) do bracket, step
            mid = (bracket[1] + bracket[2]) / 2
            ((bracket[1], mid), mid)
        end"""
    @test roundtrip(scan)
    # A docstring that spans lines is a block; quotes, `\$` and backslashes in
    # it are written so that the text reads back unchanged.
    documented = Expr(:macrocall, GlobalRef(Core, Symbol("@doc")), LineNumberNode(1),
        "Costs \$5 for a \"wide\" basis.\nSee `a\\b`.", :(f(x) = begin
            return x
        end))
    printed = sprint(io -> BRM._rk_print_definition(io, documented))
    @test printed == """
        \"\"\"
        Costs \\\$5 for a "wide" basis.
        See `a\\\\b`.
        \"\"\"
        @rkppl f(x) = begin
            return x
        end"""
    @test same_program(only(parsed_definitions(printed)), documented)
end
