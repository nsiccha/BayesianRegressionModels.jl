# test/column_type_size.jl — a column's type does not spell its expression tree.
#
# Run: julia --project=test test/column_type_size.jl
#
# `ExprColumn` and `NamedColumn` used to carry their arguments' and their
# definition's types as type parameters, and `_BRMPreparedExpr` its arguments',
# so a formula reading a named value spelled that value's whole tree in its own
# type. On a 479-statement model those types reached ~200 KB, and building the
# model and planning its RK program compiled each method that touches them once
# per tree: about ten minutes before the first RK gradient (snag
# `rkbrmi-start-up-8ed0a306`). A deeper chain of assignments must give columns
# of the same types, and building, planning and emitting it must compile no
# method whose signature spells such a tree.
using Test
using BayesianRegressionModels
using Distributions: Exponential, Normal
using ReactiveKernels, ReactiveKernelsPPL

const BRM = BayesianRegressionModels

# `depth` model-scope assignments, each reading the one before, a predictor
# whose covariate is the last, and an observation. Every operation name carries
# the marker `zqchain<depth>`.
function chain_builder(depth)
    v(k) = Symbol(:v, k, :_zqchain, depth)
    tag(base) = Symbol(base, :_zqchain, depth)
    statements = Any[:($(tag(:log_w)) ~ Normal(0, 0.5)),
        :($(v(1)) = exp($(tag(:log_w))) * x)]
    for k in 2:depth
        push!(statements, :($(v(k)) = $(v(k - 1)) + 0.1 * x))
    end
    push!(statements, :($(tag(:sigma)) ~ Exponential(1)),
        :($(tag(:mu)) ~ 1 + $(v(depth)) + (1 | g)),
        :(y ~ Normal($(tag(:mu)), $(tag(:sigma)))))
    BRM.brm(Expr(:block, statements...); mod=@__MODULE__)
end

const DATA = (; x=[-0.6, -0.1, 0.3, 0.8, 1.2, -0.4],
    g=["a", "b", "a", "c", "b", "c"], y=[0.2, -0.1, 0.5, 0.9, 1.1, 0.0])

# The largest type of any operation or of its parent column.
column_type_bytes(brmi) = maximum(BRM._brm_operation_values(brmi)) do op
    max(sizeof(string(typeof(op))), sizeof(string(typeof(parent(op)))))
end

Base.@nospecializeinfer function chain_stage(@nospecialize(builder))
    brmi = builder(DATA)
    artifact = BRM.emit_rk_artifact(brmi; case_id="column-type-size")
    (; brmi, artifact, backend=RKBRMI(brmi))
end

Base.@nospecializeinfer function compiled_during(@nospecialize(f))
    path, io = mktemp()
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), io.handle)
    value = try
        Base.invokelatest(f)
    finally
        ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), C_NULL)
        close(io)
    end
    lines = [last(split(line, '\t'; limit=2)) for line in eachline(path)]
    rm(path)
    value, lines
end

# Column and prepared-expression types; a signature naming one of them spells
# whatever tree its type carries.
const COLUMN_TYPE = r"BayesianRegressionModels\.(ExprColumn|NamedColumn|_BRMPreparedExpr)\{"

# One outer testset, so a failure in one check still runs the other.
@testset "column types do not spell their expression tree" begin
    @testset "a deeper chain gives columns of the same types" begin
        shallow = chain_builder(3)(DATA)
        deep = chain_builder(60)(DATA)
        @test column_type_bytes(deep) == column_type_bytes(shallow)
        @test column_type_bytes(deep) < 100
    end

    @testset "building and planning a deep chain compiles no tree-spelling signature" begin
        Base.invokelatest(chain_stage, chain_builder(3))
        member, lines = compiled_during(() -> chain_stage(chain_builder(60)))
        @test member.backend isa RKBRMI
        # Positive control: the dump records the deep member's own construction,
        # whose signatures name its operations.
        @test any(line -> occursin("_zqchain60", line), lines)
        # Signatures naming the member's operations are the per-name
        # specializations test/rk_planning_specialization.jl governs; every other
        # signature holding a column type is short whatever the chain's depth.
        spelled = filter(lines) do line
            occursin(COLUMN_TYPE, line) && !occursin("_zqchain60", line) && sizeof(line) > 4096
        end
        foreach(line -> println("SPELLED ", sizeof(line), " ", first(line, 200)), spelled)
        @test isempty(spelled)
    end
end
