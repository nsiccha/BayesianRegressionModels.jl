using Test, BayesianRegressionModels
const DOC_RK_BRM = BayesianRegressionModels

function documented_rk(statement, text="Authored explanation."; owner=:piece)
    doc = BRMDocumentation((owner,), owner, text, deepcopy(statement), :authored_model, 17)
    DOC_RK_BRM._brm_documented_statement(statement, (doc,))
end

@testset "The emitter selects an exact documentation scope before renaming" begin
    source = quote
        "Global value."
        theta ~ Normal(0, 1)
        @plate for i in eachindex(x)
            "Cell value."
            theta_cell = theta * x[i]
            expected[i] = theta_cell
        end
    end
    documentation = DOC_RK_BRM._brm_source_documentation(source)
    block = :(begin theta ~ Normal(0, 1); theta_cell = theta*x; end)
    documented = DOC_RK_BRM._rk_documentation_block(block, documentation)
    @test length(DOC_RK_BRM._brm_source_documentation(documented)) == 1
    @test only(DOC_RK_BRM._brm_source_documentation(documented)).markdown == "Global value."
    local_block = :(begin private_cell = theta*x; end)
    scoped = DOC_RK_BRM._rk_documentation_block(local_block, documentation;
        path=(:expected,), names=Dict(:theta_cell => :private_cell))
    @test only(DOC_RK_BRM._brm_source_documentation(scoped)).markdown == "Cell value."
    @test DOC_RK_BRM._brm_strip_docs(scoped) == local_block
    @test DOC_RK_BRM._rk_documentation_block(documented, documentation) == documented
    unrelated = DOC_RK_BRM._rk_documentation_block(local_block, documentation;
        path=(:another_component,), names=Dict(:theta_cell => :private_cell))
    @test unrelated == local_block
    dotted = BRMDocumentation((Symbol("expected.theta_cell"),),
        Symbol("expected.theta_cell"), "A literal dotted name.",
        :(private_cell = theta*x), :authored_model, 21)
    @test DOC_RK_BRM._rk_documentation_block(local_block, (dotted,);
        path=(:expected,), names=Dict(:theta_cell => :private_cell)) == local_block
end

struct CountedDocumentation <: AbstractVector{BRMDocumentation}
    records::Vector{BRMDocumentation}
    reads::Base.RefValue{Int}
end
Base.size(d::CountedDocumentation) = size(d.records)
function Base.getindex(d::CountedDocumentation, i::Int)
    d.reads[] += 1
    d.records[i]
end

@testset "Attaching documentation reads each input record once" begin
    names = [Symbol(:theta_, i) for i in 1:128]
    statements = [Expr(:(=), name, i) for (i, name) in enumerate(names)]
    docs = CountedDocumentation([BRMDocumentation((name,), name, string(name),
        statement, :large_model, i) for (i, (name, statement)) in
        enumerate(zip(names, statements))], Ref(0))
    source = Expr(:block, statements...)
    documented = DOC_RK_BRM._rk_documentation_block(source, docs)
    @test docs.reads[] == length(docs)
    @test DOC_RK_BRM._brm_strip_docs(documented) == source
    @test [d.path for d in DOC_RK_BRM._brm_source_documentation(documented)] ==
        [(name,) for name in names]
end

@testset "RK definitions retain outer documentation through canonicalization" begin
    definition = :(piece(location) = begin
        theta ~ Normal(location, tau)
        return theta
    end)
    documented = documented_rk(definition)
    explicit = documented_rk(:(ReactiveKernelsPPL.@rkppl $definition))
    for source in (documented, explicit)
        @test DOC_RK_BRM._rk_source_definition(source) == (; name=:piece, kind=:rkppl)
        canonical = DOC_RK_BRM._rk_canonical_source_definition(source)
        @test DOC_RK_BRM._brm_is_doc(canonical)
        @test DOC_RK_BRM._brm_undoc(canonical) == definition
        evaluated = DOC_RK_BRM._rk_definition_expression(canonical)
        @test DOC_RK_BRM._brm_is_doc(evaluated)
        @test evaluated.args[4].args[1] === Symbol("@rkppl")
        @test evaluated.args[4].args[end] == definition
        @test DOC_RK_BRM._rk_definition_parts(canonical) ==
            DOC_RK_BRM._rk_definition_parts(definition)
    end
    native = documented_rk(:(function square(x); x*x; end))
    @test DOC_RK_BRM._rk_source_definition(native) == (; name=:square, kind=:function)
    @test DOC_RK_BRM._rk_definition_expression(native) == native
    kernel = documented_rk(:(ReactiveKernels.@kernel square(x) = x*x))
    @test DOC_RK_BRM._rk_source_definition(kernel) == (; name=:square, kind=:kernel)
    @test DOC_RK_BRM._rk_kernel_entry_ports(kernel) == [:x]
end

@testset "Documented RK source preserves dependency slicing and fitted summaries" begin
    definition = :(piece(location) = begin
        theta ~ Normal(location, tau)
        return theta
    end)
    body = quote
        center = mean(training_x)
        tau = 0.5
        q ~ piece(center)
        unused ~ Normal(0, 1)
        y .~ Normal.(q, 1)
    end
    Base.remove_linenums!(body)
    plain = DOC_RK_BRM._RKEmittedProgram([definition], body)
    documented_definition = deepcopy(definition)
    documented_definition.args[2].args = Any[
        s isa Expr && DOC_RK_BRM.isxcall(s, :~) ? documented_rk(s; owner=:theta) : s
        for s in definition.args[2].args]
    documented = DOC_RK_BRM._RKEmittedProgram(
        [documented_rk(documented_definition)],
        Expr(:block, (documented_rk(s) for s in body.args)...))
    @test DOC_RK_BRM._rk_validate_source_definitions(documented) === nothing
    @test DOC_RK_BRM._rk_program_statements(documented) == DOC_RK_BRM._rk_program_statements(plain)
    @test DOC_RK_BRM._rk_source_symbols!(Set{Symbol}(), documented.main) ==
        DOC_RK_BRM._rk_source_symbols!(Set{Symbol}(), plain.main)
    @test DOC_RK_BRM._rk_block_free_reads(documented.defs) == DOC_RK_BRM._rk_block_free_reads(plain.defs)
    @test DOC_RK_BRM._rk_prior_reads(documented, (:y,)) == DOC_RK_BRM._rk_prior_reads(plain, (:y,))
    @test DOC_RK_BRM._rk_free_reads!(Set{Symbol}(), only(documented.defs)) == Set([:tau])
    selected = DOC_RK_BRM._rk_fitted_source(documented, (:y,))
    control = DOC_RK_BRM._rk_fitted_source(plain, (:y,))
    @test DOC_RK_BRM._rk_program_statements(selected) == DOC_RK_BRM._rk_program_statements(control)
    @test length(selected.main.args) == 4
    @test all(DOC_RK_BRM._brm_is_doc, selected.main.args)
    @test DOC_RK_BRM._brm_strip_docs(DOC_RK_BRM._rk_active_observation_source(
        documented, (:y,), Set([:y]))) ==
        DOC_RK_BRM._rk_active_observation_source(plain, (:y,), Set([:y]))
    plan = (; columns=Dict(:training_x => [-1., 1.], :y => [.2, .4], :center => [0.]))
    fitted = DOC_RK_BRM._rk_fitted_definitions(documented, plan, Set([:training_x]))
    @test fitted == Dict(:center => Set([:training_x]))
    @test DOC_RK_BRM._rk_fitted_definition(first(documented.main.args), fitted) === :center
    @test DOC_RK_BRM._rk_fitted_value_names(documented, plan, Set([:training_x]); prefix="test") ==
        DOC_RK_BRM._rk_fitted_value_names(plain, plan, Set([:training_x]); prefix="test")
    @test DOC_RK_BRM._rk_source_data_columns(plan, documented) ==
        DOC_RK_BRM._rk_source_data_columns(plan, plain)
end

@testset "A documented plate local does not become a model output" begin
    source = :(@plate for i in eachindex(x)
        local_mean = x[i]
        expected[i] = local_mean
        y[i] ~ Normal(local_mean, 1)
    end)
    Base.remove_linenums!(source)
    documented = deepcopy(source)
    body = last(last(documented.args).args)
    body.args = Any[documented_rk(s) for s in body.args]
    @test DOC_RK_BRM._rk_source_outputs!(Set{Symbol}(), documented) == Set([:expected, :y])
    @test DOC_RK_BRM._rk_source_outputs!(Set{Symbol}(), documented) ==
        DOC_RK_BRM._rk_source_outputs!(Set{Symbol}(), source)
end
