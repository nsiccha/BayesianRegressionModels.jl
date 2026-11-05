using Test, BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
const DOC_NATIVE_BRM = BayesianRegressionModels
const DOC_NATIVE_EXT = Base.get_extension(DOC_NATIVE_BRM, :BayesianRegressionModelsReactiveKernelsExt)

@testset "BRM authored records lower through ordinary RKPPL wrappers" begin
    source = quote
        "Global slope."
        theta ~ Normal(0, 1)
        "Expected response."
        mu = theta .* x
        "Measured response."
        y ~ Normal(mu, 1)
    end
    data = (; x=[-.4, .2, .8], y=[.1, .3, .7])
    model = brm(source; mod=@__MODULE__)(data)
    plan = DOC_NATIVE_BRM._brm_rk_plan(model)
    plain = DOC_NATIVE_BRM._rk_emit_ast(plan)
    documented = DOC_NATIVE_BRM._rk_emit_ast(plan; documentation=brm_documentation(model))
    @test DOC_NATIVE_BRM._rk_program_statements(documented) ==
        DOC_NATIVE_BRM._rk_program_statements(plain)
    bound = DOC_NATIVE_EXT._rk_translate_from_emitted(plan, documented)
    control = DOC_NATIVE_EXT._rk_translate_from_emitted(plan, plain)
    records = rkppl_documentation(bound)
    @test [(d.path, d.markdown) for d in records] ==
        [((:theta,), "Global slope."), ((:mu,), "Expected response."),
         ((:y,), "Measured response.")]
    @test all(d -> d.line > 0 && d.file !== nothing, records)
    built, original = build_kernel(bound), build_kernel(control)
    @test coordinate_names(built.layout) == coordinate_names(original.layout)
    for u in ([0.0], [.13], [-.2]), preset in (:sampler, :prior, :likelihood)
        @test isequal(Base.invokelatest(prepare_query(built, bound, preset), u),
            Base.invokelatest(prepare_query(original, control, preset), u))
    end
    parsed = DOC_NATIVE_BRM._RKEmittedProgram(
        [Meta.parse(sprint(Base.show_unquoted, d)) for d in documented.defs],
        Meta.parse(sprint(Base.show_unquoted, documented.main)), documented.bindings)
    replay = DOC_NATIVE_EXT._rk_translate_from_emitted(plan, parsed)
    @test [(d.path, d.markdown) for d in rkppl_documentation(replay)] ==
        [(d.path, d.markdown) for d in records]
    backend = RKBRMI(model)
    @test [(d.path, d.markdown) for d in rkppl_documentation(rk_bound_plan(backend))] ==
        [(d.path, d.markdown) for d in records]
    artifact = DOC_NATIVE_BRM.emit_rk_artifact(model; case_id="authored-native")
    @test [(d.path, d.markdown) for d in rkppl_documentation(rk_translate_artifact(artifact))] ==
        [(d.path, d.markdown) for d in records]
end

@testset "Plate-local BRM prose follows explicit emitted names" begin
    source = quote
        "Global slope."
        theta ~ Normal(0, 1)
        @plate for i in eachindex(x)
            "Cell mean."
            mean_i = theta*x[i]
            "Cell output."
            expected[i] = mean_i
            "Cell response."
            y[i] ~ Normal(mean_i, 1)
        end
    end
    data = (; x=[-.4, .2, .8], y=[.1, .3, .7])
    model = brm(source; mod=@__MODULE__)(data)
    artifact = DOC_NATIVE_BRM.emit_rk_artifact(model; case_id="authored-plate")
    control = DOC_NATIVE_BRM.emit_rk_artifact(
        brm(DOC_NATIVE_BRM._brm_strip_docs(source); mod=@__MODULE__)(data);
        case_id="plain-plate")
    @test DOC_NATIVE_BRM._brm_strip_docs(artifact.ast) == control.ast
    @test DOC_NATIVE_BRM._brm_strip_docs(Expr(:block, artifact.defs...)) ==
        Expr(:block, control.defs...)
    records = rkppl_documentation(rk_translate_artifact(artifact))
    @test Set(d.markdown for d in records) ==
        Set(["Global slope.", "Cell mean.", "Cell output.", "Cell response."])
    @test count(d -> d.markdown == "Cell mean.", records) == 1
end

@testset "Outer Julia documentation reaches reusable RKPPL definitions" begin
    declaration = :(theta ~ Normal(0, 1))
    local_doc = BRMDocumentation((:theta,), :theta, "Component prior.",
        declaration, :authored_component, 11)
    definition = :(piece() = begin $declaration; return theta; end)
    definition.args[2] = DOC_NATIVE_BRM._rk_documentation_block(
        definition.args[2], (local_doc,))
    header = BRMDocumentation((), :piece, "Reusable component.",
        definition, :authored_component, 9)
    documented = DOC_NATIVE_BRM._brm_documented_statement(definition, (header,))
    main = :(begin q ~ piece(); y .~ Normal.(q, 1); end)
    program = DOC_NATIVE_BRM._RKEmittedProgram([documented], main)
    mod = DOC_NATIVE_EXT._rk_emit_module(program)
    bound = bind_data(lower_rkppl(main, (; y=[.1, .3]); mod, conditioned=(:y,)),
        (; y=[.1, .3]))
    records = rkppl_documentation(bound)
    @test any(d -> d.path == (:q,) && d.markdown == "Reusable component.", records)
    @test any(d -> d.path == (:q, :theta) && d.markdown == "Component prior.", records)
    @test any(d -> d.definition == :piece && d.origin == (:theta,), records)
    @test any(d -> d.file == :authored_component && d.line == 11, records)
end
