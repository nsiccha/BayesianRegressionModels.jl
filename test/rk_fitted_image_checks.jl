# Runs in a fresh Julia after loading RKImageWorkload's image. The workload
# exercised the fitted constructor and discarded its state and backend.
@testset "open two fits of an image's regression" begin
    W = RKImageWorkload
    BRM = BayesianRegressionModels
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    data = (; x=[0., 1., 2.], y=[2., 1., 4.])
    train(shift) = RKBRMI(W.template((; x=data.x .+ shift, data.y)))
    first_fit, second_fit = train(0), train(10)
    states = brm_fitted_state.((first_fit, second_fit))
    design = (; x=reverse(data.x), y=reverse(data.y))
    mean_prediction(rk) = begin
        bound = rk_bound_data(rk)
        query = ReactiveKernels.prepare(rk.model.spec;
            have=(:unconstrained, keys(bound)...), want=(:mu,), bound)
        u = collect(range(-0.6, 0.8; length=rk.model.layout.total))
        result = Base.invokelatest(query, u)
        result isa Tuple ? only(result) : result
    end

    a, b = fetch.([Threads.@spawn RKBRMI(W.template(design), state) for state in states])
    @test parentmodule(ext._rk_program(a).mod) === W
    @test ext._rk_program(b).mod === ext._rk_program(a).mod
    @test ext._rk_program(a).captured !== ext._rk_program(b).captured
    @test rk_bound_plan(a) !== rk_bound_plan(b)
    for (rk, fit, state) in zip((a, b), (first_fit, second_fit), states)
        @test mean_prediction(rk) == mean_prediction(reprocess(fit, design))
        @test BRM._rk_layout_coordinate_names(rk) == state.coordinates
        @test mean_prediction(RKBRMI(W.template(data), state)) == reverse(mean_prediction(rk))
    end
    @test mean_prediction(a) != mean_prediction(b)
    other = replace(W.FORMULA, "center(x)" => "x")
    builder = brm(other; mod=W)
    own = brm_fitted_state(RKBRMI(builder(data)))
    @test parentmodule(ext._rk_program(RKBRMI(builder(data), own)).mod) === Main
    @test_throws "another model" RKBRMI(builder(data), first(states))

    # Changing a returned builder's mutable expressions cannot change its
    # sibling or the image. Identical syntax resolves names in its own caller.
    cached() = brm(W.FORMULA; mod=W)
    first_builder = cached()
    expected_source = deepcopy(first_builder.source)
    empty!(first_builder.source.args)
    empty!(first_builder.f.body.args)
    @test cached().source == expected_source
    @test !isempty(cached().f.body.args)
    shadow = Module(:ReplayShadow)
    Core.eval(shadow, :(const Normal = 42))
    @test_throws "cannot be called" brm(W.FORMULA; mod=shadow)(data)

    # Both exports spell the call identically. Only First's function was
    # exercised in the workload; the other module must bind its own function.
    first_call, second_call = map((W.First, W.Second)) do scope
        model = brm(W.CALLABLE; mod=scope)(data)
        RKBRMI(model, brm_fitted_state(RKBRMI(model)))
    end
    @test first_call.fitted.state.program == second_call.fitted.state.program
    @test parentmodule(ext._rk_program(first_call).mod) === W
    @test parentmodule(ext._rk_program(second_call).mod) === Main
    @test mean_prediction(second_call) ≈ mean_prediction(first_call) .+ data.x

    table = BRM._brm_image_table(W)
    @test any(entry -> BRM._brm_same_syntax(entry.source, BRM._brm_source_block(W.FORMULA)), table.builders)
    @test !any(entry -> BRM._brm_same_syntax(entry.source, BRM._brm_source_block(other)), table.builders)
    previous = W.EXPANSIONS[]
    brm(W.WITH_MACRO; mod=W)
    brm(W.WITH_MACRO; mod=W)
    @test W.EXPANSIONS[] == previous + 2
    @test !any(entry -> BRM._brm_same_syntax(entry.source, BRM._brm_source_block(W.WITH_MACRO)), table.builders)
    @test !BRM._brm_same_syntax(:(f(1)), :(f(1.0)))
    @test !BRM._brm_image_syntax(:(f($(data.x))))
    @test !ext._rk_image_callable(let captured = data.x; () -> captured end)
end
