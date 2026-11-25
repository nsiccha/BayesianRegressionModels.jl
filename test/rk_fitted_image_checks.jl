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
        u = collect(range(-0.6; step=0.2, length=rk.model.layout.total))
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

    # Independent opens of a runtime miss converge on one definition, even
    # across tasks. Every public graph and binding belongs to that open.
    opens = fetch.([Threads.@spawn RKBRMI(builder(design), own) for _ in 1:4])
    @test parentmodule(ext._rk_program(first(opens)).mod) === Main
    @test all(rk -> ext._rk_program(rk).mod === ext._rk_program(first(opens)).mod, opens)
    @test all(rk -> mean_prediction(rk) == mean_prediction(first(opens)), opens)
    @test first(opens).model.spec.ports !== last(opens).model.spec.ports
    @test rk_bound_plan(first(opens)) !== rk_bound_plan(last(opens))
    function discarded_binding()
        rk = RKBRMI(builder(design), deepcopy(own))
        WeakRef(rk.bound), WeakRef(rk.fitted.state.fits)
    end
    binding, fits = discarded_binding()
    GC.gc(true)
    @test isnothing(binding.value) && isnothing(fits.value)

    # The generated constant's module and type identify code. Its scalar
    # value comes from the current fit, including in a fresh image process.
    constant_builder = brm(W.WITH_CONSTANT; mod=W)
    for scale in (2.0, 5.0)
        fit = RKBRMI(constant_builder(merge(data, (; scale))))
        reopened = RKBRMI(constant_builder(design), brm_fitted_state(fit))
        @test parentmodule(ext._rk_program(reopened).mod) === W
        @test mean_prediction(reopened) == mean_prediction(reprocess(fit, design))
    end

    grouped = brm("sigma ~ Exponential(1); mu ~ 1 + (1 | g); y ~ Normal(mu, sigma)"; mod=W)
    group_fits = map(([1, 2, 1], [1, 2, 3])) do g
        model = grouped((; g, data.y))
        fit = RKBRMI(model)
        state = brm_fitted_state(fit)
        rk = RKBRMI(model, state)
        @test BRM._rk_layout_coordinate_names(rk) == state.coordinates
        @test mean_prediction(rk) == mean_prediction(fit)
        rk
    end
    @test first(group_fits).model.layout.total + 1 == last(group_fits).model.layout.total

    # A provider can supply an existing package-owned graph to its emitted
    # kernel. Redirecting its output changes subsequent independent opens.
    child_template = W.RKImageModels.with_child
    child_fit = RKBRMI(child_template(data))
    child_state = brm_fitted_state(child_fit)
    reopen() = RKBRMI(child_template(design), child_state)
    image_constructors = IdDict(def.code.value.builds => def.code.value.builds.entry
        for def in BRM._brm_image_table(W).definitions)
    fixed = W.RKImageModels.fixed_child(data)
    fixed_state = brm_fitted_state(RKBRMI(fixed))
    fixed_rk = RKBRMI(fixed, fixed_state)
    fixed_prediction = mean_prediction(fixed_rk)
    fixed_builds = ext._rk_program(fixed_rk).builds
    @test parentmodule(ext._rk_program(fixed_rk).mod) === W
    @test !isnothing(fixed_builds.entry) && fixed_builds.entry === image_constructors[fixed_builds]
    before = reopen()
    @test parentmodule(ext._rk_program(before).mod) === W
    builds = ext._rk_program(before).builds
    # RKPPL 0.1.260 builds the positive-scale graph fresh. BRM still
    # shares its definitions and observes a child edit on the next open.
    @test !isnothing(builds)
    @test ext._rk_program(reopen()).builds === builds
    # A provider returning a graph created for this call must not root it
    # in a package image or in the runtime definitions table.
    code = Dict{Symbol,Any}()
    BRM._rk_record_provider_binding!(code, :package_shift, before.model.spec,
        (Expr[], Pair{Symbol,Any}[], :shift_parameter, W.RKImageModels.shift_parameter))
    @test isempty(code)
    expected_child = mean_prediction(before)
    @test mean_prediction(reprocess(child_fit, design)) == expected_child
    child = W.RKImageModels.shifted_input
    original = child.ports[:z]
    try
        child.ports[:z] = child.ports[:a]
        changed = reopen()
        @test ext._rk_program(changed).mod !== ext._rk_program(before).mod
        @test mean_prediction(changed) ≈ expected_child .- 1
        @test mean_prediction(reprocess(before, design)) ≈ expected_child .- 1
        @test mean_prediction(before) == expected_child
        fixed_changed = RKBRMI(fixed, fixed_state)
        @test !isnothing(ext._rk_program(fixed_changed).builds.entry)
        @test mean_prediction(fixed_changed) ≈ fixed_prediction .- 1
        @test mean_prediction(fixed_rk) == fixed_prediction
    finally
        child.ports[:z] = original
    end

    cell_builder = brm(W.WITH_CELL; mod=W)
    cell_fit = RKBRMI(cell_builder(merge(data, (; scale=5.0))))
    cell_open = RKBRMI(cell_builder(design), brm_fitted_state(cell_fit))
    @test parentmodule(ext._rk_program(cell_open).mod) === W
    @test mean_prediction(cell_open) == mean_prediction(reprocess(cell_fit, design))

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
    @test !ext._rk_image_definition(:(function native(x::choose_type())
        x
    end))
    @test !ext._rk_image_callable(let captured = data.x; () -> captured end)

    # Touch the oldest entry, then insert one more. Only the second-oldest
    # should miss. An executing backend still owns its definition on eviction.
    entries = [BRM._rk_definition_slot("limit-test-$i", Pair{Symbol,Any}[])
        for i in 1:BRM._RK_RECENT_LIMIT]
    @test BRM._rk_definition_slot("limit-test-1", Pair{Symbol,Any}[]) === first(entries)
    BRM._rk_definition_slot("limit-test-extra", Pair{Symbol,Any}[])
    @test BRM._rk_definition_slot("limit-test-1", Pair{Symbol,Any}[]) === first(entries)
    @test BRM._rk_definition_slot("limit-test-2", Pair{Symbol,Any}[]) !== entries[2]
    @test length(BRM._RK_RECENT_DEFINITIONS) == BRM._RK_RECENT_LIMIT
    expected = mean_prediction(first(opens))
    delete!(first(opens).model.spec.ports, :mu)
    @test mean_prediction(last(opens)) == expected
end
