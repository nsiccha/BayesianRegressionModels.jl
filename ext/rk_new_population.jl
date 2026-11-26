# Sample the level-local declarations of the emitted model, then provide them
# as HAVE to its own group-effect calculation. Thus known and inferred scales,
# correlations and the Student-t row scaling follow exactly the fitted graph.
function _rk_population_sites(binder)
    inputs = binder.level_inputs
    groups = Tuple(name for (name, input) in pairs(inputs) if input.kind === :group_effects)
    locals = Dict{Symbol,Tuple}()
    if !isempty(groups)
        for r in BRM._rk_coordinate_records(binder.rk; adaptive=true)
            r.kind in (:ranef, :mm_ranef, :r2d2_ranef) || continue
            block = BRM._rk_record_root(r.z)
            block in groups || continue
            locals[block] = haskey(r, :mixing) ? (r.z, r.mixing) : (r.z,)
        end
    end
    declarations = Tuple(site => input.columns for (name, input) in pairs(inputs)
        for site in get(locals, name, (name,)))
    sites, columns = map(first, declarations), map(last, declarations)
    (; groups, sites, columns, locals)
end

# A fresh local generator makes the kernel a pure function of its inputs.
# SHA supplies a stable namespace across processes. Keep the absolute position
# in a separate word so a chunk or a different scheduling order is immaterial.
function _rk_population_key(seed, columns, site)
    digest = BRM.sha256(join((seed, columns..., site), '\0'))
    foldl((key, byte) -> (key << 8) | byte, @view(digest[1:8]); init=UInt64(0))
end

function _rk_level_sample(key, position, shape, law, a, b)
    rng = BRM.Random.Xoshiro(UInt128(key) << 64 | UInt128(position))
    rand(rng, law(a, b), shape)
end

# The RHSs of the lowered array priors already carry the fitted scalar/data
# names (and the emitted expressions for their arguments, e.g. nu / 2).
# Author one RK fragment with distinct outputs so the fit's old z or w can
# never be selected as the recipe for a fresh draw.
function _rk_population_graph(binder)
    rk = binder.rk
    (; groups, sites, columns, locals) = _rk_population_sites(binder)
    plan = BRM.rk_bound_plan(rk)
    priors = Dict(p.name => p for declarations in
        (plan.array_parameters, plan.plate_parameters, plan.vector_parameters)
        for p in declarations if p.name in sites)
    spec = rk.model.spec
    taken = Set{Symbol}(keys(spec.ports))
    fresh(stem) = BRM._rk_ast_fresh_name(stem, taken)
    position = fresh("brm_draw_position")
    keys_port = fresh("brm_stream_keys")
    shapes_port = fresh("brm_level_shapes")
    fresh_sites = Tuple(fresh("brm_new_" * string(site)) for site in sites)
    parents = Set{Symbol}()
    body = Expr(:block)
    for (i, (site, out)) in enumerate(zip(sites, fresh_sites))
        prior = priors[site]
        law = if prior.family === :normal
            GlobalRef(BRM, :Normal)
        elseif prior.family === :inverse_gamma
            GlobalRef(BRM, :InverseGamma)
        else
            throw(ArgumentError("unsupported new-level law $(prior.family) for `$site`"))
        end
        args = values(prior.args)
        for arg in args
            union!(parents, intersect(BRM._rk_source_symbols!(Set{Symbol}(), arg), keys(spec.ports)))
        end
        call = Expr(:call, GlobalRef(@__MODULE__, :_rk_level_sample),
            :($keys_port[$i]), position, :($shapes_port[$i]), law, args...)
        push!(body.args, Expr(:(=), out, call))
    end
    arguments = Any[position, keys_port, shapes_port]
    for name in sort!(collect(parents); by=string)
        push!(arguments, Expr(:(::), name, ReactiveKernels.valtype(spec.ports[name])))
    end
    definition = Expr(:(=), Expr(:call, :brm_new_levels, arguments...), body)
    mod = _rk_new_module()
    for (name, value) in BRM._rk_retained_program(rk).bindings
        Core.eval(mod, Expr(:const, Expr(:(=), name, QuoteNode(value))))
    end
    noise = Core.eval(mod, Expr(:macrocall, GlobalRef(ReactiveKernels, Symbol("@kernel")),
        LineNumberNode(0), definition))
    (; spec=ReactiveKernels.compose(spec, noise), groups, sites, columns, locals,
        position, keys_port, shapes_port, fresh_sites)
end

# RK's single-output reader may return its array directly.
_rk_population_outputs(names, x) = NamedTuple{names}(x isa Tuple ? x : (x,))

function BRM.rk_new_population_effects(binder::BRM.RKDesignBinder, design,
        U::AbstractMatrix; seed::Integer, draw_positions=axes(U, 2))
    plan = BRM._rk_design_plan(binder, design)
    levels = binder.levels
    BRM._rk_check_fit_coordinates(levels, plan; prefix="rk_new_population_effects")
    isempty(levels.uncovered) || throw(ArgumentError(
        "rk_new_population_effects: no new-level reader inputs for $(join(levels.uncovered, ", "))"))
    positions = collect(draw_positions)
    positions isa AbstractVector || throw(ArgumentError("draw positions must form a vector"))
    length(positions) == size(U, 2) || throw(DimensionMismatch(
        "pass one draw position for each of the $(size(U, 2)) draw columns"))
    all(d -> d isa Integer && 0 < d <= typemax(UInt64), positions) ||
        throw(ArgumentError("draw positions must be positive integers representable as UInt64"))
    isempty(binder.level_inputs) && return (;)
    size(U, 1) == length(levels.fit_names) || throw(DimensionMismatch(
        "expected $(length(levels.fit_names)) fitted coordinates, got $(size(U, 1))"))

    graph = BRM._rk_once!(() -> _rk_population_graph(binder), binder.population)
    # The design layout sizes only the fresh draws. The fit layout still
    # unpacks U, including for a fitted state restored without training rows.
    shapes = Dict(Symbol(join(p.path, ".")) => p.shape
        for p in parameter_metadata(assign_layout(plan)))
    names = keys(binder.level_inputs)
    if isempty(positions)
        return NamedTuple{names}(map(names) do name
            site = name in graph.groups ? first(graph.locals[name]) : name
            Array{Float64}(undef, shapes[site]..., 0)
        end)
    end
    bound = merge(BRM._rk_plan_columns(plan), levels.values)
    settings = NamedTuple{(graph.keys_port, graph.shapes_port)}((
        map((cols, site) -> _rk_population_key(seed, cols, site), graph.columns, graph.sites),
        map(s -> shapes[s], graph.sites)))
    sampling = ReactiveKernels.prepare(graph.spec;
        have=(:unconstrained, graph.position, keys(bound)..., keys(settings)...),
        want=graph.fresh_sites, bound=merge(bound, settings))
    draw = ReactiveKernels.vectorize(sampling; batched=(:unconstrained, graph.position))
    noise = _rk_population_outputs(graph.sites, Base.invokelatest(draw, U, UInt64.(positions)))
    effects = if isempty(graph.groups)
        (;)
    else
        innovations = Tuple(s for group in graph.groups for s in graph.locals[group])
        transform = ReactiveKernels.prepare(binder.rk.model.spec;
            have=(:unconstrained, innovations..., keys(bound)...), want=graph.groups, bound)
        read = ReactiveKernels.vectorize(transform; batched=(:unconstrained, innovations...))
        _rk_population_outputs(graph.groups, Base.invokelatest(read, U, (noise[s] for s in innovations)...))
    end
    NamedTuple{names}(map(n -> haskey(effects, n) ? effects[n] : noise[n], names))
end
