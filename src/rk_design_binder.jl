# src/rk_design_binder.jl — the graph inputs of new designs for a fitted
# RKBRMI, built once per fit.
#
# `rk_bound_data(reprocess(rk, design))` plans, emits and lowers the whole
# program again for every design, although none of that depends on the
# design's values: everything a design changes is its raw columns, which the
# fitted program reads directly (its preprocessing, level indices and ragged
# row partitions are graph values over them, with fitted constants read from
# fit-role inputs). A binder replays the first design of each kind in full —
# the ordinary `reprocess`, which validates it — and records where each graph
# input comes from: a column of the design, or a value no design changes.
# Every later design of that kind is checked as `reprocess` checks its values,
# and its raw columns are bound into the program lowered once for that kind,
# with the values fitted from fixed inputs alone frozen as that replay bound
# them.
#
# An input the planner derives from the design in Julia (a value BRM has not
# moved into the graph, or data a custom hook writes) cannot be recomputed
# without planning; designs of such a model take the planned replay, which
# `show(binder)` names.

"""
    rk_design_binder(rk::RKBRMI; resample_groups=(), held_out=<the fit's>) -> RKDesignBinder

The graph inputs of new designs (prediction scenarios) for the fitted model
`rk`, built once, at a draw of the fit. `binder(design)` returns the inputs
`rk_bound_data(reprocess(rk, design; resample_groups, held_out))` holds, as a
`NamedTuple`, without planning, emitting or lowering the program again: the
first design of each kind (the same columns, column types, empty columns and
missing values) is replayed in full, and every later one is checked and bound
into that replay's lowered program.

- `resample_groups=[:g]` (new levels of `g`): the inputs keep the fit's level
  set of `g` (`g_fit_levels`), which sizes `g`'s standardized draws as in the
  fit, while the rows read the design's own levels by position (`g_level`). So
  `u` is a draw of the fit for any number of new levels, and the new levels'
  effects come in HAVE (`have=(:unconstrained, :b_g, ...)`, one row per new
  level in level order). This is where `binder(design)` differs from
  `rk_bound_data(reprocess(...))`: that replay is a model of its own rows, whose
  new levels' standardized draws are its own coordinates. A block whose draws
  the design's own columns size (`gr(g, by = s)`, or a multi-membership block of
  a data-free fit) is refused on a design that gives it other levels than the
  fit's.
- `binder.fitted` holds the inputs no design changes (the training columns
  behind fitted anchors, a `car(...)` term's fitted graph, `constant(...)`
  values, a resampled grouping's fitted level set). Bind them once with
  `ReactiveKernels.prepare(spec; have, want, bound=binder.fitted)` and pass the
  rest per design; values that read only them are then computed once.
- A numeric scalar a design omits (a prior argument the fitted state carries)
  takes the fit's value, as on `reprocess`, and stays on the cached path. It is
  among `binder(design)`'s inputs, not `binder.fitted`: a design of another kind
  may hold it.
- What the program fits from those inputs alone and binding evaluates (spline
  and HSGP fits, fitted means and SDs, level sets) is taken from the first
  design of each kind and bound frozen into every later one (RKPPL's
  `bind_data(…; frozen)`): a later design computes only what its own columns
  determine, such as a basis at its rows.
- A design is refused exactly where `reprocess` refuses its values: unseen
  factor, `mo` or group levels, HSGP axis values past the fitted basis
  interval or fixed domain, non-finite HSGP axis values, a `rows=` column
  that is not 0/1 or not as long as the axis, a `missing` axis value in a row
  the mask carries, a `car` area outside the fitted graph, ragged labels that
  name no cell, and per-cell inputs whose cell counts disagree, and a numeric
  scalar a prior reads or the program holds as a literal, at another value.
- Checks of data-only model assignments read their values from the bound
  program, so derived axes, grouping keys and masks use the same cached path.
- A model whose inputs include a value BRM derives from the design in Julia
  replays every design in full; `show(binder)` names those inputs.
"""
function rk_design_binder end

struct RKDesignBinder{B<:RKBRMI}
    rk::B
    resample_groups::Tuple{Vararg{Symbol}}
    held_out::Tuple{Vararg{Symbol}}
    # The fit's data columns: what a design's kind is read from.
    columns::Vector{Symbol}
    fitted::NamedTuple
    # The fit's level sets of the resampled groupings, which keep a draw of the
    # fit in the fit's coordinates (`_rk_fit_coordinate_levels`).
    levels::Any
    classes::Dict{Any,Any}
    lock::ReentrantLock
end

# Where a graph input of a design comes from.
struct _RKFixedInput
    value::Any
end
struct _RKDesignInput
    column::Symbol
end

# Raw-column checks run before binding. Checks that read data-only definitions
# run on their bound graph values, including definitions preparation folds
# instead of materializing as bound columns.
struct _RKDesignChecks
    raw::Vector{Any}
    bound::Vector{Any}
    columns::Vector{Pair{Symbol,Symbol}}
end
_RKDesignChecks() = _RKDesignChecks(Any[], Any[], Pair{Symbol,Symbol}[])

# One kind of design: how its inputs are read, the program lowered for it, the
# value checks, and the definitions it binds frozen (RKPPL's `FrozenValues`, or
# `nothing`). `derived` names the inputs that force a planned replay. `scalars`
# are the numeric scalars the kind holds whose values its cached lowering cannot
# take from a later design (`_rk_design_kept_scalars`).
struct _RKDesignClass
    sources::Vector{Pair{Symbol,Any}}
    lowering::Any
    checks::_RKDesignChecks
    derived::Vector{Symbol}
    frozen::Any
    scalars::Vector{Pair{Symbol,Any}}
end
_RKDesignClass(derived::Vector{Symbol}) = _RKDesignClass(Pair{Symbol,Any}[], nothing,
    _RKDesignChecks(), derived, nothing, Pair{Symbol,Any}[])
_rk_design_planned(class::_RKDesignClass) = !isempty(class.derived)

function rk_design_binder(rk::RKBRMI; resample_groups=(),
        held_out=_rk_held_out_names(rk.plan))
    prefix = "RK design binder"
    groups = _rk_resample_group_set(resample_groups; prefix)
    training = _rk_fit_column_names(rk)
    unknown = setdiff(groups, training)
    isempty(unknown) || error(
        "$prefix: `resample_groups` names $(sort!(collect(unknown))), which " *
        "is not a data column of the fitted model")
    groups = Tuple(sort!(collect(groups)))
    levels = _rk_fit_coordinate_levels(rk, groups)
    RKDesignBinder(rk, groups, Tuple(held_out), sort!(collect(training)),
        merge(_rk_design_fitted(rk, groups), levels.values), levels,
        Dict{Any,Any}(), ReentrantLock())
end

# The data columns of the fit `rk` replays: its own, or on a data-free backend
# those of the fit its state was exported from.
_rk_fit_column_names(rk::RKBRMI) = rk.fitted === nothing ?
    Set{Symbol}(keys(_rk_training_columns(parent(rk)))) :
    Set{Symbol}(keys(_rk_fitted_state(rk).columns))
_rk_fit_constant_names(rk::RKBRMI) = rk.fitted === nothing ?
    Set{Symbol}(c.name for c in last(_brm_resolve_constants(parent(rk)))) :
    Set{Symbol}(keys(_rk_fitted_state(rk).constants))

function (binder::RKDesignBinder)(design)
    plan = _rk_design_plan(binder, design)
    _rk_check_fit_coordinates(binder.levels, plan; prefix="RK design binder")
    merge(_rk_plan_columns(plan), binder.levels.values)
end

# A draw of the fit on a design that resamples groupings. The design's binding
# sizes each resampled block's standardized draws (`b_g.z`) by the design's
# levels, and RKPPL unpacks `:unconstrained` with the binding's layout (a query
# reads its binding's layout, never its build's). So a cut at a draw of the fit
# needs the block sized by the fit's levels. That size is the block's level
# set, a data-only definition `g_fit_levels = brm_fit_levels(g_fit)`, or
# `b_mm_groups = vcat(g1, g2)` for multi-membership. Its value is the first
# argument of the block's component, and every other read of it is a data-only
# definition that the inputs already hold, such as the rows' positions
# `g_level`. With the fit's value in its place, the block keeps the fit's
# coordinates and the rows keep the design's level positions. The caller
# supplies the new levels' effects (`b_g`) in HAVE, as with any number of new
# levels. A `gr(g, by = s)` block reads its sizes from the raw grouping columns
# themselves, and a data-free fit holds no multi-membership level set. Those
# blocks are uncovered: a design whose levels give them other coordinates than
# the fit's is refused.
struct _RKFitCoordinateLevels
    values::NamedTuple
    sized::Set{Symbol}
    uncovered::Vector{String}
    fit_names::Vector{Symbol}
end

function _rk_fit_coordinate_levels(rk::RKBRMI, groups)
    none = _RKFitCoordinateLevels((;), Set{Symbol}(), String[], Symbol[])
    isempty(groups) && return none
    statements = [_brm_undoc(s) for s in _rk_retained_program(rk).main.args
        if !(s isa LineNumberNode)]
    definitions = Dict{Symbol,Any}(s.args[1] => s.args[2] for s in statements
        if Meta.isexpr(s, :(=), 2) && s.args[1] isa Symbol)
    twins = _rk_plan_fit_inputs(rk.plan)
    inputs = Set{Symbol}(keys(rk_bound_data(rk)))
    values = Pair{Symbol,Any}[]
    uncovered = String[]
    for bucket in _rk_replay_buckets(rk.plan)
        grouping = bucket.grouping
        columns = Symbol[grouping.columns...]
        grouping.by === nothing || push!(columns, grouping.by)
        any(in(groups), columns) || continue
        name = _rk_bucket_level_set(grouping, definitions, twins)
        value = name === nothing || !_rk_sizes_only(statements, name, inputs) ? nothing :
            _rk_fitted_level_set(rk, name, definitions[name], twins)
        if value === nothing
            push!(uncovered, string("the `", grouping.form, "(...)` block over ",
                join(("`$c`" for c in columns), ", ")))
        else
            any(p -> first(p) === name, values) || push!(values, name => only(value))
        end
    end
    isempty(values) && isempty(uncovered) && return none
    levels = NamedTuple{Tuple(first.(values))}(Tuple(last.(values)))
    sized = Set{Symbol}(_rk_record_root(s.args[2]) for s in statements
        if Meta.isexpr(s, :call, 3) && s.args[1] === :~ && Meta.isexpr(s.args[3], :call) &&
            length(s.args[3].args) >= 2 && s.args[3].args[2] in keys(levels))
    fit_names = rk.fitted === nothing ? Vector{Symbol}(_rk_layout_coordinate_names(rk)) :
        copy(_rk_fitted_state(rk).coordinates)
    _RKFitCoordinateLevels(levels, sized, uncovered, fit_names)
end

# The data-only definition that sizes a block's standardized draws: a plain
# grouping's fitted level set (`brm_fit_levels` of its fit-role input), or a
# multi-membership block's joint labels (`vcat` of its columns).
function _rk_bucket_level_set(grouping, definitions, twins)
    if grouping.form === :plain && !_rk_joins_columns(grouping)
        column = only(grouping.columns)
        for (name, rhs) in definitions
            Meta.isexpr(rhs, :call, 2) && rhs.args[1] === :brm_fit_levels &&
                get(twins, rhs.args[2], nothing) === column && return name
        end
    elseif grouping.form === :mm
        joint = Expr(:call, :vcat, grouping.columns...)
        for (name, rhs) in definitions
            isequal(rhs, joint) && return name
        end
    end
    nothing
end

# Whether the level set `name` only sizes blocks: it is the first argument of
# each component that reads it, and every other statement reading it is a
# data-only definition the inputs hold (the rows' level positions), which a cut
# takes as given rather than computing again from the level set.
function _rk_sizes_only(statements, name, inputs)
    reads(x) = name in _rk_source_symbols!(Set{Symbol}(), x)
    for s in statements
        if Meta.isexpr(s, :call, 3) && s.args[1] === :~
            rhs = s.args[3]
            reads(rhs) || continue
            Meta.isexpr(rhs, :call) && length(rhs.args) >= 2 && rhs.args[2] === name &&
                !reads(rhs.args[3:end]) || return false
        elseif Meta.isexpr(s, :(=), 2) && s.args[1] isa Symbol
            s.args[1] === name || !reads(s.args[2]) || s.args[1] in inputs || return false
        elseif reads(s)
            return false
        end
    end
    true
end

# The fit's value of the level set `name`, as `(value,)`, or `nothing`. A
# data-free fit holds it among its frozen values. A data-bound backend's own
# binding holds it when every input its definition reads is the training
# column; a backend that resampled them itself does not hold the fit's.
function _rk_fitted_level_set(rk::RKBRMI, name, rhs, twins)
    if rk.fitted !== nothing
        frozen = _rk_fitted_state(rk).fitted.values
        return haskey(frozen, name) ? (frozen[name],) : nothing
    end
    bound = rk_bound_data(rk)
    haskey(bound, name) || return nothing
    training = _rk_frozen_training(parent(rk))
    for input in _rk_source_symbols!(Set{Symbol}(), rhs)
        haskey(bound, input) || continue
        column = get(twins, input, input)
        haskey(training, column) && isequal(bound[input], training[column]) || return nothing
    end
    (bound[name],)
end

# A design's binding against the fit's coordinates: every coordinate the level
# sets do not size must be the fit's, so a block they do not cover (a
# `gr(g, by = s)` block) must have the fit's level counts on this design.
function _rk_check_fit_coordinates(levels::_RKFitCoordinateLevels, plan; prefix)
    isempty(levels.uncovered) && return nothing
    outside(names) = [n for n in names if !(_rk_record_root(n) in levels.sized)]
    design = outside(_rk_plan_coordinate_names(plan))
    fitted = outside(levels.fit_names)
    design == fitted && return nothing
    counts(names) = foldl((d, n) -> (d[n] = get(d, n, 0) + 1; d), _rk_record_root.(names);
        init=Dict{Symbol,Int}())
    a, b = counts(design), counts(fitted)
    roots = sort!([r for r in union(keys(a), keys(b)) if get(a, r, 0) != get(b, r, 0)]; by=string)
    isempty(roots) && (roots = sort!(collect(keys(a)); by=string))
    error("$prefix: this design gives $(join(("`$r`" for r in roots), ", ")) other " *
        "coordinates than the fit ($(length(design)) on the design, $(length(fitted)) " *
        "fitted). $(uppercasefirst(join(levels.uncovered, "; "))) sizes its draws by the design's own " *
        "columns, which no fitted level set replaces, so a draw of the fit does not apply " *
        "to this design. Predict it on a design with the fit's levels of those columns, or " *
        "evaluate `reprocess(rk, design; resample_groups)`, whose coordinates are the " *
        "design's own.")
end

# The coordinate names a bound plan packs (ReactiveKernels package extension).
function _rk_plan_coordinate_names end

# A design's RKPPL plan: bound into the lowered program of its kind, or the
# planned replay's own binding for the first design of a kind and for a kind
# whose inputs BRM derives in Julia.
function _rk_design_plan(binder::RKDesignBinder, design)
    key = _rk_design_key(binder, design)
    class = lock(() -> get(binder.classes, key, nothing), binder.lock)
    class === nothing && return _rk_design_first!(binder, key, design)
    _rk_design_planned(class) && return rk_bound_plan(_rk_design_replay(binder, design))
    all(scalar -> isequal(_brm_df_column(design, first(scalar)), last(scalar)), class.scalars) ||
        return rk_bound_plan(_rk_design_replay(binder, design))
    for check in class.checks.raw
        check(design)
    end
    columns = Dict{Symbol,Any}(name => _rk_design_input(source, design)
        for (name, source) in class.sources)
    plan = try
        _rk_bind_design_plan(class.lowering, columns, class.frozen)
    catch err
        # Binding may read a derived level index before its validation value
        # is available. Replay a failed binding to report the planner's public
        # refusal (or bind a value shape that requires a different plan).
        # Successful designs still use only the cached lowering.
        err isa InterruptException && rethrow()
        isempty(class.checks.bound) && rethrow()
        return rk_bound_plan(_rk_design_replay(binder, design))
    end
    if !isempty(class.checks.bound)
        data = _BRMReplayData(design, _rk_design_check_values(plan, class.checks.columns))
        for check in class.checks.bound
            check(data)
        end
    end
    plan
end

_rk_design_replay(binder::RKDesignBinder, design) = reprocess(binder.rk, design;
    resample_groups=binder.resample_groups, held_out=binder.held_out)

_rk_design_input(source::_RKFixedInput, _design) = source.value
_rk_design_input(source::_RKDesignInput, design) = _brm_df_column(design, source.column)

# A design's kind: for each fitted data column, absent, or its type, whether it
# is empty, whether it holds missing values and whether it holds one array per
# index — every property of a value that lowering reads (rkppl-use §1) or that
# routes planning.
function _rk_design_key(binder::RKDesignBinder, design)
    Tuple(_brm_df_has_column(design, name) ?
        _rk_design_value_kind(name, _brm_df_column(design, name)) : (name, :absent)
        for name in binder.columns)
end
_rk_design_value_kind(name, value) = (name, typeof(value))
_rk_design_value_kind(name, value::AbstractArray) = (name, typeof(value), isempty(value),
    Missing <: eltype(value) && any(ismissing, value), _rk_design_holds_arrays(value))
_rk_design_holds_arrays(value::AbstractArray) = !(eltype(value) <: Number) &&
    (isempty(value) ? eltype(value) <: AbstractArray : all(x -> x isa AbstractArray, value))

function _rk_design_first!(binder::RKDesignBinder, key, design)
    replay = _rk_design_replay(binder, design)
    plan = rk_bound_plan(replay)
    class = _rk_design_class(binder, replay, design)
    lock(() -> get!(binder.classes, key, class), binder.lock)
    plan
end

# The fit-role inputs of a plan, by the column each one twins.
_rk_plan_fit_inputs(plan::_RKStructuralPlan) = _rk_fit_inputs(plan.derived)
function _rk_plan_fit_inputs(plan::_RKValuePlan)
    inputs = _rk_fit_inputs(plan.regression.derived)
    for assignment in plan.assignments
        assignment isa _RKPreparedSubmodelAssignment || continue
        merge!(inputs, assignment.fit_inputs)
    end
    inputs
end
_rk_plan_fit_inputs(plan::_RKHeldOutPlan) = _rk_plan_fit_inputs(plan.parent)
_rk_plan_fit_inputs(_plan) = Dict{Symbol,Symbol}()
function _rk_fit_inputs(derived)
    inputs = Dict{Symbol,Symbol}()
    for spec in derived
        Meta.isexpr(spec.expression, :_rk_fit_input) || continue
        twinned = _rk_fit_twinned(spec.label)
        twinned === nothing || (inputs[spec.name] = twinned)
    end
    inputs
end

_rk_plan_kernels(plan::_RKValuePlan) =
    Tuple(a for a in plan.assignments if a isa _RKPreparedKernelAssignment)
_rk_plan_kernels(plan::_RKHeldOutPlan) = _rk_plan_kernels(plan.parent)
_rk_plan_kernels(_plan) = ()

# An areal field's graph inputs, by its area column: the graph is a fit of the
# training areas and edges, refitted only when the areas are resampled
# (`_brm_car_fitted_graph`).
function _rk_plan_graph_inputs(plan)
    inputs = Dict{Symbol,Symbol}()
    for predictor in _rk_replay_predictors(plan), term in predictor.terms
        term.kind === :car || continue
        foreach(name -> inputs[name] = term.options.source, term.options.inputs)
    end
    inputs
end

# The inputs no design changes: fit-role inputs and fitted graphs of the
# columns that are not resampled, and fitted `constant(...)` values. A
# data-free backend binds no fit-role input: its frozen fitted values take
# their place.
function _rk_design_fitted(rk::RKBRMI, groups)
    columns = _rk_source_data_columns(rk.plan, _rk_retained_program(rk))
    # Each fitted input, by the column whose resampling refits it.
    fits = merge(_rk_plan_fit_inputs(rk.plan), _rk_plan_graph_inputs(rk.plan))
    constants = _rk_fit_constant_names(rk)
    frozen = rk.fitted !== nothing
    generated = _rk_generated_inputs(rk.plan)
    names = sort!([name for name in keys(columns) if name in constants ||
        haskey(generated, name) ||
        (!frozen && haskey(fits, name) && !(fits[name] in groups))])
    fitted = NamedTuple{Tuple(names)}(Tuple(columns[name] for name in names))
    frozen || return fitted
    state = _rk_fitted_state(rk)
    merge(fitted, _rk_frozen_subset(state, _rk_resampled_fit_inputs(state, groups)).values)
end

function _rk_design_class(binder::RKDesignBinder, replay::RKBRMI, design)
    rk = binder.rk
    # A provider that emits new callables per design binds its own program;
    # a plan whose value checks the binder cannot read replays in full.
    replay.program === rk.program || return _RKDesignClass([:callables])
    _rk_design_checked(replay.plan) || return _RKDesignClass([:plan])
    columns = _rk_source_data_columns(replay.plan, _rk_retained_program(rk))
    # A data-free replay binds its fitted values frozen, not the fit-role inputs
    # they are fitted from.
    replay.fitted === nothing ||
        foreach(name -> delete!(columns, name), keys(replay.fitted.frozen.inputs))
    twins = _rk_plan_fit_inputs(replay.plan)
    fitted = binder.fitted
    sources = Pair{Symbol,Any}[]
    derived = Symbol[]
    extents = Set(first(binding) for binding in _rk_extent_bindings(replay.plan))
    scalars = _rk_fitted_scalars(rk)
    for (name, value) in columns
        # A computed integer extent can equal an unrelated scalar data value.
        # Replan its authored expression; value equality is not its source.
        if name in extents
            push!(derived, name)
            continue
        end
        # A fitted input reads the training values on every replay.
        if haskey(fitted, name) && isequal(value, fitted[name])
            push!(sources, name => _RKFixedInput(fitted[name]))
            continue
        end
        # A scalar the design omits takes the fit's value on every replay
        # (`_rk_omitted_scalars`), and the kind records that it is absent. It
        # has no column to find, whatever other column holds an equal number.
        if haskey(scalars, name) && !_brm_df_has_column(design, name)
            push!(sources, name => _RKFixedInput(value))
            continue
        end
        column = _rk_design_column_of(binder, design, name, get(twins, name, nothing), value)
        column === nothing ? push!(derived, name) :
            push!(sources, name => _RKDesignInput(column))
    end
    checks = _rk_design_checks(binder, replay.plan, design, derived)
    isempty(derived) || return _RKDesignClass(sort!(derived))
    # What the program fits from the fixed inputs alone is the same for every
    # design: bound frozen, it is computed once, here, not on every binding.
    fixed = Set{Symbol}(name for (name, source) in sources if source isa _RKFixedInput)
    _RKDesignClass(sources, _rk_design_lowering(replay), checks, Symbol[],
        _rk_design_frozen(replay, fixed), _rk_design_kept_scalars(rk, replay, design, columns))
end

# The numeric scalars this design holds that a later design of its kind must not
# change on the cached path: those a prior reads, and those the program binds no
# input for (a literal, or an input of `constant(...)`). A later design holding
# another value replays in full, and `reprocess` refuses it or plans it.
function _rk_design_kept_scalars(rk, replay, design, columns)
    priors = _rk_prior_reads(_rk_retained_program(rk), _rk_response_names(replay.plan))
    Pair{Symbol,Any}[name => _brm_df_column(design, name) for name in keys(_rk_fitted_scalars(rk))
        if _brm_df_has_column(design, name) && (name in priors || !haskey(columns, name))]
end

_rk_design_checked(::Union{_RKStructuralPlan,_RKValuePlan}) = true
_rk_design_checked(plan::_RKHeldOutPlan) = _rk_design_checked(plan.parent)
_rk_design_checked(_plan) = false

# The design column an input is (by identity: the planner binds raw columns
# unchanged), or `nothing` when the planner derived it.
function _rk_design_column_of(binder, design, name, twin, value)
    for column in (something(twin, name), name, binder.columns...)
        _brm_df_has_column(design, column) || continue
        _brm_df_column(design, column) === value && return column
    end
    nothing
end

# The checks `reprocess` applies to a design's values, as closures over the
# planned terms. A term whose check cannot be read off the plan marks the kind
# as planned (`derived`).
function _rk_design_checks(binder::RKDesignBinder, plan, design, derived)
    checks = _RKDesignChecks()
    definitions = Dict(spec.label => spec.name for spec in _rk_plan_derived(plan)
        if _rk_fit_twinned(spec.label) === nothing &&
            Meta.isexpr(spec.expression, :_rk_data_assignment))
    has(column) = column !== nothing &&
        (haskey(definitions, column) || _brm_df_has_column(design, column))
    function add(check, columns)
        computed = [column => definitions[column] for column in columns
            if haskey(definitions, column)]
        if isempty(computed)
            push!(checks.raw, check)
        else
            push!(checks.bound, check)
            union!(checks.columns, computed)
        end
    end
    for predictor in _rk_replay_predictors(plan), term in predictor.terms
        options = term.options
        if term.kind === :hsgp && !get(options, :latent, false)
            if !all(has, _rk_design_hsgp_columns(term))
                push!(derived, Symbol(predictor.name, "#hsgp"))
                continue
            end
            add(_rk_design_hsgp_check(term), _rk_design_hsgp_columns(term))
            if get(options, :level_cor, nothing) isa _BRMMultigroupCoordinates
                required = (options.group_column, options.level_cor.axes...)
                if all(has, required)
                    add(_rk_design_multigroup_check(term, binder.resample_groups), required)
                else
                    push!(derived, Symbol(predictor.name, "#multigroup"))
                end
            end
        elseif term.kind === :factor && haskey(options, :index_levels)
            source = only(term.columns)
            has(source) || continue
            levels = options.index_levels
            add(design -> _brm_apply_fitted_levels(levels,
                _brm_df_column(design, source); prefix="RK reprocess"), (source,))
        elseif term.kind === :monotonic && haskey(options, :levels)
            source = options.source
            has(source) || continue
            levels = options.levels
            add(design -> _brm_apply_fitted_levels(levels,
                _brm_df_column(design, source); prefix="RK reprocess"), (source,))
        elseif term.kind === :car
            # Each row's area must be a node of the fitted graph.
            source = options.source
            has(source) || continue
            nodes = options.nodes
            add(design -> _brm_car_row_idx(nodes,
                _brm_car_labels(_brm_df_column(design, source), "area")), (source,))
        end
    end
    for kernel in _rk_plan_kernels(plan)
        _rk_design_cell_check!(add, kernel, has)
    end
    # Responses bind as the design holds them: an ordered outcome lies in the
    # fitted levels, and weights are checked as the planner checks them.
    weighted = Dict{Symbol,Symbol}(_rk_value_weight_sources(plan))
    for response in _rk_replay_responses(plan)
        key = response.response
        response.weights === nothing || (weighted[key] = response.weights)
        if response.family === :ordered_logit && has(key)
            K = response.n_levels
            add(design -> _rk_check_ordered_outcomes(key,
                _brm_df_column(design, key), K), (key,))
        end
    end
    for (key, K) in _rk_ordered_observation_levels(plan)
        has(key) || continue
        add(design -> _brm_ordered_outcome_levels(key,
            _brm_df_column(design, key), (; n_levels=K)), (key,))
    end
    isempty(weighted) || _rk_design_weight_checks!(add, binder, weighted, has)
    checks
end

# The level count of each `OrderedLogistic` observation with an authored
# location: one more than its implicit cutpoint vector's length.
_rk_ordered_observation_levels(plan::_RKHeldOutPlan) = _rk_ordered_observation_levels(plan.parent)
_rk_ordered_observation_levels(_plan) = Dict{Symbol,Int}()
function _rk_ordered_observation_levels(plan::_RKValuePlan)
    sizes = Dict{Symbol,Any}(v.name => v.size for v in plan.regression.vector_parameters)
    levels = Dict{Symbol,Int}()
    for observation in plan.observations
        observation isa _BRMPreparedObservation || continue
        law = observation.distribution
        law isa _BRMPreparedExpr && law.callable === OrderedLogistic || continue
        cutpoints = Symbol(observation.name, :_cutpoints)
        any(a -> a isa _BRMPreparedRef && a.name === cutpoints, law.args) || continue
        levels[observation.name] = sizes[cutpoints] + 1
    end
    levels
end

# A reader-route response keeps its formula weight column under the column's
# own name (`_rk_value_observation_weight`); in-cell weights are cell values.
_rk_value_weight_sources(plan::_RKHeldOutPlan) = _rk_value_weight_sources(plan.parent)
_rk_value_weight_sources(_plan) = Pair{Symbol,Symbol}[]
_rk_value_weight_sources(plan::_RKValuePlan) = Pair{Symbol,Symbol}[
    observation.name => observation.weight.name for observation in plan.observations
    if observation isa _BRMPreparedObservation && observation.weight isa _BRMPreparedRef]

# `weighted` maps each weighted response to its weight column. The weight kind
# is the likelihood's syntax (`fweights(n)`), read off the fitted model.
function _rk_design_weight_checks!(add, binder::RKDesignBinder, weighted, has)
    prefix = "RK backend"
    for observation in _brm_direct_observation_list(parent(binder.rk); prefix)
        target = observation.key
        source = get(weighted, target, nothing)
        source !== nothing && has(source) || continue
        # `weighted` is this function's Dict argument here, not the combinator.
        observation.rhs isa ExprColumn &&
            getf(observation.rhs) === BayesianRegressionModels.weighted || continue
        # The fitted plan already validated the weight source. Read only the
        # constructor here: the authored source may be a data-only assignment.
        kind = _brm_observation_weight_kind(getf(last(getargs(observation.rhs))))
        response = has(target) ? target : nothing
        add(design -> _brm_check_observation_weight_values(kind,
            _brm_df_column(design, source),
            response === nothing ? nothing : length(_brm_df_column(design, response)),
            target, source; prefix), response === nothing ? (source,) : (source, response))
    end
    nothing
end

# The design columns a raw-axis HSGP term reads: its axes and `rows=` column.
function _rk_design_hsgp_columns(term)
    source = get(term.options, :mask_source, nothing)
    source === nothing ? Tuple(term.columns) : (term.columns..., source)
end

# What `reprocess` checks of a raw-axis HSGP term's values: 0/1 `rows=` values
# as long as the axes (`_brm_hsgp_row_mask`), `missing` only in rows the mask
# does not carry (`_brm_axis_values`), finite axes and, unless the basis
# clamps, the carried values inside the fitted interval. A design's kind fixes
# which columns are empty or hold `missing`, so those refusals come from its
# first, planned replay.
function _rk_design_hsgp_check(term)
    options = term.options
    axes, source = Tuple(term.columns), get(options, :mask_source, nothing)
    periodic = get(options, :cov, :exp_quad) === :periodic
    interval = !periodic && get(options, :extrapolate, :error) === :error
    fixed_domain = get(options, :fixed_fits, nothing) !== nothing
    function check(design)
        raw = Tuple(_brm_df_column(design, axis) for axis in axes)
        mask = source === nothing ? nothing : _brm_hsgp_row_mask(
            _brm_df_column(design, source), length(first(raw)), source)
        values = mask === nothing ? raw : Tuple(_brm_axis_values(column, mask,
            "BRM backend lowering: data column `$axis`") for (axis, column) in zip(axes, raw))
        foreach(axis -> _brm_check_finite_axis(axis, :hsgp), values)
        interval && _brm_hsgp_replay_axes(values, options.fits, nothing, :error, mask;
            prefix="RK reprocess", names=axes, fixed_domain)
        nothing
    end
end

# A kernel's per-cell inputs and its cell label column must agree on the cell
# count, its cell labels must be unique and each ragged join's labels and axis
# lengths must hold, as `_rk_prepare_kernel_value` requires: a plate iterating
# a shorter input would silently drop cells. Each join is checked as the graph
# checks it when it computes the partition, without building the partition,
# which binding never reads: the graph computes it once the design is bound.
function _rk_design_cell_check!(add, kernel, has)
    per_cell = unique!(Symbol[input.column for input in kernel.inputs
        if input.kind === :element && has(input.column)])
    cells = has(kernel.cell_column) ? [kernel.cell_column] : Symbol[]
    joins = unique!([input.partition for input in kernel.inputs if input.kind === :gather &&
        get(input, :partition, nothing) !== nothing &&
        all(has, (input.partition.group, input.partition.cell, input.partition.axes...)) &&
        (input.partition.order === nothing || has(input.partition.order))])
    name = kernel.name
    function check(design)
        for cell in cells
            _brm_kernel_subject_values(_brm_df_column(design, cell), cell; prefix="RK backend")
        end
        lengths = Int[length(_brm_df_column(design, column)) for column in (cells..., per_cell...)]
        all(==(first(lengths)), lengths) || error(
            "RK backend: kernel `$name` positional inputs disagree on subject count")
        column(c) = _brm_df_column(design, c)
        for entry in joins
            _brm_check_ragged_rows(entry.x, entry.group, column(entry.group),
                entry.cell, column(entry.cell), entry.axes,
                Tuple(column(axis) for axis in entry.axes); entry.order,
                times=entry.order === nothing ? nothing : column(entry.order),
                observed=get(entry, :observed, false))
        end
        nothing
    end
    columns = unique!([cells; per_cell])
    for entry in joins
        union!(columns, (entry.group, entry.cell, entry.axes...))
        entry.order === nothing || union!(columns, (entry.order,))
    end
    isempty(columns) || add(check, columns)
    nothing
end

# Lowering, freezing and binding live in the ReactiveKernels extension.
function _rk_design_lowering end
function _rk_design_frozen end
function _rk_bind_design_plan end
function _rk_design_check_values end
function _rk_plan_columns end

function Base.show(io::IO, binder::RKDesignBinder)
    classes = lock(() -> collect(values(binder.classes)), binder.lock)
    planned = filter(_rk_design_planned, classes)
    print(io, "RKDesignBinder(", length(classes), " design kind",
        length(classes) == 1 ? "" : "s", ", ", length(binder.fitted), " fitted input",
        length(binder.fitted) == 1 ? "" : "s")
    isempty(binder.resample_groups) ||
        print(io, ", resample_groups=", collect(binder.resample_groups))
    print(io, ")")
    isempty(planned) && return
    derived = sort!(unique!(reduce(vcat, (c.derived for c in planned))))
    print(io, "\n  replayed in full for each design: the planner derives ",
        join(("`$d`" for d in derived), ", "), " from the design")
end
