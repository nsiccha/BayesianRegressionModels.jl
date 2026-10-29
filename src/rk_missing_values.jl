# Missing covariates retain their subject rows. The design describes their
# geometry and fixed observed-only anchors; executable columns use completed
# model values. Missing entries remain missing in geometry, never fake data.
struct _RKMissingValueSpec
    source::Symbol
    observed::Symbol
    observed_rows::Symbol
    missing_rows::Symbol
    observed_component::Symbol
    missing_lookup::Symbol
    missing_mask::Symbol
    nmissing::Int
    distribution::_BRMPreparedExpr
    # Sample a standard normal draw per missing value and compute the value
    # from its law's location and scale (`RKBRMI(...; completions=:noncentered)`).
    noncentered::Bool
    # `nothing`, or the names of the free and grouped missing rows, the groups'
    # ends and residuals of `mi(x)::shares(total, rows)` (cell_total_coordinates.jl).
    totals::Union{Nothing,NamedTuple}
end
_RKMissingValueSpec(source, observed, observed_rows, missing_rows, observed_component,
        missing_lookup, missing_mask, nmissing, distribution, noncentered=false) =
    _RKMissingValueSpec(source, observed, observed_rows, missing_rows, observed_component,
        missing_lookup, missing_mask, nmissing, distribution, noncentered, nothing)

function _rk_mi_predictor_plan(term::NamedColumn)
    operation = parent(term)
    operation isa ExprColumn && getf(operation) === (~) || return nothing
    _brm_missing_response_plan(first(getargs(operation)); prefix="RK backend")
end
_rk_mi_predictor_plan(_) = nothing

# Formula geometry needs the declared row axis, not a numerical evaluation at
# invented parameter values. Track pointwise assignment dependencies back to
# their actual data/observation axes; scalar sampled parents add no row axis.
_rk_model_value_axis(::Number, _context) = ()
_rk_model_value_axis(_value, _context=nothing) = nothing
_rk_model_value_axis(value::NamedColumn, context) =
    _rk_model_value_axis(value, parent(value), context)
function _rk_model_value_axis(value::NamedColumn, backing::DataColumn, _context)
    raw = parent(backing)
    raw isa Real && return ()
    raw isa AbstractVector{<:Union{Missing,Real}} || return nothing
    (source=name(value), nrows=length(raw))
end
_rk_model_value_axis(value::NamedColumn, backing::ExprColumn{typeof(assign)}, context) =
    _rk_model_value_axis(last(getargs(backing)), context)
function _rk_model_value_axis(value::NamedColumn, backing::ExprColumn{typeof(~)}, context)
    lhs, rhs = getargs(backing)
    completion = _brm_missing_response_plan(lhs; prefix="RK backend")
    completion === nothing || return (
        source=completion.source, nrows=length(completion.values))
    _brm_observation_name(lhs) === nothing || return _rk_model_value_axis(lhs, context)
    _brm_prior_expression(rhs) && return (
        _brm_parameter_reference_axis(rhs) === :scalar ? () : nothing)
    context === nothing && return nothing
    # Reuse the predictor's actual geometry, including declared consumers and
    # group rows. The number of group levels is a coefficient axis, not this
    # predictor's row axis; no parameter values are invented to resolve it.
    geometry = _rk_model_predictor_geometry(context.parent, context, name(value);
        available_predictors=Tuple(p.name for p in linear_predictors(context.parent)),
        tolerant_default=true)
    (; source=geometry.design.row_source, nrows=size(geometry.design.matrix, 1))
end
_rk_model_value_axis(::NamedColumn, _backing, _context) = nothing
function _rk_model_value_axis(value::ExprColumn, context)
    callable = getf(value)
    (haskey(_RK_DERIVED_BINOPS, callable) ||
     haskey(_RK_DERIVED_MATH, callable) ||
     haskey(_RK_DERIVED_CMP, callable)) && isempty(getkwargs(value)) || return nothing
    axes = map(argument -> _rk_model_value_axis(argument, context), getargs(value))
    any(isnothing, axes) && return nothing
    rows = filter(!isempty, axes)
    isempty(rows) && return ()
    shape = Base.Broadcast.broadcast_shape(((row.nrows,) for row in rows)...)
    rows[findfirst(row -> row.nrows == only(shape), rows)]
end
# A gather `x[idx]` of a value with a known axis lies on its index's rows
# (`c_b = c[pos_b]`, part 1's completed value on part 2's rows).
function _rk_model_value_axis(value::ExprColumn{typeof(getindex)}, context)
    args = getargs(value)
    length(args) == 2 && isempty(getkwargs(value)) || return nothing
    source, index = args
    _rk_model_value_axis(source, context) === nothing && return nothing
    _rk_model_value_axis(index, context)
end

function _rk_named_model_population_column(term::NamedColumn, context)
    parent(term) isa DataColumn && return nothing
    axis = _rk_model_value_axis(term, context)
    (axis === nothing || isempty(axis)) && return nothing
    label = name(term)
    # Missing geometry marks values unavailable until graph execution. It is
    # never bound as data: the term consumes the emitted assignment by name.
    (; label, effect_addresses=(label,), effect_block=label, source=axis.source,
        values=fill(missing, axis.nrows),
        preprocess=_BRMPopulationPreprocess(:model_value, nothing, term),
        runtime_expression=label)
end
_rk_named_model_population_column(_term, _context) = nothing

function _rk_model_population_column(term, context=nothing)
    inner = term
    kind = :model_value
    if term isa ExprColumn && getf(term) in (standardize, zscale, center)
        length(getargs(term)) == 1 && isempty(getkwargs(term)) || return nothing
        inner = only(getargs(term))
        kind = nameof(getf(term))
    end
    plan = _rk_mi_predictor_plan(inner)
    plan === nothing && return kind === :model_value ?
        _rk_named_model_population_column(inner, context) : nothing
    label = kind === :model_value ? name(inner) : _brm_wrapper_col_name(kind, inner)
    raw = plan.values
    make_error = message -> ArgumentError("RK backend: missing covariate transform: $message")
    # A frozen replay fits the anchors to the training observations.
    observed = _rk_fit_observed_values(plan)
    fit = kind === :model_value ? nothing :
        _brm_fitted(Symbol(:observed_, kind), plan.source, plan.values) do values
            observations = Float64.(collect(skipmissing(values)))
            kind === :center ? (; mean=_brm_fit_mean_numeric(observations, :predictor,
                    :center, make_error), scale=1.0) :
                _brm_fit_zscale_numeric(observations, :predictor, make_error)
        end
    values = fit === nothing ? raw : (raw .- fit.mean) ./ fit.scale
    anchors = Expr(:call, :_brm_observed_values, plan.source, observed)
    expression = fit === nothing ? name(inner) :
        Expr(:call, :./,
            Expr(:call, :.-, name(inner),
                Expr(:_rk_data_preparation, :brm_covariate_mean, anchors)),
            kind === :center ? 1.0 :
                Expr(:_rk_data_preparation, :brm_covariate_sd, anchors))
    preprocess = _BRMPopulationPreprocess(kind,
        fit === nothing ? nothing : (fit.mean, fit.scale), inner)
    (; label, effect_addresses=(label,), effect_block=label,
        source=plan.source, values, preprocess, runtime_expression=expression)
end

# The observed values a completed covariate's anchors are fitted to: the bound
# column's, or the training column's on a frozen replay (`_brm_fit_input_source`).
_rk_fit_observed_values(plan) =
    Float64.(collect(skipmissing(_brm_fit_input_source(plan.source, plan.values))))

# A completed covariate's anchors read a fit-role input holding those observed
# values (`_rk_fit_input!`), so a frozen `reprocess` keeps the training
# anchors. Replaces the planning marker `_brm_observed_values(source, values)`.
function _rk_fit_observed_anchors!(value::Expr, derived, columns, taken)
    if Meta.isexpr(value, :call, 3) && first(value.args) === :_brm_observed_values
        _, source, observed = value.args
        return _rk_fit_input!(derived, columns, taken, Symbol(source, :_observed), observed)
    end
    Expr(value.head, map(arg -> _rk_fit_observed_anchors!(arg, derived, columns, taken),
        value.args)...)
end
_rk_fit_observed_anchors!(value, _derived, _columns, _taken) = value

# The marker is planning metadata, resolved to the actual packed observation
# port before source is lowered. Its mean/std remain executable graph math.
function _rk_observed_anchor_source(value::Expr, observations)
    if Meta.isexpr(value, :call, 2) && first(value.args) === :_brm_observed_values
        return observations[value.args[2]]
    end
    Expr(value.head, map(arg -> _rk_observed_anchor_source(arg, observations), value.args)...)
end
_rk_observed_anchor_source(value, _observations) = value

function _rk_model_population_columns(term; cellmeans=false, context=nothing)
    # An offset of a model value is a coefficient-free graph summand
    # (`_rk_model_offset_spec`): it adds no design column.
    term isa ExprColumn && getf(term) === offset && length(getargs(term)) == 1 &&
        _brm_reads_model_value(only(getargs(term))) && return ()
    column = _rk_model_population_column(term, context)
    column === nothing ? _brm_population_columns(term; cellmeans) : (column,)
end

function _rk_model_random_effect_columns(term; cellmeans=false, context)
    column = _rk_model_population_column(term, context)
    column === nothing ? _brm_random_effect_columns(term; cellmeans) : (column,)
end

Base.@nospecializeinfer function _rk_model_predictor_geometry(@nospecialize(brmi::BRMI), context, target; kwargs...)
    data = copy(context.data)
    # An `mi(...)` observation column never reaches the materialized data, yet
    # it alone sizes an intercept-only predictor its likelihood reads
    # (`log(sigma) ~ 1`). As for a modeled covariate below, that row axis is
    # metadata, not executable values.
    source = get(context.target_obs, target, nothing)
    rows = isnothing(source) ? nothing : _brm_skipped_observation_rows(context, source)
    isnothing(rows) || haskey(data, source) || (data[source] = Base.OneTo(rows))
    geometry_context = _BRMBackendContext(context.parent, data, context.prepass,
        context.target_obs, context.target_axes, context.term_priors,
        context.group_declarations)
    function population_columns(term; cellmeans=false)
        columns = _rk_model_population_columns(term; cellmeans, context=geometry_context)
        if columns !== nothing
            for column in columns
                column.source === nothing && continue
                # A wholly modeled covariate has no materialized source.
                # Its declared row axis is metadata, not executable values.
                haskey(data, column.source) ||
                    (data[column.source] = Base.OneTo(length(column.values)))
            end
        end
        columns
    end
    _brm_prepare_predictor_geometry(brmi, geometry_context, target; kwargs...,
        population_columns,
        random_effect_columns=(term; cellmeans=false) ->
            _rk_model_random_effect_columns(term; cellmeans, context=geometry_context),
        prepare_term=_rk_prepare_term)
end

# RK evaluates every HSGP basis in its graph (`brm_hsgp_basis_graph`,
# `brm_hsgp_periodic_basis`); its planner reads a term's fits, floor and
# options, so it prepares the term without evaluating a basis.
_rk_prepare_term(term, target, context) = _brm_prepare_term(term, target, context)
_rk_prepare_term(term::ExprColumn{typeof(hsgp)}, target, context) =
    _brm_prepare_term(term, target, context; basis=_brm_hsgp_basis_geometry)

function _rk_mi_downstream(program, key, source)
    any(operation -> operation.name !== key &&
        _rk_refs_name(operation.expression, source), program.operations)
end

# A distribution parameter on the response row axis is gathered separately
# for observed and missing rows. Scalar parameters retain scalar broadcasting.
_rk_mi_gather(value, _rows) = value
function _rk_mi_gather(value::_BRMPreparedRef, rows)
    value.axis in (:observation, :observation_row) || return value
    indices = rows isa Symbol ? _BRMPreparedRef(rows, :whole) : rows
    _BRMPreparedExpr(getindex, (value, indices), (;))
end
function _rk_mi_gather(value::_BRMPreparedExpr, rows)
    _BRMPreparedExpr(value.callable,
        map(argument -> _rk_mi_gather(argument, rows), value.args),
        map(argument -> _rk_mi_gather(argument, rows), value.kwargs))
end

function _rk_prepare_missing_value!(columns, taken, observation, program, completions, derived)
    plan = observation.missing_response
    plan === nothing && return observation
    observation.modifier === nothing || error(
        "RK backend: missing response `$(observation.name)` with response modifiers requires a supported completion law")
    observed = _rk_ast_fresh_name(string(plan.source, "_obs"), taken)
    jobs = _rk_ast_fresh_name(string("Jobs_", plan.source), taken)
    raw = _rk_ast_fresh_name(string(plan.source, "_raw"), taken)
    columns[raw] = plan.values
    columns[observed], columns[jobs] = plan.observed_values, plan.observed_indices
    push!(derived, _RKDerivedSpec(observed,
        Expr(:_rk_data_preparation, :brm_covariate_observed, raw), observed))
    push!(derived, _RKDerivedSpec(jobs,
        Expr(:_rk_data_preparation, :brm_covariate_observed_rows, raw), jobs))
    delete!(columns, plan.source)
    delete!(columns, observation.name)
    # Grouped entries carry what their residuals say about the law whether or
    # not anything reads the completion.
    grouped = _brm_mi_grouped(plan.totals)
    if grouped || _rk_mi_downstream(program, observation.name, plan.source)
        jmis = _rk_ast_fresh_name(string("Jmis_", plan.source), taken)
        component = _rk_ast_fresh_name(string(plan.source, "_observed_component"), taken)
        lookup = _rk_ast_fresh_name(string(plan.source, "_missing_lookup"), taken)
        mask = _rk_ast_fresh_name(string(plan.source, "_missing_mask"), taken)
        # Row partitions describe the declared covariate geometry. The
        # observed contribution, missing lookup and mask are graph values.
        columns[jmis] = plan.missing_indices
        totals = grouped ? merge(_rk_prepare_missing_totals!(columns, taken, derived,
            observation, plan, raw),
            (; downstream=_rk_mi_downstream(program, observation.name, plan.source))) : nothing
        push!(derived, _RKDerivedSpec(jmis, grouped ?
            Expr(:_rk_data_preparation, :brm_covariate_total_rows, raw, totals.inputs...) :
            Expr(:_rk_data_preparation, :brm_covariate_missing_rows, raw), jmis))
        push!(completions, _RKMissingValueSpec(plan.source, observed,
            jobs, jmis, component, lookup, mask, length(plan.missing_indices),
            observation.distribution, false, totals))
    end
    _BRMPreparedObservation(observed, observation.lhs,
        _rk_mi_gather(observation.distribution, jobs), plan.observed_values,
        nothing, observation.weight, nothing)
end

# The data of a completion's group totals: the three columns bound under
# fresh names, and the free and grouped rows, ends and residuals as data
# preparations from them (`brm_covariate_total_*`), so a replay recomputes
# them from its own data.
function _rk_prepare_missing_totals!(columns, taken, derived, observation, plan, raw)
    spec = _brm_mi_totals_spec(observation.lhs; prefix="RK backend")
    inputs = map((:group, :cells, :total)) do key
        name = _rk_ast_fresh_name(string(plan.source, "_shares_", key), taken)
        columns[name] = getproperty(spec, key).values
        name
    end
    layout = _brm_totals_layout(plan.values, spec.group.values, spec.cells.values,
        spec.total.values)
    names = map((free_rows=(:free, :brm_covariate_total_free_rows),
            grouped_rows=(:grouped, :brm_covariate_total_grouped_rows),
            ends=(:ends, :brm_covariate_total_ends),
            residuals=(:residuals, :brm_covariate_total_residuals),
            coordinates=(:coordinates, :brm_covariate_total_coordinates))) do (field, recipe)
        name = _rk_ast_fresh_name(string(plan.source, "_shares_", field), taken)
        columns[name] = getproperty(layout, field)
        push!(derived, _RKDerivedSpec(name,
            Expr(:_rk_data_preparation, recipe, raw, inputs...), name))
        name
    end
    # The placeholder a non-centered completion's groups observe is bound data:
    # a submodel observes only an argument bound to a data column.
    weights = _rk_ast_fresh_name(string(plan.source, "_shares_weights"), taken)
    columns[weights] = layout.weights
    merge(names, (; weights, inputs))
end

# Emits the completion block and returns one entry per part holding sampled
# coordinates (the missing entries; with group totals the free entries, then
# the grouped entries' shares), or `nothing` when nothing is missing. Each
# entry is `(; sampled, standardized)`: the block local holding the part's
# coordinates (`y_mis`, `z_mis` when non-centered, `y_free` / `y_shares` with
# group totals), and, for a part standardized by its law's location and scale,
# the locals of the location, the scale and the value
# (`_rk_emit_standardized!`), else `nothing`.
function _rk_emit_missing_value!(definitions, statements, bindings, taken, completion)
    geometry = _rk_ast_statistical_call!(definitions, taken,
        :brm_covariate_geometry, completion.observed, completion.observed_rows,
        completion.missing_rows; kernel=true)
    push!(statements, Expr(:(=), Expr(:tuple, completion.observed_component,
        completion.missing_lookup, completion.missing_mask), geometry))
    if completion.nmissing == 0
        push!(statements, Expr(:(=), completion.source, completion.observed_component))
        return nothing
    end
    completion.totals === nothing ||
        return _rk_emit_missing_totals!(definitions, statements, bindings, taken, completion)
    distribution = _rk_mi_gather(completion.distribution, completion.missing_rows)
    # One block, as StanBlocks' `_sb_mi_response`: the missing entries with
    # their law, completed onto the covariate's original row axis. The
    # declared dimensions belong to the missing-row axis; unlike an
    # observation broadcast, the statement samples that independent array.
    if completion.noncentered
        location, scale = map(arg -> _rk_value_expr!(bindings, arg, taken), distribution.args)
        part = (; callable=distribution.callable, location, scale)
        block = _rk_block_body(location, scale)
    else
        law = _rk_ast_value_distribution(distribution, bindings, taken)
        block = _rk_block_body(law)
    end
    observed = _rk_block_argument!(block, :observed, completion.observed_component)
    rows = _rk_block_argument!(block, :rows, completion.missing_rows)
    lookup = _rk_block_argument!(block, :lookup, completion.missing_lookup)
    mask = _rk_block_argument!(block, :mask, completion.missing_mask)
    if completion.noncentered
        entry = _rk_emit_standardized!(block, part, rows, :z_mis, :y_mis)
        missing = entry.standardized.value
    else
        missing = _rk_block_local!(block, :y_mis)
        push!(block.statements, Expr(:call, :.~,
            Expr(:ref, missing, Expr(:call, :axes, rows, 1)), law))
        entry = (; sampled=missing, standardized=nothing)
    end
    # Completion is a whole covariate value on its original row axis. Keep
    # the gather and mask in an explicit graph when downstream likelihoods
    # have different observation axes, rather than inferring a scalar recipe.
    completed = _rk_block_local!(block, :completed)
    push!(block.statements, Expr(:(=), completed, _rk_ast_statistical_call!(definitions,
        taken, :brm_completed_covariate, observed, missing, lookup, mask; kernel=true)))
    push!(statements, Expr(:call, :~, completion.source,
        _rk_ast_block_call!(definitions, taken, "brm_missing_covariate", block,
            completed)))
    (entry,)
end

# A location-scale law in standardized coordinates:
# `draws[axes(rows, 1)] .~ Normal(0, 1)` and
# `value = location .+ scale .* draws` (`exp.(…)` for LogNormal), the same law.
# The location and scale are also bound to block locals of their own, which
# the coordinate transport reads by name to move the draws to the values a
# centred lowering samples (`brm_coordinate_transport`). The value reads the
# law's expressions, not those locals, so no density reads them: the graph
# RK differentiates is the one without them.
function _rk_emit_standardized!(block, part, rows, draws, value)
    location = _rk_block_local!(block, Symbol(value, "_location"))
    push!(block.statements, Expr(:(=), location, part.location))
    scale = _rk_block_local!(block, Symbol(value, "_scale"))
    push!(block.statements, Expr(:(=), scale, part.scale))
    sampled = _rk_block_local!(block, draws)
    push!(block.statements, Expr(:call, :.~,
        Expr(:ref, sampled, Expr(:call, :axes, rows, 1)), _rk_ast_dotted(:Normal, 0, 1)))
    completed = _rk_block_local!(block, value)
    affine = Expr(:call, :.+, part.location, Expr(:call, :.*, part.scale, sampled))
    push!(block.statements, Expr(:(=), completed,
        part.callable === LogNormal ? _rk_ast_dotted(:exp, affine) : affine))
    (; sampled, standardized=(; location, scale, value=completed), law=part.callable)
end

# The completion of `mi(x)::shares(total, rows)`, as StanBlocks'
# `_sb_mi_shares_response`: the free missing entries under the law as written,
# and the grouped ones as one joint latent whose coordinates are their shares
# (`_BRMGroupedShares`), free first in `missing_rows` order. Returns the block
# locals that hold sampled coordinates. When nothing reads the completion, the
# free entries integrate out and are not emitted (a generated draw, as on
# SBBRMI); the grouped ones stay, since their density says what the residuals
# say about the law.
#
# Non-centered (`RKBRMI(brmi; completions = :noncentered)`), free entries are
# standard normal draws standardized by the law's location and scale, as for
# any completion, and the grouped entries' coordinates are standard normal
# draws whose groups each observe the rest of the joint latent's density
# (`_BRMGroupedSharesWeights`). The coordinates and the density are the
# centred ones; only the prior/likelihood split differs.
function _rk_emit_missing_totals!(definitions, statements, bindings, taken, completion)
    t = completion.totals
    t.downstream || return _rk_emit_missing_shares!(definitions, statements, bindings,
        taken, completion)
    free_part = _rk_mi_free_part(bindings, taken, completion)
    grouped_part = _rk_mi_grouped_part(bindings, taken, completion)
    block = _rk_block_body(free_part.reads, grouped_part.reads)
    observed = _rk_block_argument!(block, :observed, completion.observed_component)
    free_rows = _rk_block_argument!(block, :free_rows, t.free_rows)
    grouped_rows = _rk_block_argument!(block, :grouped_rows, t.grouped_rows)
    lookup = _rk_block_argument!(block, :lookup, completion.missing_lookup)
    mask = _rk_block_argument!(block, :mask, completion.missing_mask)
    grouped_data = _rk_mi_grouped_arguments!(block, completion)
    free_entry, free = _rk_emit_mi_free!(block, free_part, free_rows)
    shares_sampled, shares = _rk_emit_mi_grouped!(block, grouped_part, grouped_rows,
        grouped_data)
    missing = _rk_block_local!(block, :y_mis)
    push!(block.statements, Expr(:(=), missing, Expr(:call, :vcat, free, shares)))
    completed = _rk_block_local!(block, :completed)
    push!(block.statements, Expr(:(=), completed, _rk_ast_statistical_call!(definitions,
        taken, :brm_completed_covariate, observed, missing, lookup, mask; kernel=true)))
    push!(statements, Expr(:call, :~, completion.source,
        _rk_ast_block_call!(definitions, taken, "brm_missing_covariate", block, completed)))
    (free_entry, (; sampled=shares_sampled, standardized=nothing))
end

function _rk_emit_missing_shares!(definitions, statements, bindings, taken, completion)
    t = completion.totals
    grouped_part = _rk_mi_grouped_part(bindings, taken, completion)
    block = _rk_block_body(grouped_part.reads)
    grouped_rows = _rk_block_argument!(block, :grouped_rows, t.grouped_rows)
    grouped_data = _rk_mi_grouped_arguments!(block, completion)
    sampled, shares = _rk_emit_mi_grouped!(block, grouped_part, grouped_rows, grouped_data)
    push!(statements, Expr(:call, :~, completion.source,
        _rk_ast_block_call!(definitions, taken, "brm_missing_shares", block, shares)))
    ((; sampled, standardized=nothing),)
end

# The free entries' law, gathered onto their rows: their law as written, or
# its location and scale when non-centered. A non-centered completion with no
# free entries may have any law (`_rk_noncentered_completion`); its empty free
# entries stay under the law as written.
function _rk_mi_free_part(bindings, taken, completion)
    distribution = _rk_mi_gather(completion.distribution, completion.totals.free_rows)
    if completion.noncentered && distribution.callable in _RK_NONCENTERED_LAWS
        location, scale = map(arg -> _rk_value_expr!(bindings, arg, taken), distribution.args)
        return (; noncentered=true, callable=distribution.callable, location, scale,
            reads=Expr(:tuple, location, scale))
    end
    law = _rk_ast_value_distribution(distribution, bindings, taken)
    (; noncentered=false, law, reads=law)
end

function _rk_emit_mi_free!(block, part, free_rows)
    if part.noncentered
        entry = _rk_emit_standardized!(block, part, free_rows, :z_free, :y_free)
        return entry, entry.standardized.value
    end
    free = _rk_block_local!(block, :y_free)
    push!(block.statements, Expr(:call, :.~,
        Expr(:ref, free, Expr(:call, :axes, free_rows, 1)), part.law))
    (; sampled=free, standardized=nothing), free
end

# The grouped entries' law: `_BRMGroupedShares`, or, non-centered, the callable
# of their groups' weights and the shares' values. `arguments` are the law's
# arguments gathered onto the grouped entries.
function _rk_mi_grouped_part(bindings, taken, completion)
    t = completion.totals
    F = completion.distribution.callable
    grouped = _rk_mi_gather(completion.distribution, t.grouped_rows)
    if completion.noncentered
        arguments = map(arg -> _rk_value_expr!(bindings, arg, taken), grouped.args)
        return (; noncentered=true, ends=t.ends, residuals=t.residuals, arguments,
            weights=_rk_value_callee!(bindings, _BRMGroupedSharesWeights{F}(), taken),
            values=_rk_value_callee!(bindings, _brm_grouped_shares_values, taken),
            weight=_rk_value_callee!(bindings, _BRMLogWeight, taken),
            reads=Expr(:tuple, t.ends, t.residuals, arguments...))
    end
    family = _rk_value_callee!(bindings, _BRMGroupedShares{F}, taken)
    law = Expr(:call, family, t.ends, t.residuals,
        map(arg -> _rk_value_expr!(bindings, arg, taken), grouped.args)...)
    (; noncentered=false, law, reads=law)
end

# A non-centered completion's grouped entries read the axis of their
# coordinates and observe one placeholder per group.
function _rk_mi_grouped_arguments!(block, completion)
    completion.noncentered || return nothing
    t = completion.totals
    (; coordinates=_rk_block_argument!(block, :coordinates, t.coordinates),
       weights=_rk_block_argument!(block, :weights, t.weights))
end

function _rk_emit_mi_grouped!(block, part, grouped_rows, data)
    if part.noncentered
        sampled = _rk_block_local!(block, :z_shares)
        push!(block.statements, Expr(:call, :.~,
            Expr(:ref, sampled, Expr(:call, :axes, data.coordinates, 1)),
            _rk_ast_dotted(:Normal, 0, 1)))
        shares = _rk_block_local!(block, :y_shares)
        push!(block.statements, Expr(:(=), shares,
            Expr(:call, part.values, sampled, part.ends, part.residuals)))
        weights = _rk_block_local!(block, :group_weights)
        push!(block.statements, Expr(:(=), weights, Expr(:call, part.weights,
            sampled, shares, part.ends, part.residuals, part.arguments...)))
        push!(block.statements, Expr(:call, :.~, data.weights,
            Expr(:., part.weight, Expr(:tuple, weights))))
        return sampled, shares
    end
    shares = _rk_block_local!(block, :y_shares)
    push!(block.statements, Expr(:call, :~,
        Expr(:ref, shares, Expr(:call, :axes, grouped_rows, 1)), part.law))
    shares, shares
end

function _rk_source_loop_indices!(names, statement)
    statement isa Expr || return names
    if statement.head === :for
        iterator = first(statement.args)
        Meta.isexpr(iterator, :(=)) && _rk_source_lhs!(names, first(iterator.args))
    end
    foreach(argument -> _rk_source_loop_indices!(names, argument), statement.args)
    names
end

# A block call `lhs ~ block(...)` also reads the caller names its submodel
# body mentions freely (for example a prior's hyperparameter or a missing
# covariate's law): every body symbol that is not an argument, a local
# output or a loop index.
function _rk_block_free_reads(definitions)
    reads = Dict{Symbol,Set{Symbol}}()
    for definition in definitions
        Meta.isexpr(definition, :(=), 2) || continue
        call, body = definition.args
        Meta.isexpr(call, :call) && Meta.isexpr(body, :block) || continue
        names = setdiff!(_rk_source_symbols!(Set{Symbol}(), body), call.args[2:end])
        for statement in body.args
            setdiff!(names, _rk_source_outputs!(Set{Symbol}(), statement))
            setdiff!(names, _rk_source_loop_indices!(Set{Symbol}(), statement))
        end
        reads[first(call.args)] = names
    end
    reads
end

# An observation reads its response rather than defining it: a top-level `~`,
# or a per-subject `@plate` holding only observations (`_rk_observation_statement`).
function _rk_observation_source_statement(statement)
    Meta.isexpr(statement, :call) && return first(statement.args) in (:~, :.~)
    body = _rk_plate_body(statement)
    body !== nothing && !isempty(body) && all(_rk_observation_source_statement, body)
end

# The statements of an emitted `@plate for … end`, or `nothing`.
function _rk_plate_body(statement)
    Meta.isexpr(statement, :macrocall) && first(statement.args) === Symbol("@plate") &&
        Meta.isexpr(last(statement.args), :for) || return nothing
    [s for s in last(last(statement.args).args).args if !(s isa LineNumberNode)]
end

# Completion, formula columns, sampled priors and readers may depend on one
# another. Order ordinary emitted statements by their declared outputs before
# PPL authoring, preserving independent source order and bound data ownership.
function _rk_order_value_statements(statements, data_names, definitions=(); observed=())
    block_reads = _rk_block_free_reads(definitions)
    outputs = map(statements) do statement
        names = _rk_source_outputs!(Set{Symbol}(), statement)
        _rk_observation_source_statement(statement) && setdiff!(names, observed)
        setdiff!(names, data_names)
        setdiff!(names, _rk_source_loop_indices!(Set{Symbol}(), statement))
    end
    producers = Dict{Symbol,Int}()
    for (index, names) in enumerate(outputs), name in names
        haskey(producers, name) && error("RK backend: multiple emitted definitions of `$name`")
        producers[name] = index
    end
    dependencies = map(eachindex(statements)) do index
        references = _rk_source_symbols!(Set{Symbol}(), statements[index])
        for name in collect(references)
            haskey(block_reads, name) && union!(references, block_reads[name])
        end
        Set(producers[name] for name in references if haskey(producers, name) &&
            producers[name] != index)
    end
    pending, finished, ordered = collect(eachindex(statements)), Set{Int}(), Expr[]
    while !isempty(pending)
        next = findfirst(index -> issubset(dependencies[index], finished), pending)
        next === nothing && error("RK backend: cyclic completed-value source dependencies")
        index = popat!(pending, next)
        push!(finished, index)
        push!(ordered, statements[index])
    end
    ordered
end
