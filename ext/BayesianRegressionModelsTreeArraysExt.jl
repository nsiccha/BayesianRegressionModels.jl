module BayesianRegressionModelsTreeArraysExt

using BayesianRegressionModels
using TreeArrays: TreeData
const BRM = BayesianRegressionModels

_output(d, logical::Symbol, role) = brm_output(d, logical; role)
function _output(d, output::BRM.BRMOutput, role)
    isnothing(role) || role === output.role || throw(ArgumentError(
        "brm_output_tree: role does not match the supplied BRMOutput"))
    output in d.outputs || throw(ArgumentError(
        "brm_output_tree: supplied output is not part of this descriptor"))
    output
end

# Return coordinates in the carrier's own element order. The caller's name
# permutation is applied separately, together with the view's column selection.
function _output_axis(output, caller)
    dims = output.dims
    isnothing(dims) && throw(ArgumentError(
        "$caller: output $(output.name) has no descriptor shape for its element axis"))
    isempty(dims) && return nothing
    if output.role === :membership && length(dims) == 2 && !isnothing(output.labels)
        length(output.labels) == dims[2] || throw(DimensionMismatch(
            "$caller: component labels do not match output $(output.name)"))
        return :element => [(observation=i, component=label)
                            for label in output.labels for i in 1:dims[1]]
    end
    if !isnothing(output.labels)
        length(output.labels) == prod(dims) || throw(DimensionMismatch(
            "$caller: labels do not match the element axis of $(output.name)"))
        return :coefficient => output.labels
    end
    if length(dims) == 1
        if output.role in (:linear_predictor, :posterior_predictive, :pointwise_loglik) ||
           (isnothing(output.segments) && output.component === :contribution)
            # The descriptor shape is role-specific: ragged predictions have
            # one value per element, while their likelihood has one per group.
            return :observation => Base.OneTo(only(dims))
        elseif isnothing(output.segments) && output.role === :parameter &&
                isnothing(output.component) && isempty(output.predictors)
            # An explicitly declared vector's positions ARE its coordinates.
            return :element => Base.OneTo(only(dims))
        end
    end
    throw(ArgumentError(
        "$caller: descriptor cannot name the element axis of output $(output.name) " *
        "(role=$(output.role), shape=$dims)"))
end

# A model-scope assignment inherits observations only through pointwise
# operations on known row outputs and scalars. Equal lengths alone cannot
# establish this: a coefficient vector or an indexed/reordered value may
# happen to have the same length as the observations.
_row_extent(d, x, seen) = nothing
_row_extent(d, x::Real, seen) = 0
_row_extent(d, x::BRM.NamedColumn, seen) = _row_extent(d, BRM.name(x), seen)
function _row_extent(d, name::Symbol, seen)
    name in seen && return nothing
    # Fitted constants and scalar data are inputs, not output carriers. Their
    # declared scalar shape is broadcast-neutral; an array constant has no
    # observation semantics merely because its length matches the rows.
    any(i -> i.name === name && i.type in (:real, :int) && isempty(i.size), d.inputs) &&
        return 0
    outputs = filter(o -> o.name === name, d.outputs)
    length(outputs) == 1 || return nothing
    output = only(outputs)
    output.dims == () && return 0
    isnothing(output.dims) && return nothing
    length(output.dims) == 1 && isnothing(output.segments) || return nothing
    (output.role === :linear_predictor || output.component === :contribution) &&
        return only(output.dims)
    brmi = d.plan.parent
    BRM._brm_has_operation(brmi, name) || return nothing
    op = BRM._named_op(BRM._brm_operation(brmi, name))
    op isa BRM.ExprColumn{typeof(BRM.assign)} || return nothing
    extent = _row_extent(d, last(BRM.getargs(op)), union(seen, (name,)))
    extent == only(output.dims) ? extent : nothing
end
function _row_extent(d, x::BRM.ExprColumn, seen)
    BRM.getf(x) in (+, -, *, /, ^, exp, expm1, log, log1p, sqrt,
                   abs, abs2, sin, cos, tan, tanh) || return nothing
    isempty(BRM.getkwargs(x)) || return nothing
    extents = map(arg -> _row_extent(d, arg, seen), BRM.getargs(x))
    any(isnothing, extents) && return nothing
    rows = unique(filter(!iszero, collect(extents)))
    isempty(rows) ? 0 : length(rows) == 1 ? only(rows) : nothing
end

function _output_axis(d, output, caller)
    if output.role === :stan_derived && !isnothing(output.dims) && length(output.dims) == 1
        extent = _row_extent(d, output.name, Set{Symbol}())
        isnothing(extent) || return :observation => Base.OneTo(extent)
    end
    _output_axis(output, caller)
end

# Random-effect carriers have several physical layouts for the same fitted
# coordinates. Reuse the emission table and margin metadata used by the other
# readers; neither the carrier's spelling nor equal-sized dimensions identify
# which coordinate varies fastest.
_ranef_draw_axes(::Val{:group}, levels, margins) = (:level => levels,)
_ranef_draw_axes(::Val{:group_term}, levels, margins) =
    (:level => levels, :margin => margins)
_ranef_draw_axes(::Val{:term_group}, levels, margins) =
    (:margin => margins, :level => levels)
_ranef_draw_axes(::Val{:flat_term_group}, levels, margins) =
    _ranef_draw_axes(Val(:term_group), levels, margins)

function _ranef_output_axes(d, output)
    isnothing(output.component) && return nothing
    output.dims == () && return nothing
    margins_by_block = BRM._adaptive_block_margins(d)
    for block in ranef_blocks(d)
        spec = BRM._RANEF_FAMILIES[block.family]
        keywords = BRM._brm_ranef_keywords(d, block)
        (output.name => output.component) in
            BRM._ranef_component_carriers(block.binding, spec, keywords) || continue
        margins = collect(margins_by_block[block.binding])
        length(margins) == block.n_terms || throw(DimensionMismatch(
            "brm_output_tree: margins do not match block $(block.binding)"))
        if output.name === block.z
            return _ranef_draw_axes(Val(spec.layout), block.levels, margins)
        end
        return _ranef_component_axes(Val(output.component), d, output,
                                     (; block, keywords, margins))
    end
    nothing
end

_ranef_carrier(x, s) = (k = get(x.keywords, s, nothing);
    k isa Symbol ? k : Symbol(x.block.binding, :_, s))
_ranef_component_axes(::Val, d, output, x) = nothing
function _ranef_component_axes(::Val{:standardized}, d, output, x)
    margins = output.name === _ranef_carrier(x, :z_inactive) ?
        x.margins[collect(d.plan.bindings[x.block.binding].margins)] : x.margins
    (:margin => margins, :level => x.block.levels)
end
function _ranef_component_axes(::Val{:effects}, d, output, x)
    output.name in (_ranef_carrier(x, :b_T), _ranef_carrier(x, :b_cols)) &&
        return (:margin => x.margins, :level => x.block.levels)
    (:level => x.block.levels, :margin => x.margins)
end
_ranef_component_axes(::Val{:mixing_weight}, d, output, x) = (:level => x.block.levels,)
function _ranef_component_axes(::Val{:scale}, d, output, x)
    margins = x.margins
    if output.name === _ranef_carrier(x, :tau_free)
        known = d.plan.bindings[x.block.binding].known_scales
        margins = margins[findall(isnothing, collect(known))]
    elseif output.name === _ranef_carrier(x, :tau_inactive)
        margins = margins[collect(d.plan.bindings[x.block.binding].margins)]
    end
    isnothing(x.block.by) && return (:margin => margins,)
    strata = BRM._brm_ranef_strata(d, x.block, "brm_output_tree").strata
    (:margin => margins, :stratum => strata)
end
function _ranef_component_axes(::Val{:correlation_cholesky}, d, output, x)
    margins = x.margins
    isnothing(x.block.by) && return (:margin => margins, :margin2 => margins)
    strata = BRM._brm_ranef_strata(d, x.block, "brm_output_tree").strata
    (:stratum => strata, :margin => margins, :margin2 => margins)
end
_ranef_component_axes(::Val{:level_scale}, d, output, x) =
    (:margin => x.margins, :level => x.block.levels)
_ranef_component_axes(::Val{:level_correlation_cholesky}, d, output, x) =
    (:level => x.block.levels, :margin => x.margins, :margin2 => x.margins)

function _ranef_output_tree(draws, ordered, axes)
    # Reorder only a coordinate-sized index array, never the draw storage.
    # The new multi-axis outputs use fitted coordinate order even when the
    # caller permutes names; existing single-axis output ordering is unchanged.
    shape = map(axis -> length(last(axis)), axes)
    length(ordered) == prod(shape) || throw(DimensionMismatch(
        "brm_output_tree: random-effect coordinates do not match the carrier"))
    indices = reshape(ordered, shape)
    if length(axes) == 2 && first.(axes) == (:margin, :level)
        indices = PermutedDimsArray(indices, (2, 1))
        axes = reverse(axes)
    end
    # One-margin effects have only a level axis, as a random intercept does.
    if length(axes) == 2 && first.(axes) == (:level, :margin) && length(last(axes[2])) == 1
        axes = (first(axes),)
    end
    values = reshape(view(draws, :, vec(indices)), size(draws, 1),
                     map(axis -> length(last(axis)), axes)...)
    TreeData(values, :draw => Base.axes(draws, 1), axes...)
end

function BRM.brm_output_tree(d::BRMDescriptor, draws::AbstractMatrix, names;
                              logical, role=nothing)
    BRM._brm_check_draw_names(draws, names)
    output = _output(d, logical, role)
    indices = brm_output_coordinates(output, names)
    ordered = BRM._brm_element_coordinates(output, names)
    isnothing(output.dims) || length(indices) == prod(output.dims) || throw(DimensionMismatch(
        "brm_output_tree: $(output.name) needs $(prod(output.dims)) elements, got $(length(indices))"))
    ranef_axes = _ranef_output_axes(d, output)
    isnothing(ranef_axes) || return _ranef_output_tree(draws, ordered, ranef_axes)
    axis = _output_axis(d, output, "brm_output_tree")
    if isnothing(axis)
        return TreeData(view(draws, :, only(indices)), :draw => axes(draws, 1))
    end
    position = Dict(index => i for (i, index) in enumerate(ordered))
    labels = last(axis)[[position[index] for index in indices]]
    TreeData(view(draws, :, indices), :draw => axes(draws, 1), first(axis) => labels)
end

function _chains(f, chains::AbstractVector{<:AbstractMatrix})
    isempty(chains) && throw(ArgumentError("BRM draw trees require at least one chain"))
    TreeData(map(f, chains), :chain => eachindex(chains))
end

BRM.brm_output_tree(d::BRMDescriptor, chains::AbstractVector{<:AbstractMatrix}, names; kwargs...) =
    _chains(C -> brm_output_tree(d, C, names; kwargs...), chains)

function _ranef_record(entry)
    draw = :draw => axes(entry.sd, 1)
    sd = TreeData(entry.sd, draw, :margin => entry.margins)
    cor = TreeData(entry.cor, draw, :margin => entry.margins, :margin2 => entry.margins)
    effects = TreeData(entry.effects, draw, :level => entry.levels, :margin => entry.margins)
    fields = (; sd, cor, effects)
    if !isnothing(entry.level_cor)
        fields = (; fields..., level_cor=TreeData(entry.level_cor, draw,
                    :level => entry.levels, :level2 => entry.levels))
    end
    TreeData(:quantity => fields)
end

function BRM.brm_ranef_tree(entries::AbstractVector{<:NamedTuple})
    isempty(entries) && throw(ArgumentError("brm_ranef_tree: no random-effect blocks"))
    labels = [(; e.binding, e.group, e.id, e.by, e.stratum) for e in entries]
    TreeData(map(_ranef_record, entries), :block => labels)
end

BRM.brm_ranef_tree(d::BRMDescriptor, draws::AbstractMatrix, names; kwargs...) =
    brm_ranef_tree(BRM._brm_ranef_views(d, draws, names; kwargs...))

BRM.brm_ranef_tree(d::BRMDescriptor, chains::AbstractVector{<:AbstractMatrix}, names; kwargs...) =
    _chains(C -> brm_ranef_tree(d, C, names; kwargs...), chains)

function BRM.brm_predictive_tree(d::BRMDescriptor, predicted::NamedTuple)
    fields = map(keys(predicted)) do response
        output = brm_output(d, response; role=:posterior_predictive)
        values = getproperty(predicted, response)
        axis = _output_axis(output, "brm_predictive_tree")
        size(values, 2) == prod(output.dims) || throw(DimensionMismatch(
            "brm_predictive_tree: response $response does not match its descriptor shape"))
        draw = :draw => axes(values, 1)
        isnothing(axis) ? TreeData(view(values, :, 1), draw) :
            TreeData(values, draw, axis)
    end
    TreeData(:response => NamedTuple{keys(predicted)}(fields))
end

BRM.brm_predictive_tree(d::BRMDescriptor, draws::AbstractMatrix; kwargs...) =
    brm_predictive_tree(d, brm_predictive_draws(d, draws; kwargs...))

function BRM.brm_predictive_tree(d::BRMDescriptor,
        chains::AbstractVector{<:AbstractMatrix}; problem, seed::Integer)
    next_seed = seed
    _chains(chains) do C
        result = brm_predictive_tree(d, C; problem, seed=next_seed)
        next_seed = Base.checked_add(next_seed, size(C, 1))
        result
    end
end

function BRM.brm_conditional_tree(result::NamedTuple)
    grid = result.grid
    columns = isnothing(result.focal) ? Tuple(propertynames(grid)) :
        result.focal isa Symbol ? (result.focal,) : Tuple(result.focal)
    isempty(columns) && throw(ArgumentError("brm_conditional_tree: grid axis has no coordinate columns"))
    values = map(column -> BRM._brm_df_column(grid, column), columns)
    n = size(result.draws, 2)
    all(v -> length(v) == n, values) || throw(DimensionMismatch(
        "brm_conditional_tree: grid coordinates do not match the draw columns"))
    labels = [NamedTuple{columns}(map(v -> v[i], values)) for i in 1:n]
    (; result..., draws=TreeData(result.draws, :draw => axes(result.draws, 1), :grid => labels))
end

BRM.brm_conditional_tree(d::BRMDescriptor, draws::AbstractMatrix, grid; kwargs...) =
    brm_conditional_tree(brm_conditional_draws(d, draws, grid; kwargs...))

function BRM.brm_conditional_tree(d::BRMDescriptor,
        chains::AbstractVector{<:AbstractMatrix}, grid; seed::Integer=1, kwargs...)
    isempty(chains) && throw(ArgumentError("brm_conditional_tree: no chains"))
    next_seed = seed
    results = map(chains) do C
        result = brm_conditional_tree(d, C, grid; seed=next_seed, kwargs...)
        next_seed = Base.checked_add(next_seed, size(C, 1))
        result
    end
    (; first(results)...,
       draws=TreeData([r.draws for r in results], :chain => eachindex(chains)))
end

_rk_value_tree(values::AbstractVector{<:Number}, name) = TreeData(values, :draw => axes(values, 1))
function _rk_value_tree(values::AbstractVector, name)
    all(v -> v isa Number, values) && return TreeData(values, :draw => axes(values, 1))
    leaves = map(values) do value
        value isa AbstractVector{<:Number} || throw(ArgumentError(
            "rk_draw_tree: cannot name the value axes of $name with type $(typeof(value)); " *
            "rk_draw_tree(rk, U, names) labels the cells of a @plate output"))
        TreeData(value, :element => axes(value, 1))
    end
    TreeData(leaves, :draw => axes(values, 1))
end

# A plate output's draws, per cell: only the backend knows that a nested value
# holds cells, so only the form that has it labels them.
_rk_cell_tree(cell::AbstractVector{<:Number}, name) = TreeData(cell, :element => axes(cell, 1))
_rk_cell_tree(cell::AbstractMatrix{<:Number}, name) =
    TreeData(cell, :row => axes(cell, 1), :column => axes(cell, 2))
_rk_cell_tree(cell, name) = throw(ArgumentError(
    "rk_draw_tree: cannot name the value axes of a cell of $name with type $(typeof(cell))"))
_rk_cells_tree(cells::AbstractVector{<:Number}, name) = TreeData(cells, :cell => axes(cells, 1))
_rk_cells_tree(cells::AbstractVector, name) =
    TreeData(map(cell -> _rk_cell_tree(cell, name), cells), :cell => axes(cells, 1))
_rk_plate_output_tree(values, name) =
    TreeData(map(cells -> _rk_cells_tree(cells, name), values), :draw => axes(values, 1))

function BRM.rk_draw_tree(values::NamedTuple)
    fields = map(name -> _rk_value_tree(getproperty(values, name), name), keys(values))
    TreeData(:value => NamedTuple{keys(values)}(fields))
end

function BRM.rk_draw_tree(rk::RKBRMI, U::AbstractMatrix, names)
    values = rk_draw_values(rk, U, names)
    cells = BRM._rk_plate_output_cell_names(rk)
    fields = map(keys(values)) do name
        value = getproperty(values, name)
        haskey(cells, name) ? _rk_plate_output_tree(value, name) : _rk_value_tree(value, name)
    end
    TreeData(:value => NamedTuple{keys(values)}(fields))
end

BRM.rk_draw_tree(rk::RKBRMI, chains::AbstractVector{<:AbstractMatrix}, names) =
    _chains(U -> rk_draw_tree(rk, U, names), chains)

end
