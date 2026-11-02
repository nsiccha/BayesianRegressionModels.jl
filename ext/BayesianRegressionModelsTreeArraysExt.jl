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
    isnothing(output.segments) || throw(ArgumentError(
        "$caller: output $(output.name) has a ragged element axis without semantic group labels"))
    if length(dims) == 1
        if output.role in (:linear_predictor, :posterior_predictive, :pointwise_loglik)
            return :observation => Base.OneTo(only(dims))
        elseif output.role === :parameter && isnothing(output.component) && isempty(output.predictors)
            # An explicitly declared vector's positions ARE its coordinates.
            return :element => Base.OneTo(only(dims))
        end
    end
    throw(ArgumentError(
        "$caller: descriptor cannot name the element axis of output $(output.name) " *
        "(role=$(output.role), shape=$dims)"))
end

function BRM.brm_output_tree(d::BRMDescriptor, draws::AbstractMatrix, names;
                              logical, role=nothing)
    BRM._brm_check_draw_names(draws, names)
    output = _output(d, logical, role)
    indices = brm_output_coordinates(output, names)
    ordered = BRM._brm_element_coordinates(output, names)
    axis = _output_axis(output, "brm_output_tree")
    expected = prod(output.dims)
    length(indices) == expected || throw(DimensionMismatch(
        "brm_output_tree: $(output.name) needs $expected elements, got $(length(indices))"))
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
            "rk_draw_tree: cannot name the value axes of $name with type $(typeof(value))"))
        TreeData(value, :element => axes(value, 1))
    end
    TreeData(leaves, :draw => axes(values, 1))
end

function BRM.rk_draw_tree(values::NamedTuple)
    fields = map(name -> _rk_value_tree(getproperty(values, name), name), keys(values))
    TreeData(:value => NamedTuple{keys(values)}(fields))
end

BRM.rk_draw_tree(rk::RKBRMI, U::AbstractMatrix, names) =
    rk_draw_tree(rk_draw_values(rk, U, names))

BRM.rk_draw_tree(rk::RKBRMI, chains::AbstractVector{<:AbstractMatrix}, names) =
    _chains(U -> rk_draw_tree(rk, U, names), chains)

end
