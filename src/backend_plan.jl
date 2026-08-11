# Shared BRMI analysis used before a concrete backend emits or executes a model.
# Nothing in this file may depend on StanBlocks (or on any other backend): package
# extensions must be able to consume this layer after all backends become weak
# dependencies.

"""
    _BRMBackendContext

Backend-neutral facts collected from a data-bound [`BRMI`](@ref): raw data,
likelihood-decorator prepass state, and the observation row axis associated with
each referenced target. Concrete backends own all later representation choices.
"""
struct _BRMBackendContext{P<:BRMI,D<:AbstractDict,PP<:AbstractDict,TO<:AbstractDict}
    parent::P
    data::D
    prepass::PP
    target_obs::TO
end

# Materialise a user data vector without silently deleting missing rows. A
# likelihood decorator may deliberately keep its response out of this generic
# data dict and model the missing values itself; that decision is made by the
# prepass below.
function _brm_data_vec(col_name::Symbol, raw)
    if eltype(raw) >: Missing
        any(ismissing, raw) && error(
            "BRM backend lowering: data column `$col_name` contains `missing` " *
            "values. BRM never silently drops rows. Either drop/impute them " *
            "before building the model, or use a response decorator that " *
            "models missing observations explicitly.")
        return collect(nonmissingtype(eltype(raw)), raw)
    end
    raw
end

# Generic likelihood-LHS prepass. Each decorator claims its own bucket in the
# context; no concrete backend or walker needs to know the decorator list.
_brm_visit_op!(_, _) = nothing
_brm_visit_op!(ctx, op::NamedColumn) = _brm_visit_op!(ctx, parent(op))
_brm_visit_op!(ctx, op::ExprColumn{typeof(~)}) =
    _brm_register_sampling_lhs!(ctx, first(getargs(op, 2)))

_brm_register_sampling_lhs!(_, _) = nothing
_brm_register_sampling_lhs!(ctx::AbstractDict,
                            lhs::ExprColumn{typeof(mi)}) =
    _brm_register_skipped_inner!(ctx, only(getargs(lhs)))
_brm_register_sampling_lhs!(ctx::AbstractDict,
                            lhs::ExprColumn{typeof(ragged)}) =
    _brm_register_skipped_inner!(ctx, first(getargs(lhs)))

_brm_register_skipped_inner!(_, _) = nothing
_brm_register_skipped_inner!(ctx::AbstractDict, inner::NamedColumn) =
    (push!(get!(ctx, :skip_data, Set{Symbol}()), name(inner)); nothing)

# Multi-membership-only sources are Julia-side preprocessing inputs, not raw
# backend data. A source also used elsewhere remains ordinary model data.
_brm_collect_mm_sources!(_, _) = nothing
_brm_collect_mm_sources!(out, x::NamedColumn) =
    _brm_collect_mm_sources!(out, parent(x))
function _brm_collect_mm_sources!(out, x::ExprColumn)
    foreach(a -> _brm_collect_mm_sources!(out, a), getargs(x))
    foreach(v -> _brm_collect_mm_sources!(out, v), values(getkwargs(x)))
end
function _brm_collect_mm_sources!(out, x::MultiMembershipTerm)
    foreach(g -> push!(out, name(g)), getargs(x))
    weights = getfield(x, :weights)
    isnothing(weights) || foreach(w -> push!(out, name(w)), weights)
end

_brm_collect_non_mm_sources!(_, _) = nothing
function _brm_collect_non_mm_sources!(out, x::NamedColumn)
    parent(x) isa DataColumn && push!(out, name(x))
    nothing
end
function _brm_collect_non_mm_sources!(out, x::ExprColumn)
    foreach(a -> _brm_collect_non_mm_sources!(out, a), getargs(x))
    foreach(v -> _brm_collect_non_mm_sources!(out, v), values(getkwargs(x)))
end
_brm_collect_non_mm_sources!(_, ::MultiMembershipTerm) = nothing

_brm_collect_data!(_data, _x; skip=Set{Symbol}()) = nothing
function _brm_collect_data!(data, x::NamedColumn; skip=Set{Symbol}())
    backing = parent(x)
    if backing isa DataColumn && !(name(x) in skip)
        data[name(x)] = _brm_data_vec(name(x), parent(backing))
    end
    _brm_collect_data!(data, backing; skip)
end
function _brm_collect_data!(data, x::ExprColumn; skip=Set{Symbol}())
    foreach(a -> _brm_collect_data!(data, a; skip), getargs(x))
    foreach(v -> _brm_collect_data!(data, v; skip), values(getkwargs(x)))
end
function _brm_collect_data!(data, x::MultiMembershipTerm; skip=Set{Symbol}())
    foreach(a -> _brm_collect_data!(data, a; skip), getargs(x))
    weights = getfield(x, :weights)
    isnothing(weights) || foreach(a -> _brm_collect_data!(data, a; skip), weights)
end

_brm_is_nothing_column(x::NamedColumn) =
    name(x) === :nothing && parent(x) isa MissingColumn

_brm_observation_name(_) = nothing
_brm_observation_name(lhs::NamedColumn) =
    parent(lhs) isa DataColumn ? name(lhs) : nothing
_brm_observation_name(lhs::ExprColumn{typeof(ragged)}) =
    _brm_observation_name(first(getargs(lhs)))
function _brm_observation_name(lhs::ExprColumn)
    args = getargs(lhs)
    length(args) == 1 ? _brm_observation_name(only(args)) : nothing
end

_brm_collect_rhs_refs!(_target_obs, _x, _obs_name) = nothing
function _brm_collect_rhs_refs!(target_obs, x::NamedColumn, obs_name)
    !_brm_is_nothing_column(x) && get!(target_obs, name(x), obs_name)
    nothing
end
function _brm_collect_rhs_refs!(target_obs, x::ExprColumn, obs_name)
    foreach(a -> _brm_collect_rhs_refs!(target_obs, a, obs_name), getargs(x))
    foreach(v -> _brm_collect_rhs_refs!(target_obs, v, obs_name),
            values(getkwargs(x)))
end

function _brm_collect_target_obs(brmi::BRMI)
    target_obs = Dict{Symbol,Symbol}()
    for op_nc in values(brmi.operations)
        op_nc isa NamedColumn || continue
        op = parent(op_nc)
        op isa ExprColumn || continue
        getf(op) === (~) || continue
        lhs, rhs = getargs(op, 2)
        obs_name = _brm_observation_name(lhs)
        isnothing(obs_name) && continue
        get!(target_obs, obs_name, obs_name)
        _brm_collect_rhs_refs!(target_obs, rhs, obs_name)
    end
    target_obs
end

function _brm_backend_context(brmi::BRMI;
                              data::AbstractDict=Dict{Symbol,Any}())
    prepass = Dict{Symbol,Any}()
    for op in values(brmi.operations)
        _brm_visit_op!(prepass, op)
    end

    skip_data = copy(get(prepass, :skip_data, Set{Symbol}()))
    mm_sources = Set{Symbol}()
    non_mm_sources = Set{Symbol}()
    for op in values(brmi.operations)
        payload = op isa NamedColumn ? parent(op) : op
        _brm_collect_mm_sources!(mm_sources, payload)
        _brm_collect_non_mm_sources!(non_mm_sources, payload)
    end
    union!(skip_data, setdiff(mm_sources, non_mm_sources))
    prepass[:skip_data] = skip_data

    for op in values(brmi.operations)
        _brm_collect_data!(data, op; skip=skip_data)
    end

    _BRMBackendContext(brmi, data, prepass, _brm_collect_target_obs(brmi))
end

# ---- narrow shared population design --------------------------------------

struct _BRMPopulationColumn{V<:AbstractVector}
    label::Symbol
    source::Union{Nothing,Symbol}
    values::V
end

struct _BRMPopulationDesign{C<:Tuple,M<:AbstractMatrix}
    target::Symbol
    columns::C
    matrix::M
    row_source::Symbol
end

_brm_additive_terms(rhs) = begin
    terms = Any[]
    _brm_collect_additive_terms!(terms, rhs)
    terms
end
_brm_collect_additive_terms!(terms, x::ExprColumn{typeof(+)}) =
    foreach(a -> _brm_collect_additive_terms!(terms, a), getargs(x))
function _brm_collect_additive_terms!(terms, x::Integer)
    x == 0 || push!(terms, x)
    nothing
end
_brm_collect_additive_terms!(terms, x) = push!(terms, x)

function _brm_population_column(term::Integer)
    term == 1 || return nothing
    (; label=:Intercept, source=nothing, values=nothing)
end
function _brm_population_column(term::NamedColumn)
    backing = parent(term)
    backing isa DataColumn || return nothing
    raw = parent(backing)
    raw isa AbstractVector{<:Real} || return nothing
    eltype(raw) <: Integer && return nothing
    (; label=name(term), source=name(term), values=collect(Float64, raw))
end
_brm_population_column(_term) = nothing

"""
    _brm_simple_population_design(target, rhs, data, obs_name; required=false)

Materialise the backend-neutral population design for the first common surface:
an additive intercept and continuous raw-data columns. Returns `nothing` for a
term requiring richer lowering unless `required=true`, in which case it fails
loudly. Both SBBRMI and Turing consume this exact representation.
"""
function _brm_simple_population_design(target::Symbol, rhs,
                                       data::AbstractDict,
                                       obs_name::Union{Nothing,Symbol};
                                       required::Bool=false)
    raw_columns = Any[]
    for term in _brm_additive_terms(rhs)
        column = _brm_population_column(term)
        if isnothing(column)
            required && error(
                "BRM backend lowering: predictor `$target` contains unsupported " *
                "population term `$(repr(term))`; the shared initial surface " *
                "supports only `1` and continuous raw-data columns")
            return nothing
        end
        push!(raw_columns, column)
    end
    isempty(raw_columns) && begin
        required && error("BRM backend lowering: predictor `$target` has no terms")
        return nothing
    end

    concrete = filter(c -> !isnothing(c.source), raw_columns)
    row_source = if isempty(concrete)
        isnothing(obs_name) && begin
            required && error(
                "BRM backend lowering: intercept-only predictor `$target` has no " *
                "observed response from which to determine its row axis")
            return nothing
        end
        obs_name
    else
        first(concrete).source
    end
    haskey(data, row_source) || begin
        required && error(
            "BRM backend lowering: row-axis source `$row_source` for predictor " *
            "`$target` is absent from materialized data")
        return nothing
    end
    n = length(data[row_source])

    columns = map(raw_columns) do column
        values = isnothing(column.source) ? ones(Float64, n) : column.values
        length(values) == n || error(
            "BRM backend lowering: predictor `$target` mixes row axes of lengths " *
            "$n and $(length(values))")
        _BRMPopulationColumn(column.label, column.source, values)
    end
    matrix = hcat((c.values for c in columns)...)
    _BRMPopulationDesign(target, Tuple(columns), matrix, row_source)
end
