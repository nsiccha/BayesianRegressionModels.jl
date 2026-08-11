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

struct _BRMPopulationPreprocess{C,R,D<:Tuple}
    kind::Symbol
    const_::C
    raw_ref::R
    dependencies::D
end
_BRMPopulationPreprocess(kind::Symbol, const_, raw_ref) =
    _BRMPopulationPreprocess(kind, const_, raw_ref, ())

struct _BRMPopulationColumn{A<:Tuple,V<:AbstractVector,P}
    label::Symbol
    effect_addresses::A
    effect_block::Symbol
    source::Union{Nothing,Symbol}
    values::V
    preprocess::P
end

struct _BRMPopulationFixedTerm{V<:AbstractVector,P}
    label::Symbol
    source::Symbol
    values::V
    preprocess::P
end

struct _BRMPopulationDesign{C<:Tuple,M<:AbstractMatrix,T<:Tuple,V<:AbstractVector}
    target::Symbol
    columns::C
    matrix::M
    row_source::Symbol
    fixed_terms::T
    fixed::V
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
    (; label=:Intercept, effect_addresses=(:Intercept,),
       effect_block=:Intercept, source=nothing,
       values=nothing, preprocess=nothing)
end
function _brm_population_column(term::NamedColumn)
    backing = parent(term)
    backing isa DataColumn || return nothing
    raw = parent(backing)
    raw isa AbstractVector{<:Real} || return nothing
    eltype(raw) <: Integer && return nothing
    (; label=name(term), effect_addresses=(name(term),),
       effect_block=name(term), source=name(term),
       values=collect(Float64, raw), preprocess=nothing)
end

_brm_fit_zscale(v::AbstractVector{<:Real}) = let
    fit = _native_ppl_fit_zscale(v, :predictor)
    (fit.mean, fit.scale)
end
_brm_apply_zscale(c::Tuple, v::AbstractVector{<:Real}) =
    (v .- c[1]) ./ c[2]
_brm_fit_center(v::AbstractVector{<:Real}) = sum(v) / length(v)
_brm_apply_center(mu::Real, v::AbstractVector{<:Real}) = v .- mu

_brm_data_expression_sources!(sources, term::NamedColumn) = begin
    parent(term) isa DataColumn && push!(sources, name(term))
    sources
end
_brm_data_expression_sources!(sources, term::ExprColumn) = begin
    foreach(arg -> _brm_data_expression_sources!(sources, arg), getargs(term))
    sources
end
_brm_data_expression_sources!(sources, _term) = sources
_brm_data_expression_sources(term) =
    _brm_data_expression_sources!(Symbol[], term)

_brm_materialize_data_expression(x::Number) = x
_brm_materialize_data_expression(x::NamedColumn) = begin
    backing = parent(x)
    backing isa DataColumn ? parent(backing) : nothing
end
function _brm_materialize_data_expression(x::ExprColumn)
    isempty(getkwargs(x)) || return nothing
    args = map(_brm_materialize_data_expression, getargs(x))
    any(isnothing, args) && return nothing
    broadcast(getf(x), args...)
end
_brm_materialize_data_expression(_x) = nothing

function _brm_materialize_count_argument(argument, n::Integer, role::AbstractString;
                                         prefix="BRM backend lowering")
    raw = _brm_materialize_data_expression(argument)
    isnothing(raw) && error(
        "$prefix: $role must be a nonnegative integer constant or a pure " *
        "expression over raw numeric data columns")
    values = raw isa AbstractVector ? raw : fill(raw, n)
    length(values) == n || error(
        "$prefix: $role has $(length(values)) rows; expected $n")
    all(x -> x isa Integer && !(x isa Bool) && x >= 0, values) || error(
        "$prefix: $role must contain only nonnegative integers")
    Int.(values)
end

function _brm_validate_binomial_response(response::AbstractVector, trials,
                                         target::Symbol;
                                         prefix="BRM backend lowering")
    length(response) == length(trials) || error(
        "$prefix: Binomial response `$target` has $(length(response)) rows; " *
        "expected $(length(trials))")
    all(eachindex(response)) do i
        y = response[i]
        y isa Integer && !(y isa Bool) && 0 <= y <= trials[i]
    end || error(
        "$prefix: Binomial response `$target` must contain integer counts " *
        "between zero and its row's trial count")
    Int.(response)
end

_brm_wrapper_col_name(prefix::Symbol, inner::NamedColumn) =
    Symbol(prefix, :_, name(inner))
_brm_wrapper_col_name(prefix::Symbol, inner) =
    Symbol(prefix, :_expr_, string(hash(inner); base=16)[1:8])

function _brm_population_column(term::ExprColumn)
    f = getf(term)
    kind = f === zscale ? :zscale :
           f === standardize ? :standardize :
           f === center ? :center : nothing
    if isnothing(kind)
        values = _brm_materialize_data_expression(term)
        values isa AbstractVector{<:Real} || return nothing
        sources = _brm_data_expression_sources(term)
        isempty(sources) && return nothing
        label = _brm_wrapper_col_name(Symbol(getf(term)), term)
        preprocess = _BRMPopulationPreprocess(
            :protect, nothing, term)
        return (; label, effect_addresses=(label,), effect_block=label,
                source=first(sources), values=collect(Float64, values),
                preprocess)
    end
    args = getargs(term)
    length(args) == 1 || return nothing
    inner = only(args)
    inner isa NamedColumn || return nothing
    backing = parent(inner)
    backing isa DataColumn || return nothing
    raw = parent(backing)
    raw isa AbstractVector{<:Real} || return nothing

    values = collect(Float64, raw)
    const_ = kind === :center ? _brm_fit_center(values) :
                               _brm_fit_zscale(values)
    transformed = kind === :center ? _brm_apply_center(const_, values) :
                                     _brm_apply_zscale(const_, values)
    label = Symbol(kind, :_, name(inner))
    preprocess = _BRMPopulationPreprocess(kind, const_, inner)
    (; label, effect_addresses=(label,), effect_block=label,
       source=name(inner), values=transformed, preprocess)
end
_brm_population_column(_term) = nothing

function _brm_population_fixed_term(term)
    term isa ExprColumn && getf(term) === offset || return nothing
    args = getargs(term)
    length(args) == 1 || error(
        "BRM backend lowering: `offset(x)` expects exactly one positional " *
        "argument, got $(length(args))")
    isempty(getkwargs(term)) || error(
        "BRM backend lowering: `offset(x)` does not accept keyword arguments")
    inner = only(args)
    values = _brm_materialize_data_expression(inner)
    values isa AbstractVector{<:Real} || error(
        "BRM backend lowering: direct-BRMI `offset(...)` currently requires a " *
        "pure expression over raw numeric data columns")
    sources = _brm_data_expression_sources(inner)
    isempty(sources) && error(
        "BRM backend lowering: `offset(...)` has no raw data row axis")
    label = _brm_wrapper_col_name(:offset, inner)
    preprocess = _BRMPopulationPreprocess(:protect, nothing, inner)
    _BRMPopulationFixedTerm(
        label, first(sources), collect(Float64, values), preprocess)
end

_brm_fit_levels(raw::CA.CategoricalVector) = CA.levels(raw)
_brm_level_index(raw::CA.CategoricalVector) =
    length(CA.levels(raw)), Int.(CA.levelcode.(raw))
_brm_level_index(raw::AbstractVector) = begin
    levels = _brm_fit_levels(raw)
    lookup = Dict(level => i for (i, level) in enumerate(levels))
    length(levels), Int[lookup[level] for level in raw]
end

function _brm_categorical_population_columns(
        raw, source::Symbol, block::Symbol,
        effect_addresses::Tuple=(block,); ref::Integer=1)
    is_categorical = raw isa CA.CategoricalVector ||
                     (raw isa AbstractVector && eltype(raw) <: Integer)
    is_categorical || return nothing
    n_levels, indices = _brm_level_index(raw)
    n_levels >= 2 || return nothing
    levels = _brm_fit_levels(raw)
    Tuple(begin
        label = Symbol(block, :_lvl_, level)
        values = Float64[index == level ? 1.0 : 0.0 for index in indices]
        preprocess = _BRMPopulationPreprocess(
            :population_factor_dummy,
            (; levels, level, n_levels, ref), source)
        (; label, effect_addresses, effect_block=block, source, values,
           preprocess)
    end for level in 2:n_levels)
end

function _brm_population_columns(term::NamedColumn)
    backing = parent(term)
    categorical = backing isa DataColumn ?
        _brm_categorical_population_columns(
            parent(backing), name(term), name(term)) : nothing
    !isnothing(categorical) && return categorical
    column = _brm_population_column(term)
    isnothing(column) ? nothing : (column,)
end

function _brm_population_columns(term::ExprColumn{typeof(factor)})
    args = getargs(term)
    length(args) == 1 || return nothing
    inner = only(args)
    inner isa NamedColumn || return nothing
    backing = parent(inner)
    backing isa DataColumn || return nothing
    raw = parent(backing)
    raw isa AbstractVector && eltype(raw) <: Integer || return nothing
    kwargs = getkwargs(term)
    all(k -> k === :ref, keys(kwargs)) || return nothing
    ref_raw = get(kwargs, :ref, 1)
    ref_raw isa Integer || return nothing
    1 <= ref_raw <= maximum(raw) || error(
        "BRM backend lowering: `factor($(name(inner)); ref=$ref_raw)` ref " *
        "out of range (max level $(maximum(raw)))")

    source = name(inner)
    block = ref_raw == 1 ? source : Symbol(source, :__ref_, ref_raw)
    recoded = ref_raw == 1 ? raw :
        Int[value == ref_raw ? 1 : value == 1 ? ref_raw : value for value in raw]
    addresses = ref_raw == 1 ? (source,) : (block, source)
    _brm_categorical_population_columns(
        recoded, source, block, addresses; ref=ref_raw)
end
_brm_population_columns(term) = let column = _brm_population_column(term)
    isnothing(column) ? nothing : (column,)
end

_brm_population_column_is_categorical(column) =
    !isnothing(column.preprocess) &&
    column.preprocess.kind === :population_factor_dummy

function _brm_interaction_population_column(left, right)
    left_cat = _brm_population_column_is_categorical(left)
    right_cat = _brm_population_column_is_categorical(right)
    # StanBlocks names mixed interactions with the continuous operand first,
    # independent of surface order. Preserve that exact public effect address.
    if left_cat && !right_cat
        return _brm_interaction_population_column(right, left)
    end
    length(left.values) == length(right.values) || error(
        "BRM backend lowering: interaction `$(left.label) & $(right.label)` " *
        "mixes row axes of lengths $(length(left.values)) and " *
        "$(length(right.values))")

    label = Symbol(:int_, left.label, :_x_, right.label)
    values = left.values .* right.values
    dependencies = Tuple(c for c in (left, right)
                         if !isnothing(c.preprocess))
    preprocess = _BRMPopulationPreprocess(
        :interaction, nothing, (left.label, right.label), dependencies)
    (; label, effect_addresses=(label,), effect_block=label,
       source=left.source, values, preprocess)
end

function _brm_population_columns(term::ExprColumn{typeof(&)})
    args = getargs(term)
    length(args) == 2 || return nothing
    left = _brm_population_columns(args[1])
    right = _brm_population_columns(args[2])
    (isnothing(left) || isnothing(right) ||
     any(c -> isnothing(c.source), left) ||
     any(c -> isnothing(c.source), right)) && return nothing
    Tuple(_brm_interaction_population_column(l, r)
          for l in left for r in right)
end

"""
    _brm_simple_population_design(target, rhs, data, obs_name; required=false)

Materialise the backend-neutral population design for the first common surface:
an additive intercept, continuous raw-data columns, pure numeric data
expressions, fitted `zscale`/`standardize`/`center` columns, pairwise
continuous/categorical interactions, treatment contrasts for integer or
`CategoricalVector` columns, and pure data-derived fixed `offset(...)` terms.
Returns `nothing` for a term requiring richer lowering unless `required=true`,
in which case it fails loudly. SBBRMI and Turing consume the shared
coefficient-bearing representation; Turing also consumes the materialized
fixed-offset vector. Categorical columns reuse SBBRMI's established ordered
level-coding primitive.
"""
function _brm_simple_population_design(target::Symbol, rhs,
                                       data::AbstractDict,
                                       obs_name::Union{Nothing,Symbol};
                                       required::Bool=false)
    raw_columns = Any[]
    fixed_terms = _BRMPopulationFixedTerm[]
    for term in _brm_additive_terms(rhs)
        fixed = _brm_population_fixed_term(term)
        if !isnothing(fixed)
            push!(fixed_terms, fixed)
            continue
        end
        columns = _brm_population_columns(term)
        if isnothing(columns)
            required && error(
                "BRM backend lowering: predictor `$target` contains unsupported " *
                "population term `$(repr(term))`; the shared initial surface " *
                "supports `1`, continuous raw-data columns, pure numeric data " *
                "expressions and fixed data-derived `offset(...)` terms, plus " *
                "`zscale`, `standardize`, and `center` of one numeric column, " *
                "pairwise continuous/categorical interactions, and ordered " *
                "treatment contrasts for integer or `CategoricalVector` columns")
            return nothing
        end
        append!(raw_columns, columns)
    end
    isempty(raw_columns) && begin
        required && error("BRM backend lowering: predictor `$target` has no terms")
        return nothing
    end

    concrete = filter(c -> !isnothing(c.source), raw_columns)
    row_sources = Symbol[c.source for c in concrete]
    append!(row_sources, (term.source for term in fixed_terms))
    row_source = if isempty(concrete)
        if !isempty(row_sources)
            first(row_sources)
        elseif isnothing(obs_name)
            required && error(
                "BRM backend lowering: intercept-only predictor `$target` has no " *
                "observed response from which to determine its row axis")
            return nothing
        else
            obs_name
        end
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
        _BRMPopulationColumn(
            column.label, column.effect_addresses, column.effect_block,
            column.source, values, column.preprocess)
    end
    matrix = hcat((c.values for c in columns)...)
    fixed = zeros(Float64, n)
    for term in fixed_terms
        length(term.values) == n || error(
            "BRM backend lowering: predictor `$target` mixes row axes of " *
            "lengths $n and $(length(term.values)) in fixed offset `$(term.label)`")
        fixed .+= term.values
    end
    _BRMPopulationDesign(
        target, Tuple(columns), matrix, row_source, Tuple(fixed_terms), fixed)
end

struct _BRMPopulationPredictor{F,D<:_BRMPopulationDesign}
    name::Symbol
    link_lhs_fn::F
    emitted_name::Symbol
    design::D
end

_brm_lp_emitted_name(name::Symbol, link_lhs_fn) =
    link_lhs_fn === identity ? name : Symbol(nameof(link_lhs_fn), :_, name)

function _brm_simple_population_predictor(brmi::BRMI, target::Symbol,
                                          context::_BRMBackendContext;
                                          required::Bool=false)
    op = linear_predictor_op(brmi, target)
    if isnothing(op) || getf(op) !== (~)
        required && error(
            "BRM backend lowering: predictor `$target` has no population formula")
        return nothing
    end
    lhs, rhs = getargs(op, 2)
    peeled = _peel_lp_lhs(lhs)
    if isnothing(peeled) || last(peeled) !== target
        required && error(
            "BRM backend lowering: predictor `$target` has an unsupported LHS")
        return nothing
    end
    link_lhs_fn, name = peeled
    design = _brm_simple_population_design(
        name, rhs, context.data, get(context.target_obs, name, nothing);
        required)
    isnothing(design) && return nothing
    _BRMPopulationPredictor(
        name, link_lhs_fn, _brm_lp_emitted_name(name, link_lhs_fn), design)
end

# ---- shared population-effect prior semantics -----------------------------

# This is a representation test, not a Stan lowering decision. Keep it in the
# backend-neutral layer so a weak-dependency backend can validate distribution
# declarations without loading SBBRMI.
_as_distribution_type(::Type{T}) where {T<:Distribution} = T
_as_distribution_type(_) = nothing

_brm_effect_specificity(spec) =
    (spec.predictor === _EFFECT_COLON ? 0 : 1) +
    (spec.coefficient === _EFFECT_COLON ? 0 : 1)
_brm_effect_spelling(spec) =
    "effect($(spec.predictor), $(spec.coefficient))"

function _brm_validate_population_effect_spec(spec;
                                               prefix="BRM backend lowering")
    T = _as_distribution_type(spec.family)
    (!isnothing(T) && T <: Normal) || error(
        "$prefix: population `effect(...)` overrides currently support only " *
        "`Normal(location, scale)`; got `$(spec.family)`. Ordinary scalar " *
        "parameter priors remain available for other supported families.")
    isempty(spec.keywords) || error(
        "$prefix: `effect(...) ~ Normal(...)` does not accept distribution " *
        "keywords; put bounds on an explicitly declared scalar parameter instead")
    _brm_normal_effect_args(spec.expression; prefix)
    nothing
end

# One precedence engine is shared by the simple direct-BRMI plan and SBBRMI's
# richer population/categorical resolver. A cell stores both the exact parsed
# RHS and the specificity that won it; formula order never decides a tie.
function _brm_claim_effect_prior!(get_cell, set_cell!, spec, what;
                                  prefix="BRM backend lowering")
    rank = _brm_effect_specificity(spec)
    spelling = _brm_effect_spelling(spec)
    held = get_cell()
    if isnothing(held) || rank > held.rank
        set_cell!((; expression=spec.expression, rank, spelling))
    elseif rank == held.rank
        error("$prefix: `$spelling` and `$(held.spelling)` are equally specific " *
              "and both set the prior for $what. Neither wins — make one of " *
              "them more specific, or drop it.")
    end
    nothing
end

function _brm_normal_effect_args(rhs::ExprColumn;
                                 prefix="BRM backend lowering")
    args = getargs(rhs)
    length(args) <= 2 || error(
        "$prefix: `effect(...) ~ Normal(...)` must lower to exactly location " *
        "and scale arguments, got $(length(args))")
    isempty(args) && return (0.0, 1.0)
    length(args) == 1 && return (only(args), 1.0)
    args
end

_brm_numeric_constant(x::Real) = Float64(x)
function _brm_numeric_constant(x::ExprColumn)
    isempty(getkwargs(x)) || return nothing
    values = map(_brm_numeric_constant, getargs(x))
    any(isnothing, values) && return nothing
    value = try
        getf(x)(values...)
    catch
        return nothing
    end
    value isa Real ? Float64(value) : nothing
end
_brm_numeric_constant(_x) = nothing

"""
    _brm_simple_population_effect_overrides(
        brmi, design; available_predictors=(design.target,))

Resolve formula-level `effect(...) ~ Normal(...)` statements against a narrow
shared population design. The output is aligned 1:1 with `design.columns` and
contains the winning parsed RHS (or `nothing` for the default Normal(0, 1)).
An `effect(lp, categorical_column)` address fans out over that column's K-1
treatment contrasts, matching SBBRMI's one-prior-per-contrast-block contract.
For a distributional likelihood, `available_predictors` names its complete
predictor set: a prior targeting a peer is ignored by this component, while a
prior targeting no member still fails loudly.
"""
function _brm_simple_population_effect_overrides(brmi::BRMI,
                                                 design::_BRMPopulationDesign;
                                                 prefix="BRM backend lowering",
                                                 available_predictors=(design.target,))
    specs = effect_priors(brmi)
    isempty(specs) && return nothing

    labels = Symbol[c.label for c in design.columns]
    address_blocks = Dict{Symbol,Set{Symbol}}()
    for column in design.columns, address in column.effect_addresses
        push!(get!(address_blocks, address, Set{Symbol}()), column.effect_block)
    end
    ambiguous = Dict(address => blocks for (address, blocks) in address_blocks
                     if length(blocks) > 1)
    available = Symbol[address for (address, blocks) in address_blocks
                       if length(blocks) == 1]
    cells = Any[nothing for _ in labels]
    for spec in specs
        _brm_validate_population_effect_spec(spec; prefix)
        all_predictors = spec.predictor === _EFFECT_COLON
        if !all_predictors && spec.predictor !== design.target
            spec.predictor in available_predictors && continue
            error("$prefix: `$(_brm_effect_spelling(spec))` names no linear " *
                  "predictor in this backend plan. Available predictors: " *
                  "$(join(available_predictors, ", ")).")
        end

        indices = if spec.coefficient === _EFFECT_COLON
            eachindex(labels)
        else
            if haskey(ambiguous, spec.coefficient)
                blocks = sort!(collect(ambiguous[spec.coefficient]))
                error("$prefix: `$(spec.coefficient)` ambiguously names " *
                      "categorical contrast blocks $(join(blocks, ", ")). " *
                      "Address an emitted block name explicitly.")
            end
            idxs = findall(c -> spec.coefficient in c.effect_addresses,
                           design.columns)
            isempty(idxs) && error(
                "$prefix: `$(spec.coefficient)` is not a population coefficient " *
                "of `$(design.target)`. Available labels: " *
                "$(join(sort!(available), ", ")).")
            idxs
        end
        for idx in indices
            _brm_claim_effect_prior!(
                () -> cells[idx], v -> (cells[idx] = v), spec,
                "`$(design.target)`'s `$(labels[idx])` column"; prefix)
        end
    end
    Any[isnothing(cell) ? nothing : cell.expression for cell in cells]
end

function _brm_materialize_normal_effect_priors(overrides, n::Integer;
                                               prefix="BRM backend lowering")
    isnothing(overrides) && return (zeros(Float64, n), ones(Float64, n))
    length(overrides) == n || error(
        "$prefix: internal effect-prior alignment error: $(length(overrides)) " *
        "priors for $n population columns")
    location = zeros(Float64, n)
    scale = ones(Float64, n)
    for i in eachindex(overrides)
        isnothing(overrides[i]) && continue
        raw_location, raw_scale = _brm_normal_effect_args(overrides[i]; prefix)
        loc = _brm_numeric_constant(raw_location)
        sd = _brm_numeric_constant(raw_scale)
        isnothing(loc) && error(
            "$prefix: population-effect Normal location must be a numeric constant")
        isnothing(sd) && error(
            "$prefix: population-effect Normal scale must be a numeric constant")
        isfinite(loc) || error(
            "$prefix: population-effect Normal location must be finite")
        isfinite(sd) && sd > 0 || error(
            "$prefix: population-effect Normal scale must be finite and positive")
        location[i] = loc
        scale[i] = sd
    end
    location, scale
end

# Operation keys for the declarations consumed by the resolver above. These
# keys are model statements, not stray operations, and must be admitted by a
# strict direct-BRMI backend after their semantics have been validated.
function _brm_population_effect_operation_keys(brmi::BRMI)
    out = Set{Symbol}()
    for (key, op_nc) in pairs(brmi.operations)
        op = _named_op(op_nc)
        isnothing(op) && continue
        getf(op) === (~) || continue
        lhs, rhs = getargs(op, 2)
        lhs_e = _as_expr_column(lhs)
        rhs_e = _as_expr_column(rhs)
        isnothing(lhs_e) && continue
        isnothing(rhs_e) && continue
        getf(lhs_e) === effect || continue
        address = getargs(lhs_e)
        length(address) == 2 || continue
        first(address) in _NON_EFFECT_CLASSES && continue
        getf(rhs_e) === r2d2 && continue
        push!(out, key)
    end
    out
end
