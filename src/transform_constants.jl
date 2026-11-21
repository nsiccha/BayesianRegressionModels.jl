# ==============================================================================
# transform_constants.jl — the fitted constants of `center` / `zscale` /
# `standardize` terms, read from what each backend applies.
#
# A transformed predictor's coefficient multiplies `(x - center) / scale`,
# with `center`/`scale` fitted once on the training data and frozen for replay.
# Consumers that plot or evaluate a coefficient's contribution need exactly
# those constants. They are read here from the backend's own record — the
# SBBRMI `preproc` entry `reprocess` re-applies, or the anchors an RKBRMI
# graph evaluates — and never re-derived from consumer data.
# ==============================================================================

"""
    brm_transform_constants(x; submodel=nothing) -> Tuple
    brm_transform_constants(x, name::Symbol; submodel=nothing) -> NamedTuple

The fitted constants of every `center`, `zscale` and `standardize` term of a
fitted model, or of the one named `name`. `x` is an `SBBRMI`, its
`GenerativePlan`, a `BRMDescriptor` of either, or an `RKBRMI`.

Pass `submodel=(collection, key)` for the transforms of one repeated member.
An intercept-only member has an empty tuple.

Each record is

    (; name, kind, column, source, center, scale, observed_only)

- `name` — the transformed value's name, which is also its population
  coefficient label (`:standardize_age_yr` for `standardize(age_yr)`). An
  expression inner has a hashed name (`:zscale_expr_301a4a5c` for
  `zscale(log(w))`); list the records to find it by `source`. A per-group
  field read at row positions, `standardize(age_yr)[subject]`, is
  `:standardize_age_yr_subject`; its constants are fitted over `age_yr` and
  each row's value is `(age_yr[subject] - center) / scale`.
- `kind` — `:center`, `:zscale` or `:standardize`.
- `column` — the single name the transform reads (`:age_yr`), or `nothing`
  for an expression inner.
- `source` — the inner as written: a `Symbol` (`:age_yr`) or an `Expr`
  (`:(log(w))`).
- `center`, `scale` — the fitted constants: the transformed value is
  `(x .- center) ./ scale`, exactly as the model applies it. `center` has
  `scale == 1.0`; `zscale`/`standardize` use the sample SD (`n - 1`).
- `observed_only` — `true` when the inner reads an `mi(...)` covariate: the
  constants are fitted on its observed entries only and the transform is
  applied inside the model. The record keeps the coefficient's name either way.

The constants are the ones the backend applies, so they follow replay:
`reprocess(sb, new_df)` (frozen, the default) reports the training constants,
and `reprocess(sb, new_df; freeze_constants=false)` the refitted ones. They are
never recomputed here. Numerically, `zscale`/`standardize` use a streaming
mean and scaled sums of squares, so a hand `mean`/`std` agrees only to
floating-point roundoff. The SB `center` mean is `sum(x) / length(x)`, the RK
graph's the streaming mean, so the two backends' `center` agree to roundoff.

A name that is not a fitted `center`/`zscale`/`standardize` term errors,
listing the fitted names. On an `RKBRMI`, a transform used only as an `&`
interaction operand is staged under an internal name and is not listed.
"""
function brm_transform_constants end

function _brm_transform_member(model, submodel)
    if submodel isa Tuple{Symbol,Union{Int,Symbol}}
        collection, key = submodel
        for axis in _brm_collection_axes(model)
            axis.collection === collection || continue
            index = findfirst(isequal(key), axis.keys)
            isnothing(index) || return axis, index
        end
    end
    throw(ArgumentError("brm_transform_constants: unknown record member `$submodel`; use (collection, key)"))
end

function brm_transform_constants(x::Union{SBBRMI,GenerativePlan}; submodel=nothing)
    isnothing(submodel) && return _brm_transform_constants(x.preproc)
    axis, i = _brm_transform_member(x.parent, submodel)
    fits = _brm_member_fits(x.preproc, axis)
    isnothing(fits) ? () : brm_transform_constants(fits[i])
end
brm_transform_constants(d::BRMDescriptor; submodel=nothing) =
    brm_transform_constants(d.plan; submodel)

function brm_transform_constants(x, name::Symbol; submodel=nothing)
    records = brm_transform_constants(x; submodel)
    index = findfirst(record -> record.name === name, records)
    isnothing(index) || return records[index]
    scope = isnothing(submodel) ? "this model$(_brm_transform_missing_reason(x, name))" : "member `$submodel`"
    error("brm_transform_constants: `$name` is not a fitted center/zscale/" *
          "standardize term of $scope. " *
          "Fitted transforms: $(Tuple(record.name for record in records)).")
end

_brm_transform_record(name::Symbol, kind::Symbol, inner, center, scale,
                      observed_only::Bool) =
    (; name, kind, column=_brm_transform_column(inner),
       source=_brm_transform_source(inner), center=Float64(center),
       scale=Float64(scale), observed_only)

_brm_transform_column(inner::NamedColumn) = name(inner)
# `standardize(x)[g]` fits over `x`, which names the record.
_brm_transform_column(inner::_BRMIndexedSource) = _brm_transform_column(inner.value)
_brm_transform_column(_inner) = nothing

_brm_transform_source(inner::NamedColumn) = name(inner)
_brm_transform_source(inner::ExprColumn) = Expr(:call, _brm_transform_head(getf(inner)),
    map(_brm_transform_source, getargs(inner))...,
    (Expr(:kw, key, _brm_transform_source(value)) for (key, value) in pairs(getkwargs(inner)))...)
_brm_transform_source(inner::_BRMIndexedSource) = _brm_transform_source(inner.value)
_brm_transform_source(inner) = inner

_brm_transform_head(f::Union{Function,Type}) = nameof(f)
_brm_transform_head(f) = f

# ---- SBBRMI / GenerativePlan: the preprocessing records `reprocess` applies --

function _brm_transform_constants(preproc::AbstractDict)
    records = NamedTuple[]
    for key in sort!(collect(keys(preproc)); by=string)
        record = _brm_transform_record(Val(preproc[key].kind), key, preproc[key])
        isnothing(record) || push!(records, record)
    end
    Tuple(records)
end

# Records of every other kind (factor levels, spline knots, HSGP boundaries,
# `constant(...)` values, …) are not center/scale transforms.
_brm_transform_record(::Val, _key, _entry) = nothing

function _brm_transform_record(::Union{Val{:zscale},Val{:standardize}}, key, entry)
    center, scale = entry.const_
    _brm_transform_record(key, entry.kind, entry.raw_ref, center, scale, false)
end
_brm_transform_record(::Val{:center}, key, entry) =
    _brm_transform_record(key, :center, entry.raw_ref, entry.const_, 1.0, false)

# An `mi(...)` inner records its observed-only fit under `<name>_mean`, beside
# the scalar data `<name>_scale` (`_sb_emit_mi_transform!`).
for (missing_kind, kind) in ((:missing_center, :center), (:missing_zscale, :zscale),
                             (:missing_standardize, :standardize))
    @eval function _brm_transform_record(::Val{$(QuoteNode(missing_kind))}, key, entry)
        label = string(key)
        endswith(label, "_mean") || error(
            "brm_transform_constants: internal: completed-covariate transform " *
            "record `$key` is not keyed `<name>_mean`")
        name = Symbol(chopsuffix(label, "_mean"))
        entry.const_.scale_key === Symbol(name, :_scale) || error(
            "brm_transform_constants: internal: completed-covariate transform " *
            "`$name` records scale key `$(entry.const_.scale_key)`")
        fit = entry.const_.fit
        _brm_transform_record(name, $(QuoteNode(kind)), entry.raw_ref,
                              fit.mean, fit.scale, true)
    end
end

_brm_transform_missing_reason(_x, _name) = ""
_brm_transform_missing_reason(sb::SBBRMI, name::Symbol) =
    _brm_transform_missing_reason(sb.preproc, name)
_brm_transform_missing_reason(plan::GenerativePlan, name::Symbol) =
    _brm_transform_missing_reason(plan.preproc, name)
_brm_transform_missing_reason(d::BRMDescriptor, name::Symbol) =
    _brm_transform_missing_reason(d.plan, name)
_brm_transform_missing_reason(preproc::AbstractDict, name::Symbol) =
    haskey(preproc, name) ? " (it is a fitted `$(preproc[name].kind)` record)" : ""

# ---- RKBRMI: the anchors the graph evaluates --------------------------------

# Every `center`/`zscale`/`standardize` call in the model's formulas, by the
# name its value takes (`_brm_wrapper_col_name`).
function _brm_transform_sites(brmi::BRMI)
    sites = Dict{Symbol,Any}()
    for value in _brm_operation_values(brmi)
        op = _named_op(value)
        isnothing(op) || _brm_transform_sites!(sites, op)
    end
    sites
end
function _brm_transform_sites!(sites, x::ExprColumn)
    foreach(arg -> _brm_transform_sites!(sites, arg), getargs(x))
    foreach(arg -> _brm_transform_sites!(sites, arg), values(getkwargs(x)))
    sites
end
function _brm_transform_sites!(sites,
        x::ExprColumn{<:Union{typeof(center),typeof(zscale),typeof(standardize)}})
    args = getargs(x)
    length(args) == 1 && isempty(getkwargs(x)) || return sites
    kind = nameof(getf(x))
    inner = only(args)
    sites[_brm_wrapper_col_name(kind, inner)] = (; kind, inner)
    sites
end
# `standardize(x)[g]` is the value `standardize_x_g` (`_brm_indexed_col_name`),
# fitted over `x`; any other indexing is searched like other calls.
function _brm_transform_sites!(sites, x::ExprColumn{typeof(getindex)})
    args = getargs(x)
    source = first(args)
    kind = source isa ExprColumn ? _brm_transform_kind(getf(source)) : nothing
    (length(args) == 2 && isempty(getkwargs(x)) && !isnothing(kind) &&
        length(getargs(source)) == 1 && isempty(getkwargs(source))) || return invoke(
            _brm_transform_sites!, Tuple{Any,ExprColumn}, sites, x)
    inner = only(getargs(source))
    sites[_brm_indexed_col_name(kind, inner, last(args))] = (; kind, inner)
    sites
end
_brm_transform_sites!(sites, _x) = sites

_rk_plan_derived(plan::_RKStructuralPlan) = plan.derived
_rk_plan_derived(plan::_RKValuePlan) = plan.regression.derived
_rk_plan_derived(plan::_RKHeldOutPlan) = _rk_plan_derived(plan.parent)
_rk_plan_derived(plan) = error(
    "brm_transform_constants: an RK $(nameof(typeof(plan))) evaluates no " *
    "fitted predictor transforms")

# The anchors of an emitted transform (`_rk_transform_defexpr`, and the
# completed-covariate form of `_rk_model_population_column`):
# `v .- mean(fit)`, `(v .- mean(fit)) ./ sd(fit)` or `(v .- mean(fit)) ./ 1.0`.
_rk_preparation(x, recipe::Symbol) =
    Meta.isexpr(x, :_rk_data_preparation) && first(x.args) === recipe ? x : nothing
_rk_centering_anchor(x) = Meta.isexpr(x, :call, 3) && x.args[1] === :.- ?
    _rk_preparation(x.args[3], :brm_covariate_mean) : nothing
function _rk_transform_anchors(x)
    mean = _rk_centering_anchor(x)
    isnothing(mean) || return (; mean, sd=nothing)
    Meta.isexpr(x, :call, 3) && x.args[1] === :./ || return nothing
    mean = _rk_centering_anchor(x.args[2])
    isnothing(mean) && return nothing
    x.args[3] == 1.0 && return (; mean, sd=nothing)
    sd = _rk_preparation(x.args[3], :brm_covariate_sd)
    isnothing(sd) ? nothing : (; mean, sd)
end

# An indexed value `t[g]` (`_rk_lower_indexed`) reads the staged transform `t`;
# its anchors are in that definition.
function _rk_indexed_definition(derived, expression)
    Meta.isexpr(expression, :ref, 2) && first(expression.args) isa Symbol ||
        return expression
    index = findfirst(spec -> spec.name === first(expression.args), derived)
    isnothing(index) ? expression : derived[index].expression
end

# Retain the member's fit-role definitions only when it has a transform to
# inspect. In a data-free replay its exported fits have already been scoped
# to this member by _brm_component_replay_scope.
function _rk_member_transform_plan(plan, instance)
    sites = _brm_transform_sites(instance)
    isempty(sites) && return nothing
    frozen = _brm_frozen_replay()
    saved = isnothing(frozen) || isnothing(frozen.fits) ? nothing :
        _BRMFrozenReplay(Dict{Symbol,Any}(), frozen.resample, frozen.fits, nothing)
    (; sites, derived=_rk_plan_derived(plan), frozen=saved)
end

function brm_transform_constants(rk::RKBRMI; submodel=nothing)
    if !isnothing(submodel)
        axis, i = _brm_transform_member(brm_source(rk), submodel)
        for kernel in _rk_plan_kernels(rk.plan), statement in kernel.body
            _rk_is_cell_component(statement) && first(statement.args) === last(axis.path) || continue
            members = statement.args[2].args[3].transforms
            isempty(members) && return ()
            member = members[i]
            return isnothing(member) ? () : _rk_transform_records(member.sites, member.derived, member.frozen)
        end
        return ()
    end
    frozen = if !isnothing(rk.fitted)
        state = rk.fitted.state
        _BRMFrozenReplay(state.scalars, Set{Symbol}(rk.fitted.resample), state.fits, nothing)
    end
    _rk_transform_records(_brm_transform_sites(parent(rk)), _rk_plan_derived(rk.plan), frozen)
end

_rk_transform_records(sites, derived, ::Nothing) = _rk_transform_records(sites, derived)
_rk_transform_records(sites, derived, frozen::_BRMFrozenReplay) =
    _brm_with_frozen_replay(() -> _rk_transform_records(sites, derived), frozen)

function _rk_transform_records(sites, derived)
    records = NamedTuple[]
    for name in sort!(collect(keys(sites)); by=string)
        index = findfirst(spec -> spec.name === name, derived)
        # An `&` operand is staged under an internal `rkd_*` name.
        isnothing(index) && continue
        anchors = _rk_transform_anchors(
            _rk_indexed_definition(derived, derived[index].expression))
        isnothing(anchors) && error(
            "brm_transform_constants: internal: RK value `$name` is not a " *
            "fitted transform: $(derived[index].expression)")
        # Ordinary backends hold fit-role inputs; data-free backends instead
        # hold recorded fits from the training values.
        memo = Dict{Symbol,Any}()
        evaluate(anchor) = _rk_eval_dotted(anchor, Dict{Symbol,Any}(), derived, memo)
        center = evaluate(anchors.mean)
        scale = isnothing(anchors.sd) ? 1.0 : evaluate(anchors.sd)
        site = sites[name]
        push!(records, _brm_transform_record(name, site.kind, site.inner, center, scale,
                                             !isnothing(_rk_mi_predictor_plan(site.inner))))
    end
    Tuple(records)
end

_brm_transform_missing_reason(rk::RKBRMI, name::Symbol) =
    haskey(_brm_transform_sites(parent(rk)), name) ?
        " on the RK backend (it is used only inside an `&` interaction, which the " *
        "RK program stages under an internal name)" : ""
