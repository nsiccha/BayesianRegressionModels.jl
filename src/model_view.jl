"""
    brm_model_view(model; title="Model")

An inspectable model declaration, with data/parameter roles and value ranks.
Load `HTMXObjects` to enable this view. It returns an HTMXObjects semantic
tree, usable as a route result or inside another page, with HTML, Markdown
and plain-text projections. Authored declaration docstrings and their source
provenance come from [`brm_documentation`](@ref).

Pass a `BRMI`, a data-free `BRMBuilder` or `BRMComponent`, a prepared `RKBRMI`,
or an existing `BRMDescriptor`. Reachable component source comes from
[`brm_components`](@ref), retaining separate call paths and shared definitions.
Each component call's own declarations are listed with their roles under the
exact call path (`left › b`); declarations of a call inside a plate cell are
marked as made once per iteration.
The view does not bind data, prepare, fit or evaluate a model.
Unknown dimensions stay explicit. A builder lists potential data inputs; bind
it first to distinguish observed data from latent declarations. A descriptor
adds its exact carrier dimensions and roles. Value rank (number of axes) is
never inferred from the number of free coordinates. Component input tables
distinguish required inputs and optional defaults, shown without evaluation.
Joint observations label each authored response column as response data.

RK inspection requires ReactiveKernelsPPL 0.1.93 or later. A replay that shares
an earlier build leaves current parameter dimensions unresolved: that build's
layout still describes its original binding. The data table for an `RKBRMI`
describes its retained source declaration, which a replay can also share.
"""
function brm_model_view end

# These are projections of existing compiler facts, not another model registry.
# Keep the syntax walk independent of backend planning: opening a view must not
# evaluate constants, prepare designs, or topologically sort the model again.
_brm_view_shape(::Any) = (; rank=nothing, dims=nothing)
_brm_view_shape(::Number) = (; rank=0, dims=())
_brm_view_shape(x::AbstractArray) = (; rank=ndims(x), dims=size(x))

_brm_view_rank(::Any) = nothing
_brm_view_rank(::Type{Distributions.Univariate}) = 0
_brm_view_rank(::Type{Distributions.Multivariate}) = 1
_brm_view_rank(::Type{Distributions.Matrixvariate}) = 2
function _brm_view_prior_shape(prior)
    shape = _brm_distribution_shape(prior)
    rank = isnothing(shape) ? nothing : _brm_view_rank(first(shape))
    (; rank, dims=rank === 0 ? () : nothing)
end

# `path` is the exact authored address: `(name,)` at model scope, and the call
# path plus the local name inside a component call (`(:left, :b)`), so a scoped
# local stays distinct from a literal dotted name such as `Symbol("left.b")`.
# `plate` marks a declaration made once per plate iteration (a component called
# inside a plate cell); its rank and shape then describe one iteration.
_brm_view_record(name, role; path=(name,), plate=false, rank=nothing, dims=nothing,
        detail="") =
    (; name, path, role, rank, dims, detail, plate)

function _brm_view_walk!(data, callables, seen, x::NamedColumn)
    haskey(seen, x) && return
    seen[x] = nothing
    p = parent(x)
    p isa DataColumn && (data[name(x)] = p)
    _brm_view_walk!(data, callables, seen, p)
end
function _brm_view_walk!(data, callables, seen, x::ExprColumn)
    haskey(seen, x) && return
    seen[x] = nothing
    _brm_is_statement(x) || push!(callables, getf(x))
    _brm_foreach_child(a -> _brm_view_walk!(data, callables, seen, a), x)
end
function _brm_view_walk!(data, callables, seen, x::JointResponseColumn)
    foreach(a -> _brm_view_walk!(data, callables, seen, a), joint_response_columns(x))
end
function _brm_view_walk!(data, callables, seen, items::Union{Tuple,NamedTuple,AbstractArray})
    haskey(seen, items) && return
    seen[items] = nothing
    foreach(item -> _brm_view_walk!(data, callables, seen, item), items)
end
function _brm_view_walk!(_data, callables, _seen, body::_BRMInlineBody)
    expression = _brm_inline_expr(body)
    bound = _brm_cell_bound_names!(Set{Symbol}(), expression)
    _brm_view_source_callables!(callables, expression, _brm_inline_scope(body), bound)
end
_brm_view_walk!(_data, _callables, _seen, _x) = nothing

_brm_view_binding(x, scope, bound) = nothing
_brm_view_binding(x::Symbol, scope, bound) =
    x in bound || !isdefined(scope, x) ? nothing : getfield(scope, x)
_brm_view_binding(x::GlobalRef, scope, bound) =
    isdefined(x.mod, x.name) ? getfield(x.mod, x.name) : nothing
function _brm_view_binding(x::Expr, scope, bound)
    Meta.isexpr(x, :., 2) && last(x.args) isa QuoteNode || return nothing
    mod = _brm_view_binding(first(x.args), scope, bound)
    name = last(x.args).value
    mod isa Module && name isa Symbol && isdefined(mod, name) ? getfield(mod, name) : nothing
end
function _brm_view_source_callables!(callables, x::Expr, scope, bound)
    if Meta.isexpr(x, :call) ||
            (Meta.isexpr(x, :., 2) && Meta.isexpr(last(x.args), :tuple))
        callable = _brm_view_binding(first(x.args), scope, bound)
        statement = callable === (~) && Meta.isexpr(x, :call, 3)
        !statement && callable isa Union{Function,Type} && push!(callables, callable)
    end
    foreach(arg -> _brm_view_source_callables!(callables, arg, scope, bound), x.args)
end
_brm_view_source_callables!(_callables, _x, _scope, _bound) = nothing

const _BRM_VIEW_PLATE_NOTE =
    "Declared once per plate iteration; rank and shape describe one iteration."

# The declarations of one scope: the model's own operations, or a component
# call's instance (its local declarations bound to the call's arguments).
# `data_key(name)` is the bound data column that `name` reads in this scope, or
# `nothing`; `skip` holds names that are not declarations of this scope.
function _brm_view_scope!(ctx, entries, path, plate, data_key, skip)
    for (key, value) in entries
        op = parent(value)
        role = _brm_operation_role(op)
        if role === :observation
            # Use the producer's member projection: a joint response has a
            # synthetic operation key, but its authored members are the data.
            for member in _brm_covariate_members(first(getargs(op)))
                observed = data_key(member)
                isnothing(observed) || push!(ctx.responses, observed)
            end
            continue
        end
        key in skip && continue
        address = (path..., key)
        call = get(ctx.calls, address, nothing)
        isnothing(call) || (role = :component)
        cell = _brm_prune_cell(value)
        if !isnothing(cell)
            role = :cell
            for (_, _, column) in _brm_prune_cell_observations(cell)
                column isa NamedColumn || continue
                observed = data_key(name(column))
                isnothing(observed) || push!(ctx.responses, observed)
            end
        end
        isnothing(data_key(key)) || continue
        _brm_is_statement(op) || continue
        shape = role === :parameter ? _brm_view_prior_shape(last(getargs(op))) :
            (; rank=nothing, dims=nothing)
        push!(ctx.quantities, _brm_view_record(key, role; path=address, plate,
            detail=plate ? _BRM_VIEW_PLATE_NOTE : "", shape...))
        isnothing(call) || _brm_view_call!(ctx, call)
        isnothing(cell) && continue
        # Component calls inside a plate cell: one call (and one set of locals)
        # per iteration, at the cell's authored call site.
        for inner in ctx.records
            length(inner.path) == length(address) + 1 &&
                inner.path[1:end-1] == address || continue
            push!(ctx.quantities, _brm_view_record(last(inner.path), :component;
                path=inner.path, plate=true, detail=_BRM_VIEW_PLATE_NOTE))
            _brm_view_call!(ctx, inner)
        end
    end
end

# A component call's own declarations, under its exact call path. Its inputs
# refer to the caller's values and its generated result locals carry the call's
# returned value, which the caller's row already lists; neither is a local
# declaration. An input bound to caller data resolves to that data column, so a
# local observation of it marks the caller's column as response data.
function _brm_view_call!(ctx, call)
    definition = call.definition
    entries = _brm_operation_entries(call.instance)
    observed = Dict{Symbol,Symbol}()
    for (key, value) in entries
        key in definition.inputs || continue
        backing = _brm_view_input_data(value)
        column = isnothing(backing) ? nothing : get(ctx.columns, backing, nothing)
        isnothing(column) || (observed[key] = column)
    end
    skip = Set{Symbol}((definition.inputs..., _brm_component_results(definition)...))
    _brm_view_scope!(ctx, entries, call.path, call.plate,
        key -> get(observed, key, nothing), skip)
end

# The caller's data column bound to a component input, or `nothing` when the
# input refers to a model value. A local observation of the input replaces its
# operation entry; the observed column is then that statement's left-hand side.
_brm_view_input_data(_) = nothing
function _brm_view_input_data(value::NamedColumn)
    backing = parent(value)
    backing isa DataColumn && return backing
    backing isa ExprColumn{typeof(~)} || return nothing
    lhs = first(getargs(backing))
    lhs isa NamedColumn && parent(lhs) isa DataColumn ? parent(lhs) : nothing
end

Base.@nospecializeinfer function _brm_view_facts(@nospecialize(brmi::BRMI))
    data = Dict{Symbol,Any}()  # data name => its bound DataColumn
    callables = Set{Any}()
    seen = IdDict{Any,Nothing}()
    entries = _brm_operation_entries(brmi)
    records = _brm_component_records(brmi)
    components = map(_brm_public_component_record, records)
    for (_, value) in entries
        _brm_view_walk!(data, callables, seen, value)
    end
    for record in records
        haskey(seen, record.definition) && continue
        seen[record.definition] = nothing
        bound = _brm_cell_bound_names!(Set{Symbol}(brm_columns(record.definition)), record.source)
        _brm_view_source_callables!(callables, record.source, record.scope, bound)
    end
    names = sort!(collect(keys(data)); by=string)
    columns = IdDict{Any,Symbol}()
    foreach(key -> get!(columns, data[key], key), names)
    ctx = (; quantities=NamedTuple[], responses=Set{Symbol}(), records, columns,
        calls=Dict(record.path => record for record in records if !isempty(record.path)))
    _brm_view_scope!(ctx, entries, (), false, key -> haskey(data, key) ? key : nothing, ())
    inputs = [_brm_view_record(key,
        key in ctx.responses ? :response_data : :data;
        _brm_view_shape(parent(data[key]))...,
        detail=parent(data[key]) isa AbstractArray && eltype(parent(data[key])) <: AbstractArray ?
            "Outer container shape; elements have their own axes." : "")
        for key in names]
    (; source=string(brmi), inputs, quantities=ctx.quantities, callables, components,
        documentation=brm_documentation(brmi))
end

# Implemented beside the RK integration, using producer-owned layout metadata.
function _brm_view_rk_parameters end
