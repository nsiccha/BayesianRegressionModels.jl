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

_brm_view_record(name, role; rank=nothing, dims=nothing, detail="") =
    (; name, role, rank, dims, detail)

function _brm_view_walk!(data, callables, seen, x::NamedColumn)
    haskey(seen, x) && return
    seen[x] = nothing
    p = parent(x)
    p isa DataColumn && (data[name(x)] = parent(p))
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

Base.@nospecializeinfer function _brm_view_facts(@nospecialize(brmi::BRMI))
    data = Dict{Symbol,Any}()
    callables = Set{Any}()
    seen = IdDict{Any,Nothing}()
    entries = _brm_operation_entries(brmi)
    components = brm_components(brmi)
    component_targets = Set(only(record.path) for record in components if length(record.path) == 1)
    for (_, value) in entries
        _brm_view_walk!(data, callables, seen, value)
    end
    for record in components
        haskey(seen, record.definition) && continue
        seen[record.definition] = nothing
        bound = _brm_cell_bound_names!(Set{Symbol}(brm_columns(record.definition)), record.source)
        _brm_view_source_callables!(callables, record.source, record.scope, bound)
    end
    responses = Set{Symbol}()
    quantities = NamedTuple[]
    for (key, value) in entries
        op = parent(value)
        role = _brm_operation_role(op)
        if role === :observation
            # Use the producer's member projection: a joint response has a
            # synthetic operation key, but its authored members are the data.
            union!(responses, _brm_covariate_members(first(getargs(op))))
            continue
        end
        key in component_targets && (role = :component)
        cell = _brm_prune_cell(value)
        if !isnothing(cell)
            role = :cell
            for (_, _, column) in _brm_prune_cell_observations(cell)
                column isa NamedColumn && haskey(data, name(column)) &&
                    push!(responses, name(column))
            end
        end
        haskey(data, key) && continue
        _brm_is_statement(op) || continue
        shape = role === :parameter ? _brm_view_prior_shape(last(getargs(op))) :
            (; rank=nothing, dims=nothing)
        push!(quantities, _brm_view_record(key, role; shape...))
    end
    inputs = [_brm_view_record(key,
        key in responses ? :response_data : :data;
        _brm_view_shape(data[key])...,
        detail=data[key] isa AbstractArray && eltype(data[key]) <: AbstractArray ?
            "Outer container shape; elements have their own axes." : "")
        for key in sort!(collect(keys(data)); by=string)]
    (; source=string(brmi), inputs, quantities, callables, components,
        documentation=brm_documentation(brmi))
end

# Implemented beside the RK integration, using producer-owned layout metadata.
function _brm_view_rk_parameters end
