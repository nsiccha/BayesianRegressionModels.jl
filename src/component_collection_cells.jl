# A formula inside a repeated component is prepared on each record's actual
# inputs. Its native submodel body is shared by the plate. Prepared arrays
# (such as HSGP basis matrices) have one entry per record.
function _brm_collection_cell_calls(component, call, scope, columns, count)
    symbolic = _brm_component_cell_ir(component, call, scope)
    isempty(linear_predictors(_brm_component_instance(component, symbolic))) && return nothing
    arguments = _brm_component_arguments(component, symbolic)
    map(1:count) do i
        values = map(arguments) do arg
            if arg isa NamedColumn && haskey(columns, name(arg))
                return NamedColumn(name(arg), DataColumn(columns[name(arg)][i]))
            end
            arg
        end
        ExprColumn(component, values...)
    end
end

struct _BRMCollectionCellReplay
    fits::Vector{SBBRMI}
    inputs::Dict{Symbol,Symbol}
    prepared::Dict{Symbol,Symbol}
end
Base.deepcopy_internal(r::_BRMCollectionCellReplay, ::IdDict) = r

function _sb_collection_cell!(body, data, axis, names, params, scope)
    i = findfirst(st -> isxcall(st, :~) && st.args[2] === last(axis.path), body)
    isnothing(i) && return body
    lhs, call = body[i].args[2:end]
    component = _brm_component_cell_head(call, scope)
    inputs = Dict(param => source for (param, source) in zip(params, names) if haskey(data, source))
    columns = Dict(param => data[source] for (param, source) in inputs)
    calls = _brm_collection_cell_calls(component, call, scope, columns, length(axis))
    isnothing(calls) && return body
    fits = SBBRMI[SBBRMI(_brm_component_instance(component, rhs); mod=component.scope,
        total_groups=(), _component_output=_brm_component_results(component)) for rhs in calls]
    model = first(fits).model
    syntax = _sb_cell_formula_syntax(model.model)
    all(fit -> _sb_cell_formula_syntax(fit.model.model) == syntax, fits) ||
        throw(ArgumentError("component `$(component.name)` produced different declarations within a collection variant"))
    reads = _rk_source_symbols!(Set{Symbol}(), model.model)
    args = _brm_component_arguments(component, _brm_component_cell_ir(component, call, scope))
    arguments = Any[Expr(:kw, formal, _sb_scalar_expr(arg, data))
        for (formal, arg) in zip(component.inputs, args) if formal in reads]
    prepared = Dict{Symbol,Symbol}()
    taken = union(Set{Symbol}(keys(data)), Set(params), Set(names))
    for inner in sort!(collect(intersect(keys(first(fits).data), reads)))
        inner in component.inputs && continue
        outer = _rk_ast_fresh_name(string(lhs, "_", inner), taken)
        data[outer] = [fit.data[inner] for fit in fits]
        push!(names, outer)
        push!(params, outer)
        push!(arguments, Expr(:kw, inner, outer))
        prepared[inner] = outer
    end
    formal_inputs = Dict(formal => inputs[name(arg)] for (formal, arg) in zip(component.inputs, args)
        if arg isa NamedColumn && haskey(inputs, name(arg)))
    key = _rk_ast_fresh_name(string(lhs, "_cell_fits"), taken)
    _sb_record_preproc!(data, key, PreprocEntry(:collection_cells,
        _BRMCollectionCellReplay(fits, formal_inputs, prepared), first(axis.path), true))
    submodel = StanBlocks.SlicModel(Expr(:block, model.model.args..., _sb_component_return(component)),
        Dict{Symbol,Any}(), component.scope, model.observations)
    rewritten = copy(body)
    rewritten[i] = Expr(:call, :~, lhs, Expr(:call, submodel, Expr(:parameters, arguments...)))
    reads = _brm_cell_value_refs!(Set{Symbol}(), Expr(:block, rewritten...))
    unused = findall(param -> !(param in reads), params)
    for j in unused
        # Record fields belong only to this generated plate. For example,
        # subject labels are already represented by fitted integer indices.
        haskey(inputs, params[j]) && delete!(data, names[j])
    end
    deleteat!(params, unused)
    deleteat!(names, unused)
    rewritten
end

function _sb_replay_collection_cells!(data, preproc, handled, key, entry, df, freeze)
    record = entry.const_
    fits = SBBRMI[reprocess(fit,
        (; (formal => _brm_df_column(df, outer)[i] for (formal, outer) in record.inputs)...);
        freeze_constants=freeze) for (i, fit) in enumerate(record.fits)]
    for (inner, outer) in record.prepared
        data[outer] = [fit.data[inner] for fit in fits]
        push!(handled, outer)
    end
    preproc[key] = PreprocEntry(:collection_cells,
        _BRMCollectionCellReplay(fits, record.inputs, record.prepared), entry.raw_ref, true)
end
