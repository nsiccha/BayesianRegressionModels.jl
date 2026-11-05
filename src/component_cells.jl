# A cell has symbolic inputs at preparation time. Its scalar component uses
# the same compiler and native local scope as a model-level component; the
# plate owns replication of the resulting submodel call.
function _brm_component_cell_head(call, scope)
    Meta.isexpr(call, :call) || return nothing
    callee = _sb_cell_binding(scope, first(call.args))
    callee isa BRMComponent ? callee : nothing
end

function _brm_component_cell_ir(component, call, scope)
    locals = _BRMScope(scope)
    refs = _brm_cell_value_refs!(Set{Symbol}(), call)
    for name in refs
        locals.vars[name] = NamedColumn(name, _BRMComponentInput(name))
    end
    _brm_interpret(_brm_expand(_x(call), scope), locals)
end

function _sb_component_cell_call(component, call, scope)
    rhs = _brm_component_cell_ir(component, call, scope)
    instance = _brm_component_instance(component, rhs)
    compiled = SBBRMI(instance; mod=component.scope, total_groups=(),
        _component_output=component.result)
    data = Dict(k => v for (k, v) in compiled.data if !(k in component.inputs))
    body = Expr(:block, compiled.model.model.args..., Expr(:return, component.result))
    model = StanBlocks.SlicModel(body, data, component.scope, compiled.model.observations)
    inputs = Expr(:parameters, (Expr(:kw, input, _sb_scalar_expr(value, data))
        for (input, value) in zip(component.inputs, _brm_component_arguments(component, rhs)))...)
    Expr(:call, model, inputs)
end

_rk_cell_component(call) = error("RK backend: internal: a component marker was evaluated")
_rk_is_cell_component(statement) = Meta.isexpr(statement, :(=), 2) &&
    Meta.isexpr(last(statement.args), :call) &&
    first(last(statement.args).args) === _rk_cell_component

function _rk_component_cell_statement(statement, kernel, definitions, bindings, taken)
    lhs, marker = statement.args
    call = marker.args[2]
    component = _brm_component_cell_head(call, kernel.scope)
    rhs = _brm_component_cell_ir(component, call, kernel.scope)
    defs, statements = Expr[], Expr[]
    columns, bound = Dict{Symbol,Any}(), Pair{Symbol,Any}[]
    _rk_submodel_rhs!(defs, statements, columns, bound, lhs, component, rhs)
    renames = Dict{Symbol,Symbol}()
    for name in (_rk_source_definition(def).name for def in defs)
        renames[name] = _rk_ast_fresh_name(string(kernel.name, "_", name), taken)
    end
    for (name, value) in bound
        outer = _rk_ast_fresh_name(string(kernel.name, "_", name), taken)
        renames[name] = outer
        push!(bindings, outer => value)
    end
    for (name, value) in columns
        outer = _rk_ast_fresh_name(string(kernel.name, "_", name), taken)
        renames[name] = outer
        kernel.columns[outer] = value
    end
    append!(definitions, (_rk_rename_symbols(def, renames) for def in defs))
    emitted = _rk_rename_symbols(only(statements), renames)
    Expr(:(=), lhs, Expr(:call, _rk_cell_component, emitted.args[3]))
end
