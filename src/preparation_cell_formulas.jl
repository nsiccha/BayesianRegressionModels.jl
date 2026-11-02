# Scoped formula source is built through the same interpreter as model-scope
# formulas. This creates no Julia methods and keeps term prior addresses intact.
# Emitters retain ordinary cell statements and replace each formula at its
# original position with the corresponding native term declaration/read.
struct _BRMCellFormulaProgram{B}
    builder::B
    targets::Tuple{Vararg{Symbol}}
    positions::Tuple{Vararg{Int}}
    prior_positions::Tuple{Vararg{Int}}
end

_brm_cell_formula_marker(_callable) = false
_brm_cell_formula_marker(::typeof(gmrf)) = true
_brm_cell_has_formula(_value, _scope) = false
function _brm_cell_has_formula(value::Expr, scope)
    Meta.isexpr(value, :call) &&
        _brm_cell_formula_marker(_sb_cell_binding(scope, first(value.args))) && return true
    any(arg -> _brm_cell_has_formula(arg, scope), value.args)
end

function _brm_cell_formula_program(statements, scope::Module)
    positions = Int[]
    targets = Symbol[]
    priors = Int[]
    for (i, stmt) in enumerate(statements)
        _is_prior_address_statement(stmt) && (push!(priors, i); continue)
        isxcall(stmt, :~) && _brm_cell_has_formula(stmt.args[3], scope) || continue
        target = stmt.args[2]
        target isa Symbol || throw(ArgumentError(
            "a plate formula declares a bare cell-local name before its rows are read"))
        target in targets && throw(ArgumentError("duplicate plate formula declaration `$target`"))
        push!(positions, i)
        push!(targets, target)
    end
    isempty(positions) && return nothing
    selected = sort!([positions; priors])
    source = Expr(:block, (deepcopy(statements[i]) for i in selected)...)
    _BRMCellFormulaProgram(brm(source; mod=scope), Tuple(targets), Tuple(positions), Tuple(priors))
end

function _brm_prepare_cell_formulas(program::_BRMCellFormulaProgram, data; geometry::Bool=true)
    model = program.builder(data)
    context = _brm_backend_context(model)
    terms = map(program.targets) do target
        addresses = _brm_term_address_map(model, target)
        Tuple(_brm_prepare_term(term, target, context; geometry)
            for candidates in values(addresses) for term in candidates)
    end
    (; model, context, terms=NamedTuple{program.targets}(terms))
end
