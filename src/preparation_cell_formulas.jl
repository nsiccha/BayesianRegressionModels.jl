# Scoped formula source is built through the same interpreter as model-scope
# formulas. This creates no Julia methods and keeps term prior addresses intact.
# Emitters retain ordinary cell statements and replace each formula at its
# original position with the corresponding native term declaration/read.
struct _BRMCellFormulaProgram{B}
    builder::B
    targets::Tuple{Vararg{Symbol}}
    positions::Tuple{Vararg{Int}}
    prior_positions::Tuple{Vararg{Int}}
    declarations::Tuple{Vararg{Symbol}}
    declaration_positions::Tuple{Vararg{Int}}
end

# A lexical reference keeps its declared column node. Treating it as raw data
# would erase a sampled prior (notably the dimension of an LKJ factor).
struct _BRMCellFormulaInputs{D,B}
    raw::D
    bindings::B
end
struct _BRMCellFormulaColumns{D,B}
    columns::D
    bindings::B
end
function Base.hasproperty(d::_BRMCellFormulaColumns, key::Symbol)
    haskey(getfield(d, :bindings), key) || hasproperty(getfield(d, :columns), key)
end
function Base.getproperty(d::_BRMCellFormulaColumns, key::Symbol)
    bindings = getfield(d, :bindings)
    haskey(bindings, key) ? bindings[key] : getproperty(getfield(d, :columns), key)
end
data(d::_BRMCellFormulaInputs) = _BRMCellFormulaColumns(data(d.raw), d.bindings)
maybedata(d::_BRMCellFormulaInputs) = _BRMCellFormulaColumns(maybedata(d.raw), d.bindings)
function capturedata(d::_BRMCellFormulaInputs, names::Tuple)
    columns = data(d)
    (; (key => getproperty(columns, key) for key in names if hasproperty(columns, key))...)
end

_brm_cell_formula_marker(_callable) = false
_brm_cell_formula_marker(::typeof(gmrf)) = true
_brm_cell_has_formula(_value, _scope) = false
function _brm_cell_has_formula(value::Expr, scope)
    Meta.isexpr(value, :call) &&
        _brm_cell_formula_marker(_sb_cell_binding(scope, first(value.args))) && return true
    any(arg -> _brm_cell_has_formula(arg, scope), value.args)
end

function _brm_cell_formula_program(statements, scope::Module; inputs=())
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
    # Include local priors read by a formula or its hyperpriors, transitively.
    # Observation positionals and unrelated statements retain ordinary cell
    # lowering. A shared prior is supplied as a binding, never redeclared here.
    candidates = Dict(stmt.args[2] => i for (i, stmt) in enumerate(statements)
        if isxcall(stmt, :~) && stmt.args[2] isa Symbol &&
           !(stmt.args[2] in targets) && !(stmt.args[2] in inputs))
    declarations = Symbol[]
    declaration_positions = Int[]
    refs = Set{Symbol}()
    foreach(i -> _brm_cell_value_refs!(refs, statements[i]), selected)
    while true
        needed = sort!([key for key in keys(candidates) if key in refs && !(key in declarations)];
            by=key -> candidates[key])
        isempty(needed) && break
        for key in needed
            push!(declarations, key)
            push!(declaration_positions, candidates[key])
            _brm_cell_value_refs!(refs, statements[candidates[key]])
        end
    end
    append!(selected, declaration_positions)
    sort!(selected)
    source = Expr(:block, (deepcopy(statements[i]) for i in selected)...)
    _BRMCellFormulaProgram(brm(source; mod=scope), Tuple(targets), Tuple(positions), Tuple(priors),
        Tuple(declarations), Tuple(declaration_positions))
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
