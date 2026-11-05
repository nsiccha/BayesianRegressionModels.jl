# Cell-local formulas use ordinary predictor emission. Derived inputs are
# stacked on the plate axis; each invocation retains its own joint prior.
_sb_cell_formula_syntax(x) = x
_sb_cell_formula_syntax(x::Expr) = Expr(x.head, map(_sb_cell_formula_syntax, x.args)...)
_sb_cell_formula_syntax(x::StanBlocks.SlicModel) =
    (; model=_sb_cell_formula_syntax(x.model), x.data, x.mod, x.observations)
_sb_cell_formula_syntax(x::AbstractVector) = map(_sb_cell_formula_syntax, x)
_sb_cell_formula_syntax(x::AbstractDict) =
    Dict(key => _sb_cell_formula_syntax(value) for (key, value) in x)

# Formula declarations own their bindings lexically, even when caller data
# have the same names. Mark only the emitted bare declarations; observations
# and shared references retain the ordinary cell semantics.
function _sb_cell_formula_local(statement)
    lhs = if Meta.isexpr(statement, :(=), 2)
        statement.args[1]
    elseif isxcall(statement, :~)
        statement.args[2]
    else
        return statement
    end
    Meta.isexpr(lhs, :(::)) && (lhs = lhs.args[1])
    lhs isa Symbol ? Expr(:local, statement) : statement
end

function _sb_cell_formula_instance(program, raw, scope; bindings=Dict{Symbol,Any}())
    model = program.builder(_BRMCellFormulaInputs(_BRMReplayData((;), raw), bindings))
    overrides = _sb_prior_overrides(model)
    data = Dict{Symbol,Any}(_SB_PREPROC_KEY => Dict{Symbol,PreprocEntry}(),
        _SB_BINDINGS_KEY => Dict{Symbol,NamedTuple}())
    statements = Dict{Int,Vector{Any}}()
    declaration_constants = Set{Symbol}()
    for (target, position) in zip(program.targets, program.positions)
        op = linear_predictor_op(model, target)
        _, rhs = getargs(op, 2)
        emitted = Any[]
        _sb_linear_predictor!(emitted, data, target, rhs;
            effect_overrides=overrides, mod=scope)
        statements[position] = emitted
    end
    for (target, position) in zip(program.declarations, program.declaration_positions)
        op = _named_op(model.operations[target])
        _, rhs = getargs(op, 2)
        emitted = Any[]
        previous = Set(keys(data))
        _sb_emit_sampled_declaration!(emitted, data, target, getf(rhs), rhs)
        # Vector-prior dimensions are fitted declaration constants.
        for key in setdiff(Set(keys(data)), previous)
            _sb_record_static!(data, key)
            push!(declaration_constants, key)
        end
        statements[position] = emitted
    end
    preproc = pop!(data, _SB_PREPROC_KEY)
    emission = pop!(data, _SB_BINDINGS_KEY)
    predictors = Tuple((; logical=lp.name, emitted=_sb_lp_emitted_name(lp.name, lp.link_lhs_fn),
        link=lp.link_lhs_fn) for lp in linear_predictors(model) if lp.name in program.targets)
    (; data, preproc, statements, bindings=emission, predictors, declaration_constants)
end

function _sb_cell_formulas!(body, data, plate_target, names, params, scope;
        labels, group, distinct)
    program = _brm_cell_formula_program(body, scope; inputs=params)
    program === nothing && return body
    isempty(labels) && throw(ArgumentError("a formula plate needs at least one fitted cell"))
    # Only data-backed positionals participate in preparation. Other inputs
    # remain symbolic references in the native emitted cell.
    inputs = [(param, key) for (param, key) in zip(params, names) if haskey(data, key)]
    locals = Set((program.targets..., program.declarations...))
    bindings = Dict(key => column for (key, column) in
        get(data, _SB_CELL_FORMULA_BINDINGS_KEY, Dict{Symbol,Any}())
        if !(key in params) && !(key in locals))
    cells = map(eachindex(labels)) do i
        raw = Dict{Symbol,Any}(data)
        foreach(key -> delete!(raw, key), locals)
        for (param, key) in inputs
            raw[param] = data[key][i]
        end
        _sb_cell_formula_instance(program, raw, scope; bindings)
    end
    local_keys = sort!(collect(keys(first(cells).data)); by=string)
    all(c -> Set(keys(c.data)) == Set(local_keys), cells) || throw(ArgumentError(
        "cell formulas must emit the same named inputs in each plate cell"))
    syntax = _sb_cell_formula_syntax(first(cells).statements)
    all(c -> _sb_cell_formula_syntax(c.statements) == syntax, cells) || throw(ArgumentError(
        "cell formula declarations must have the same structure in each plate cell"))
    all(c -> c.bindings == first(cells).bindings && c.predictors == first(cells).predictors,
        cells) || throw(ArgumentError("cell formula ownership must agree across plate cells"))
    # Each datum is passed explicitly to the cell, including known parameter
    # values. Ragged dimensions therefore follow each field's fitted support.
    owned = Dict{Symbol,Symbol}()
    shared = Set{Symbol}()
    for key in local_keys
        key in params && throw(ArgumentError("cell formula input `$key` conflicts with a plate input"))
        outer = Symbol(:cell_formula_, plate_target, :_, key)
        haskey(data, outer) && throw(ArgumentError("cell formula reserves data key `$outer`"))
        owned[key] = outer
        if all(c -> key in c.declaration_constants &&
                isequal(c.data[key], first(cells).data[key]), cells)
            # A fixed declaration dimension (e.g. LKJCholesky(2, eta)) is
            # shared fitted data. Slicing it would falsely declare a varying
            # constrained shape; the sampled parameters remain per cell.
            data[outer] = first(cells).data[key]
            push!(shared, key)
        else
            data[outer] = [c.data[key] for c in cells]
            push!(names, outer)
            push!(params, key)
        end
    end
    record_key = Symbol(:__cell_formulas_, plate_target)
    fitted = [(; c.data, c.preproc) for c in cells]
    records = get(data, _SB_PREPROC_KEY, Dict{Symbol,PreprocEntry}())
    records isa _SBPreprocContext && (records = records.recorded)
    input_names = Set(first.(inputs))
    for cell in cells, entry in values(cell.preproc)
        entry.kind === :gmrf || continue
        state = entry.const_.prepared.state
        state.neighbors === nothing && get(state, :multiresolution, nothing) === nothing || throw(ArgumentError(
            "gmrf read-support forms are model-scope fields; independent cell fields keep mesh="))
        captured = (state.vertices_source, state.cells_source,
            (state.correlation !== nothing && state.correlation.kind === :data ?
                (state.correlation.name,) : ())...)
        for source in captured
            # A captured mesh/correlation is fitted support. A sliced one is
            # already owned by its plate input's replay record.
            source in input_names && continue
            haskey(data, source) && !haskey(records, source) && _sb_record_static!(data, source)
        end
    end
    ragged_inputs = Set(source for (_, source) in inputs
        if haskey(records, source) && records[source].kind === :kernel_ragged)
    if group !== nothing
        # A fitted latent field owns a cell axis, even if a replay has no rows
        # in one of its cells. Observation gathers must retain that same axis.
        for key in (ragged_inputs..., Symbol(:kernel_nsub_, plate_target))
            haskey(records, key) || continue
            e = records[key]
            options = e.const_ === nothing ? (;) : e.const_
            _sb_record_preproc!(data, key, PreprocEntry(e.kind,
                merge(options, (; fitted_cell_labels=collect(labels))), e.raw_ref, e.dim_coupled))
        end
    end
    _sb_record_preproc!(data, record_key, PreprocEntry(:cell_formulas,
        (; cells=fitted, labels=collect(labels), inputs, ragged_inputs, owned, shared, group, distinct,
           context=(plate_target,), bindings=first(cells).bindings,
           predictors=first(cells).predictors), group, true))
    replacements = first(cells).statements
    rewritten = Any[]
    for (i, statement) in enumerate(body)
        i in program.prior_positions && continue
        haskey(replacements, i) ?
            append!(rewritten, _sb_cell_formula_local.(replacements[i])) :
            push!(rewritten, statement)
    end
    for key in shared
        rewritten = Any[_sb_subst_sym(statement, key, owned[key]) for statement in rewritten]
    end
    # Formula preparation consumes labels and raw coordinates. Only inputs
    # still read by the emitted cell belong in its Stan signature; the full
    # input/provenance records above remain available for frozen replay.
    reads = _brm_cell_value_refs!(Set{Symbol}(), Expr(:block, rewritten...))
    unused = findall(param -> !(param in reads), params)
    deleteat!(names, unused)
    deleteat!(params, unused)
    rewritten
end

function _sb_prune_cell_formula_inputs!(data, preproc, statements)
    reads = Set{Symbol}()
    foreach(statement -> _sb_read_symbols!(reads, statement), statements)
    for entry in values(preproc)
        entry.kind === :cell_formulas || continue
        for (_, source) in entry.const_.inputs
            source in reads || delete!(data, source)
        end
    end
    nothing
end

# The names an emitted program reads from its data, with plate-cell scoping: a
# do-block parameter and a cell's `local` binding shadow an outer name in the
# whole cell, so a read of that name there is the cell's own value. As in
# `_sb_read_symbols!`, a keyword NAME and the target of `x = …` are not reads,
# and non-expression payloads (a nested `@slic` model) are opaque.
_sb_scoped_reads!(out, x, bound) = out
_sb_scoped_reads!(out, x::Symbol, bound) = (x in bound || push!(out, x); out)
function _sb_scoped_reads!(out, x::Expr, bound)
    if x.head === :-> && length(x.args) == 2
        body = x.args[2]
        cell = union(bound, _sb_plan_param_names(x.args[1]), Set{Symbol}(
            something(_sb_local_name(s)) for s in _sb_plan_statements(body)
            if _sb_local_name(s) !== nothing))
        return _sb_scoped_reads!(out, body, cell)
    end
    args = x.head === :kw ? x.args[2:end] :
        x.head === :(=) && x.args[1] isa Symbol ? x.args[2:end] :
        x.head === :. && length(x.args) == 2 && x.args[2] isa QuoteNode ? x.args[1:1] :
        x.args
    foreach(a -> _sb_scoped_reads!(out, a, bound), args)
    out
end

# An outer data key that a plate cell declares `local` (a formula cell's own
# prior next to caller data of the same name) and that no statement reads
# outside such a cell is not an input of the program. StanBlocks declares no
# data for it, and keeping it would make replay demand the shadowed value.
function _sb_drop_shadowed_data!(data, preproc, statements)
    shadowed = Set{Symbol}()
    visit(x) = nothing
    function visit(x::Expr)
        if x.head === :-> && length(x.args) == 2
            for s in _sb_plan_statements(x.args[2])
                name = _sb_local_name(s)
                name === nothing || push!(shadowed, name)
            end
        end
        foreach(visit, x.args)
    end
    foreach(visit, statements)
    isempty(shadowed) && return nothing
    reads = Set{Symbol}()
    foreach(s -> _sb_scoped_reads!(reads, s, Set{Symbol}()), statements)
    for key in shadowed
        haskey(data, key) && !(key in reads) || continue
        delete!(data, key)
        delete!(preproc, key)
    end
    nothing
end

function _sb_replay_cell_formulas!(data, preproc, handled, key, entry, df, freeze)
    fit = entry.const_
    current_labels = fit.group === nothing ? collect(eachindex(data[last(first(fit.inputs))])) :
        _sb_reprocess_kernel_cells(df, fit.group, (; distinct_labels=fit.distinct))
    labels = freeze ? fit.labels : current_labels
    lookup = Dict(label => i for (i, label) in enumerate(fit.labels))
    all(haskey(lookup, label) for label in current_labels) || throw(ArgumentError(
        "cell formula replay cannot introduce an unfitted cell; rebuild the model"))
    cells = map(eachindex(labels)) do i
        old = fit.cells[lookup[labels[i]]]
        raw = Dict{Symbol,Any}()
        for (param, source) in fit.inputs
            if source in fit.ragged_inputs
                raw[param] = data[source][i]
            else
                row = findfirst(isequal(labels[i]), current_labels)
                row === nothing && throw(ArgumentError(
                    "cell formula replay needs the per-cell input `$source` for fitted cell $(labels[i])"))
                raw[param] = _sb_df_column(df, source)[row]
            end
        end
        new_data, new_preproc, cell_handled = Dict{Symbol,Any}(), Dict{Symbol,PreprocEntry}(), Set{Symbol}()
        for (local_key, local_entry) in old.preproc
            _sb_reprocess_entry!(new_data, new_preproc, cell_handled,
                local_key, local_entry, _BRMReplayData(df, raw), freeze)
        end
        all(haskey(new_data, name) for name in keys(fit.owned)) || throw(ArgumentError(
            "cell formula replay has an input without fitted provenance"))
        (; data=new_data, preproc=new_preproc)
    end
    for (local_key, outer) in fit.owned
        values = [c.data[local_key] for c in cells]
        if local_key in fit.shared
            all(isequal(first(values)), values) || throw(ArgumentError(
                "cell formula replay changed a shared declaration dimension `$local_key`"))
            data[outer] = first(values)
        else
            data[outer] = values
        end
        push!(handled, outer)
    end
    preproc[key] = PreprocEntry(:cell_formulas, merge(fit, (; cells, labels)), entry.raw_ref, true)
    nothing
end
