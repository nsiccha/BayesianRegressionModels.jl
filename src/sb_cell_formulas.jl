# Cell-local formulas use ordinary predictor emission. Derived inputs are
# stacked on the plate axis; each invocation retains its own joint prior.
_sb_cell_formula_syntax(x) = x
_sb_cell_formula_syntax(x::Expr) = Expr(x.head, map(_sb_cell_formula_syntax, x.args)...)
_sb_cell_formula_syntax(x::StanBlocks.SlicModel) =
    (; model=_sb_cell_formula_syntax(x.model), x.data, x.mod, x.observations)
_sb_cell_formula_syntax(x::AbstractVector) = map(_sb_cell_formula_syntax, x)
_sb_cell_formula_syntax(x::AbstractDict) =
    Dict(key => _sb_cell_formula_syntax(value) for (key, value) in x)

function _sb_cell_formula_instance(program, raw, scope; bindings=Dict{Symbol,Any}())
    model = program.builder(_BRMCellFormulaInputs(_BRMReplayData((;), raw), bindings))
    overrides = _sb_prior_overrides(model)
    data = Dict{Symbol,Any}(_SB_PREPROC_KEY => Dict{Symbol,PreprocEntry}())
    statements = Dict{Int,Vector{Any}}()
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
        end
        statements[position] = emitted
    end
    preproc = pop!(data, _SB_PREPROC_KEY)
    (; data, preproc, statements)
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
    # Each datum is passed explicitly to the cell, including known parameter
    # values. Ragged dimensions therefore follow each field's fitted support.
    owned = Dict{Symbol,Symbol}()
    for key in local_keys
        key in params && throw(ArgumentError("cell formula input `$key` conflicts with a plate input"))
        outer = Symbol(:cell_formula_, plate_target, :_, key)
        haskey(data, outer) && throw(ArgumentError("cell formula reserves data key `$outer`"))
        data[outer] = [c.data[key] for c in cells]
        owned[key] = outer
        push!(names, outer)
        push!(params, key)
    end
    record_key = Symbol(:__cell_formulas_, plate_target)
    fitted = [(; c.data, c.preproc) for c in cells]
    records = get(data, _SB_PREPROC_KEY, Dict{Symbol,PreprocEntry}())
    records isa _SBPreprocContext && (records = records.recorded)
    input_names = Set(first.(inputs))
    for cell in cells, entry in values(cell.preproc)
        entry.kind === :gmrf || continue
        state = entry.const_.prepared.state
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
        (; cells=fitted, labels=collect(labels), inputs, ragged_inputs, owned, group, distinct), group, true))
    replacements = first(cells).statements
    rewritten = Any[]
    for (i, statement) in enumerate(body)
        i in program.prior_positions && continue
        haskey(replacements, i) ? append!(rewritten, replacements[i]) : push!(rewritten, statement)
    end
    rewritten
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
        data[outer] = [c.data[local_key] for c in cells]
        push!(handled, outer)
    end
    preproc[key] = PreprocEntry(:cell_formulas, merge(fit, (; cells, labels)), entry.raw_ref, true)
    nothing
end
