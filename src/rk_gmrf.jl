# The caller owns this joint law. RK's extension supplies its visible density
# graph and identity geometry; neither changes the vertex/channel/time law.
struct _BRMGMRFLaw{S,T,L,G}
    spatial_length::S
    temporal_length::T
    correlation_factor::L
    geometry::G
end

# The same field on its read footprint: `geometry` is a `_BRMGMRFFootprint`.
struct _BRMGMRFVecchiaLaw{S,T,L,G}
    spatial_length::S
    temporal_length::T
    correlation_factor::L
    geometry::G
end

# Symbolic sparse lookup only: no sampled values enter this operation. Zero
# denotes an absent entry. Keeping the lookup avoids storing every pair in
# every sparse elimination dot product (which can exceed the factor storage).
function _brm_gmrf_sparse_slot(rows, pointers, column, row)
    first, last = pointers[column], pointers[column + 1] - 1
    while first <= last
        middle = first + (last - first) ÷ 2
        candidate = rows[middle]
        candidate == row && return middle
        if candidate < row
            first = middle + 1
        else
            last = middle - 1
        end
    end
    0
end

# This is a data-only recipe of the emitted model, evaluated from its raw mesh
# and fitted axes. The host formula planner must not supply a numerical factor.
function brm_gmrf_geometry(vertices, cells, times, nchannels)
    mesh = _brm_gmrf_mesh(vertices, cells)
    layout = _BRMGMRFStateLayout(size(vertices, 1), collect(1:nchannels), times)
    _brm_gmrf_graph_geometry(mesh, layout)
end

_brm_gmrf_graph_geometry(mesh, layout) =
    _brm_gmrf_graph_layout(_brm_gmrf_graph_base(mesh), layout)

# A mesh's sparse factor pattern and stiffness, shared by every support on it.
function _brm_gmrf_graph_base(mesh)
    factor = _brm_gmrf_factor_data(mesh)
    columns = Vector{Int}(undef, length(factor.lrow))
    stiffness = Vector{Float64}(undef, length(factor.lrow))
    for j in eachindex(factor.mass), p in factor.lptr[j]:(factor.lptr[j + 1] - 1)
        columns[p] = j
        source = _brm_gmrf_sparse_slot(factor.grow, factor.gptr, j, factor.lrow[p])
        stiffness[p] = source == 0 ? 0.0 : factor.gval[source]
    end
    (; factor..., columns, stiffness)
end

_brm_gmrf_graph_layout(base, layout) = (; base..., times=layout.times,
    nchannels=length(layout.channels), ncoordinates=layout.ncoordinates)

function brm_gmrf_support(vertices, cells, fit_time, fit_channel)
    mesh = _brm_gmrf_mesh(vertices, cells)
    layout = _BRMGMRFStateLayout(size(vertices, 1), _brm_fit_levels(fit_channel),
        _brm_gmrf_times(fit_time))
    (; mesh, layout, geometry=_brm_gmrf_graph_geometry(mesh, layout))
end

# Independent fields of a `@plate for p in groups(g)` cell, one per fitted cell
# label, read from the fit-role (training) columns: one shared mesh and sparse
# pattern, and each label's own fitted times and channel levels, partitioned
# in one pass over the rows. `channel === nothing` is a one-channel field;
# `nchannels` (or `nothing`) is the channel count a correlation factor fixes.
function brm_gmrf_cell_supports(vertices, cells, time, channel, group, nchannels)
    length(time) == length(group) && (channel === nothing || length(channel) == length(group)) ||
        throw(DimensionMismatch("gmrf cell time, channel and grouping columns need one entry per row"))
    mesh = _brm_gmrf_mesh(vertices, cells)
    base = _brm_gmrf_graph_base(mesh)
    position = Dict{eltype(group),Int}()
    rows = Vector{Int}[]
    for (row, label) in enumerate(group)
        k = get!(position, label) do
            push!(rows, Int[])
            length(rows)
        end
        push!(rows[k], row)
    end
    supports = map(rows) do index
        channels = channel === nothing ? [1] : _brm_fit_levels([channel[r] for r in index])
        nchannels === nothing || length(channels) == nchannels || throw(DimensionMismatch(
            "gmrf channel correlation dimension must equal each cell's fitted channel count"))
        layout = _BRMGMRFStateLayout(size(vertices, 1), channels,
            _brm_gmrf_times([time[r] for r in index]))
        (; mesh, layout, geometry=_brm_gmrf_graph_layout(base, layout))
    end
    (; position, supports)
end

brm_gmrf_cell_supports(vertices, cells, time, group) =
    brm_gmrf_cell_supports(vertices, cells, time, nothing, group, nothing)
brm_gmrf_cell_supports(vertices, cells, time, channel, group) =
    brm_gmrf_cell_supports(vertices, cells, time, channel, group, nothing)

# The fitted supports of `labels`, the current plate cells in plate order. A
# label without a fitted field is refused: a cell's field belongs to the fit.
function brm_gmrf_cell_select(fitted, labels)
    map(labels) do label
        k = get(fitted.position, label, 0)
        k == 0 && throw(ArgumentError(
            "gmrf cell `$label` has no fitted field; replay onto the fit's cells or rebuild the model"))
        fitted.supports[k]
    end
end

brm_gmrf_rows(support, east, north, time) =
    brm_gmrf_rows(support, east, north, time, ones(Int, length(time)))

function brm_gmrf_rows(support, east, north, time, channel)
    interpolation = _brm_gmrf_interpolation(support.mesh, east, north)
    indices = _brm_gmrf_state_indices(interpolation.indices, time, channel, support.layout)
    weights = interpolation.weights
    (; index1=indices[:, 1], index2=indices[:, 2], index3=indices[:, 3],
       weight1=weights[:, 1], weight2=weights[:, 2], weight3=weights[:, 3])
end

# A footprint field's fitted support, from the fit-role columns in the graph.
struct _BRMGMRFFootprintSupport{C}
    geometry::_BRMGMRFFootprint
    channels::C
end

function brm_gmrf_footprint_support(east, north, time, channel, neighbors)
    channels = _brm_fit_levels(channel)
    _BRMGMRFFootprintSupport(
        _brm_vecchia_footprint(east, north, time, length(channels); neighbors), channels)
end

brm_gmrf_rows(support::_BRMGMRFFootprintSupport, east, north, time, channel) =
    (; index=_brm_vecchia_field_index(support.geometry, support.channels, east, north, time, channel))

# One field of a cell row read (`brm_gmrf_cell_read`), as a function of its
# own so a broadcast over the reads infers one concrete element type
# (`getproperty.(reads, :index1)` with a runtime Symbol infers a Union, which
# native Enzyme refuses).
brm_gmrf_index1(r) = r.index1
brm_gmrf_index2(r) = r.index2
brm_gmrf_index3(r) = r.index3
brm_gmrf_weight1(r) = r.weight1
brm_gmrf_weight2(r) = r.weight2
brm_gmrf_weight3(r) = r.weight3
const _BRM_GMRF_READ_FIELDS = (; index1=brm_gmrf_index1, index2=brm_gmrf_index2,
    index3=brm_gmrf_index3, weight1=brm_gmrf_weight1, weight2=brm_gmrf_weight2,
    weight3=brm_gmrf_weight3)

# The shared mesh locator of a plate's cell fields, built once from their
# fitted supports (`brm_gmrf_cell_supports`).
brm_gmrf_cell_locator(fitted) = _brm_gmrf_locator(first(fitted.supports).mesh)

# One row's three vertex reads in a cell's support: state indices and
# barycentric weights. A cell broadcasts it over its rows,
# `brm_gmrf_cell_read.(Ref(locator), Ref(support), east, north, time[, channel])`,
# so ReactiveKernels prepares each read at its own rows (a dotted call) while
# the locator and support are read whole.
function brm_gmrf_cell_read(locator, support, east, north, time, channel=1)
    isfinite(east) && isfinite(north) ||
        throw(ArgumentError("gmrf spatial coordinates must be finite"))
    mesh, layout = support.mesh, support.layout
    cell, b = _brm_gmrf_locate(mesh, locator, east, north, Int[], 1)
    t = searchsortedfirst(layout.times, time)
    t <= length(layout.times) && layout.times[t] == time ||
        throw(ArgumentError("gmrf row time $time is outside the fitted state grid"))
    # A typed search: the cell's derivative passes through this read, and
    # Enzyme refuses `findfirst`'s `Union{Nothing,Int}`.
    c = 0
    for (k, level) in enumerate(layout.channels)
        isequal(level, channel) && (c = k; break)
    end
    c == 0 && throw(ArgumentError("gmrf row channel $channel is unknown"))
    offset = layout.nvertices * ((c - 1) + length(layout.channels) * (t - 1))
    (; index1=offset + mesh.cells[cell, 1], index2=offset + mesh.cells[cell, 2],
       index3=offset + mesh.cells[cell, 3], weight1=b[1], weight2=b[2], weight3=b[3])
end

function _rk_plan_gmrf_term!(prepared, target, data, columns, taken; derived)
    state = prepared.state
    sources = collect(prepared.source)
    for source in sources
        columns[source] = data[source]
    end
    id = _rk_mint_smooth_id!(taken, columns,
        string("gmrf_", target, "_", join(sources, "_")))
    if length(sources) == 3
        channel = _rk_mint_generated!(taken, columns, string(id, "_channel"))
        columns[channel] = ones(Int, length(columns[sources[3]]))
        push!(sources, channel)
    end
    if get(state, :multiresolution, nothing) !== nothing
        fit = (; east=_rk_fit_twin!(derived, columns, taken, data, sources[1]),
            north=_rk_fit_twin!(derived, columns, taken, data, sources[2]),
            time=_rk_fit_twin!(derived, columns, taken, data, sources[3]),
            channel=length(prepared.source) == 3 ?
                _rk_fit_input!(derived, columns, taken, sources[4], [1]) :
                _rk_fit_twin!(derived, columns, taken, data, sources[4]))
        # The rank is an input of the program, not a number it spells: one
        # program serves every rank (user principle `0ikeidi`).
        rank = _rk_fit_input!(derived, columns, taken, Symbol(id, :_rank),
            [state.multiresolution])
        return _RKTermSpec(:gmrf, sources,
            (; id, neighbors=nothing, multiresolution=state.multiresolution, fit, rank,
               source=prepared.source, correlation=state.correlation,
               nchannels=length(state.channels), priors=_brm_term_prior_expressions(prepared),
               state_axes=(; times=state.times, channels=state.channels)), id, id)
    end
    if state.neighbors !== nothing
        # One build per plan, bound like the mesh matrices: a footprint costs far
        # more than a mesh, and each prepared query would otherwise rebuild it.
        support = _brm_fitted_value(:gmrf_footprint_support, Symbol(id)) do
            brm_gmrf_footprint_support(_brm_fit_input_source(sources[1], data[sources[1]]),
                _brm_fit_input_source(sources[2], data[sources[2]]),
                _brm_fit_input_source(sources[3], data[sources[3]]),
                length(prepared.source) == 3 ? [1] : _brm_fit_input_source(sources[4], data[sources[4]]),
                state.neighbors)
        end
        fitted = _rk_fit_input!(derived, columns, taken, Symbol(id, :_footprint), [support])
        return _RKTermSpec(:gmrf, sources,
            (; id, neighbors=state.neighbors, multiresolution=nothing, fitted,
               source=prepared.source, correlation=state.correlation,
               nchannels=length(state.channels), priors=_brm_term_prior_expressions(prepared),
               state_axes=(; times=state.times, channels=state.channels)),
            id, id)
    end
    fit_time = _rk_fit_twin!(derived, columns, taken, data, sources[3])
    fit_channel = length(prepared.source) == 3 ?
        _rk_fit_input!(derived, columns, taken, sources[4], [1]) :
        _rk_fit_twin!(derived, columns, taken, data, sources[4])
    vertices = _rk_fit_input!(derived, columns, taken, state.vertices_source, [state.vertices])
    cells = _rk_fit_input!(derived, columns, taken, state.cells_source, [state.cells])
    _RKTermSpec(:gmrf, sources,
        (; id, neighbors=nothing, multiresolution=nothing, vertices, cells, fit_time, fit_channel, source=prepared.source, correlation=state.correlation,
           nchannels=length(state.channels), priors=_brm_term_prior_expressions(prepared),
           state_axes=(; nvertices=state.nvertices, times=state.times, channels=state.channels)),
        id, id)
end

function _rk_ast_gmrf!(definitions, statements, taken, bindings, term)
    options = term.options
    support = _rk_ast_fresh_name(string(options.id, "_support"), taken)
    rows = _rk_ast_fresh_name(string(options.id, "_rows"), taken)
    reader = _rk_value_callee!(bindings, brm_gmrf_rows, taken)
    footprint = options.neighbors !== nothing
    multiresolution = options.multiresolution !== nothing
    law = _rk_value_callee!(bindings, multiresolution ? _BRMGMRFMultiresolutionLaw :
        footprint ? _BRMGMRFVecchiaLaw : _BRMGMRFLaw, taken)
    fit = if multiresolution
        Expr(:call, _rk_value_callee!(bindings, brm_gmrf_multiresolution_support, taken),
            options.fit.east, options.fit.north, options.fit.time, options.fit.channel,
            Expr(:ref, options.rank, 1))
    elseif footprint
        Expr(:ref, options.fitted, 1)
    else
        Expr(:call, _rk_value_callee!(bindings, brm_gmrf_support, taken),
            Expr(:ref, options.vertices, 1), Expr(:ref, options.cells, 1),
            options.fit_time, options.fit_channel)
    end
    push!(statements, Expr(:(=), support, fit))
    push!(statements, Expr(:(=), rows, Expr(:call, reader, support, term.columns...)))
    prior = map(p -> _rk_ast_positive_prior(p, bindings, taken; default=:LogNormal), options.priors)
    block = _rk_block_body(prior...)
    geometry = _rk_block_argument!(block, :geometry, _rk_ast_property(support, :geometry))
    reads = _rk_block_argument!(block, :rows, rows)
    correlation = options.correlation === nothing ?
        _rk_ast_identity_factor(options.nchannels) : _rk_ast_level_factor_value(options.correlation)
    L = _rk_block_argument!(block, :L, correlation)
    multiresolution_functions = multiresolution ?
        (; basis=_rk_value_callee!(bindings, _rk_gmrf_multiresolution_basis(), taken),
           reader=_rk_gmrf_multiresolution_reader!(definitions, taken, bindings)) : nothing
    locals = _rk_ast_gmrf_block!(definitions, taken, law, block, prior, geometry, reads, L;
        footprint, multiresolution_functions)
    push!(statements, Expr(:call, :~, options.id, locals.call))
    (; locals.sigma, locals.spatial_length, locals.temporal_length, locals.field,
       geometry, support, rows, priors=prior)
end

_rk_ast_property(value, name) = Expr(:., value, QuoteNode(name))
_rk_ast_identity_factor(n) = Expr(:call, :hcat, (Expr(:vect, (i == j ? 1.0 : 0.0 for i in 1:n)...)
    for j in 1:n)...)

# One field's block: its hyperparameters under their priors, its joint law on
# `geometry`, and the rows' three-vertex reads scaled by its SD, as the call of
# a shared definition. `block` already holds the arguments `geometry`, `reads`
# and `L` (and any prior inputs the priors read). A block called per plate cell
# declares its field bare, `field ~ law(...)`, its shape the law's sampling
# geometry (RKPPL cell submodels bind bare names only).
function _rk_ast_gmrf_block!(definitions, taken, law, block, prior, geometry, reads, L;
        cell::Bool=false, accessors=nothing, footprint::Bool=false, multiresolution_functions=nothing)
    sigma = _rk_block_local!(block, :sigma)
    spatial_length = _rk_block_local!(block, :spatial_length)
    temporal_length = _rk_block_local!(block, :temporal_length)
    field = _rk_block_local!(block, :field)
    push!(block.statements,
        _rk_ast_hyper_statement(sigma, prior.sigma),
        _rk_ast_hyper_statement(spatial_length, prior.spatial_length),
        _rk_ast_hyper_statement(temporal_length, prior.temporal_length),
        Expr(:call, :~, cell ? field : Expr(:ref, field, Expr(:call, :(:), 1,
            _rk_ast_property(geometry, :ncoordinates))),
            Expr(:call, law, spatial_length, temporal_length, L, geometry)))
    # A cell's reads are one named tuple per row (`brm_gmrf_cell_read`), read
    # through dotted accessors (`accessors`, callee names) so each stays a row
    # form with a concrete element type.
    access(name) = accessors === nothing ? _rk_ast_property(reads, name) :
        Expr(:., getproperty(accessors, name), Expr(:tuple, reads))
    # A footprint field has a state at every read, so a row reads that state.
    read = if multiresolution_functions !== nothing
        weights = _rk_block_local!(block, :weights)
        push!(block.statements, Expr(:(=), weights,
            Expr(:call, multiresolution_functions.basis, spatial_length, geometry)))
        Expr(:call, multiresolution_functions.reader, field, weights, geometry, access(:read))
    elseif footprint
        Expr(:ref, field, access(:index))
    else
        Expr(:call, :.+, (Expr(:call, :.*, access(Symbol(:weight, j)),
            Expr(:ref, field, access(Symbol(:index, j)))) for j in 1:3)...)
    end
    value = Expr(:call, :.*, sigma, read)
    call = _rk_ast_block_call!(definitions, taken, "brm_gmrf", block, value)
    (; call, sigma, spatial_length, temporal_length, field)
end

# ---- Independent fields in `@plate for p in groups(g)` cells ----------------
#
# A cell formula `field ~ 0 + gmrf(...)` is one field per cell label, as on
# SBBRMI (`sb_cell_formulas.jl`). Its support is fitted in the graph from the
# fit-role copies of the mesh, time, channel and grouping columns
# (`brm_gmrf_cell_supports`, one fitted value), and selected for the plate's
# current cells by label (`brm_gmrf_cell_select`). Each cell reads its own
# rows from that support and draws its field, with its own hyperparameters,
# from the same field block as a model-scope `gmrf`.
#
# The prepared cell keeps the formula as the local assignment
# `field = _rk_cell_gmrf(spec, east, north, time[, channel], inputs...)`: its
# arguments are what the cell reads for it, so slicing and the cell's reads
# treat it as any other local. `inputs` are the names the term's priors and a
# sampled correlation factor read, passed to the block as arguments.
mutable struct _RKCellGMRF
    target::Symbol
    id::String
    group::Symbol
    time::Symbol
    channel::Union{Nothing,Symbol}
    vertices::Symbol
    cells::Symbol
    correlation::Any
    priors::Any
    inputs::Vector{Symbol}
    labels::Symbol
    # The fit-role inputs, registered with the plan (`_rk_plan_cell_gmrf!`).
    fit::Any
end

_rk_cell_gmrf(spec, args...) = error("RK backend: internal: a cell gmrf marker was evaluated")
_rk_is_cell_gmrf(statement) = Meta.isexpr(statement, :(=), 2) &&
    Meta.isexpr(last(statement.args), :call) &&
    first(last(statement.args).args) === _rk_cell_gmrf
_rk_cell_gmrf_spec(statement) = last(statement.args).args[2]

# One cell's values of a positional input, read to build the formula's static
# structure (its support is fitted in the graph per cell).
function _rk_kernel_first_cell(input, columns)
    haskey(columns, input.source) || return nothing
    values = columns[input.source]
    input.kind === :gather && return values[first(columns[input.rows])]
    input.kind === :element && return first(values)
    nothing
end

# The cell body with each `gmrf` formula replaced by its marker and its term
# prior statements removed. A body without a formula is returned unchanged.
function _rk_kernel_cell_formulas(@nospecialize(brmi::BRMI), program, kernel, scope,
        raw_body, params, inputs, columns, group_values)
    formula = _brm_cell_formula_program(raw_body, scope; inputs=params)
    formula === nothing && return raw_body
    prefix = "RK backend: kernel `$kernel`"
    group_values === nothing && error(
        "$prefix: a gmrf field in a cell needs the cells' grouping (`@plate for p in groups(g)`)")
    isempty(group_values) && error("$prefix: a gmrf field needs at least one fitted cell")
    locals = Set{Symbol}((formula.targets..., formula.declarations...))
    bindings = Dict{Symbol,Any}(key => column for (key, column) in _brm_operation_entries(brmi)
        if !(key in params) && !(key in locals))
    raw = Dict{Symbol,Any}(program.context.data)
    foreach(key -> delete!(raw, key), locals)
    for (param, input) in zip(params, inputs)
        value = _rk_kernel_first_cell(input, columns)
        value === nothing || (raw[param] = value)
    end
    model = formula.builder(_BRMCellFormulaInputs(_BRMReplayData((;), raw), bindings))
    context = _brm_backend_context(model)
    available = union(Set(keys(program.context.data)), Set(op.name for op in program.operations))
    bound = Set{Symbol}(params)
    foreach(statement -> _rk_source_outputs!(bound, statement), raw_body)
    for statement in raw_body
        isxcall(statement, :~) && statement.args[2] isa Symbol && push!(bound, statement.args[2])
    end
    prior_reads = Set{Symbol}()
    for i in formula.prior_positions
        foreach(value -> _brm_cell_value_refs!(prior_reads, value), raw_body[i].args[3:end])
    end
    taken = union(Set{Symbol}(keys(columns)), available, bound)
    labels = _rk_ast_fresh_name(string(kernel, "_cell_labels"), taken)
    replacements = Dict{Int,Any}()
    for (target, position) in zip(formula.targets, formula.positions)
        _, rhs = getargs(linear_predictor_op(model, target), 2)
        terms = _brm_additive_terms(rhs)
        fields = [term for term in terms if term isa ExprColumn && getf(term) === gmrf]
        length(fields) == 1 && all(term -> term in fields || term isa Integer && term == 0, terms) ||
            error("$prefix: cell field `$target` must be `0 + gmrf(...)` alone; other " *
                "terms in an independent cell formula are not lowered on RK yet")
        term = only(fields)
        arguments = _brm_gmrf_arguments(term)
        arguments.mesh !== nothing || error(
            "$prefix: cell field `$target` needs mesh=; read-support forms are model-scope fields")
        positional = (name.(arguments.axes)..., name(arguments.time),
            (arguments.channel === nothing ? () : (name(arguments.channel),))...)
        sources = map(positional) do param
            k = findfirst(==(param), params)
            k === nothing && error("$prefix: cell field `$target` reads `$param`, which " *
                "is no positional input of the cell; read a row column such as `$param[p]`")
            input = inputs[k]
            input.kind === :gather && haskey(program.context.data, input.source) || error(
                "$prefix: cell field `$target` needs `$param` as a data column read " *
                "through the cell's grouping (`$param[p]` in `@plate for p in groups(g)`)")
            input
        end
        groups = unique(input.group for input in sources)
        length(groups) == 1 || error(
            "$prefix: cell field `$target` reads its rows through different groupings")
        mesh = map(value -> name(value), arguments.mesh)
        all(source -> haskey(program.context.data, source) && !(source in params), mesh) ||
            error("$prefix: cell field `$target` needs its mesh as shared data columns")
        correlation = _brm_gmrf_correlation(arguments.cor)
        priors = (;
            sigma=_brm_term_prior_expression(term, target, context, :term_sd;
                default=ExprColumn(LogNormal, 0.0, 1.0)),
            spatial_length=_brm_term_prior_expression(term, target, context, :term_length_scale;
                component=:space, default=ExprColumn(LogNormal, 0.0, 1.0)),
            temporal_length=_brm_term_prior_expression(term, target, context, :term_length_scale;
                component=:time, default=ExprColumn(LogNormal, 0.0, 1.0)))
        reads = copy(prior_reads)
        correlation !== nothing && correlation.kind === :parameter && push!(reads, correlation.name)
        inputs_read = sort!(collect(intersect(reads, union(bound, available))))
        spec = _RKCellGMRF(target, string("gmrf_", target, "_", join(positional, "_")),
            only(groups), sources[3].source,
            arguments.channel === nothing ? nothing : sources[4].source,
            mesh.vertices, mesh.cells, correlation, priors, inputs_read, labels, nothing)
        replacements[position] = Expr(:(=), target,
            Expr(:call, _rk_cell_gmrf, spec, positional..., inputs_read...))
    end
    columns[labels] = collect(group_values)
    Any[get(replacements, i, statement) for (i, statement) in enumerate(raw_body)
        if !(i in formula.prior_positions)]
end

# Register each cell field's fit-role inputs with the plan: the training mesh
# (fitted like a model-scope field's) and twins of its time, channel and
# grouping columns, so a replay fits the same supports from the training rows.
function _rk_plan_cell_gmrf!(kernels, plan, data, taken)
    derived, columns = plan.regression.derived, plan.columns
    for kernel in kernels, statement in kernel.body
        _rk_is_cell_gmrf(statement) || continue
        spec = _rk_cell_gmrf_spec(statement)
        spec.fit === nothing || continue
        vertices = _brm_fitted(copy, :gmrf_vertices, spec.vertices, data[spec.vertices])
        cells = _brm_fitted(copy, :gmrf_cells, spec.cells, data[spec.cells])
        spec.fit = (;
            vertices=_rk_fit_input!(derived, columns, taken, spec.vertices, [vertices]),
            cells=_rk_fit_input!(derived, columns, taken, spec.cells, [cells]),
            time=_rk_fit_twin!(derived, columns, taken, data, spec.time),
            channel=spec.channel === nothing ? nothing :
                _rk_fit_twin!(derived, columns, taken, data, spec.channel),
            group=_rk_fit_twin!(derived, columns, taken, data, spec.group))
    end
    nothing
end

# The plate's model-scope statements for its cell fields: each field's fitted
# supports and their selection for the current cells. Returns the selection's
# name per field.
function _rk_emit_cell_gmrf_supports!(statements, bindings, taken, body)
    selected = IdDict{Any,Any}()
    for statement in body
        _rk_is_cell_gmrf(statement) || continue
        spec = _rk_cell_gmrf_spec(statement)
        haskey(selected, spec) && continue
        fit = spec.fit
        fit === nothing && error(
            "RK backend: internal: cell field `$(spec.target)` has no fit-role inputs")
        fitted = _rk_ast_fresh_name(string(spec.id, "_supports"), taken)
        cells = _rk_ast_fresh_name(string(spec.id, "_cells"), taken)
        fitter = _rk_value_callee!(bindings, brm_gmrf_cell_supports, taken)
        select = _rk_value_callee!(bindings, brm_gmrf_cell_select, taken)
        nchannels = spec.correlation === nothing ? () : (spec.correlation.K,)
        push!(statements, Expr(:(=), fitted, Expr(:call, fitter,
            Expr(:ref, fit.vertices, 1), Expr(:ref, fit.cells, 1), fit.time,
            (fit.channel === nothing ? () : (fit.channel,))..., fit.group, nchannels...)))
        push!(statements, Expr(:(=), cells, Expr(:call, select, fitted, spec.labels)))
        locator = _rk_ast_fresh_name(string(spec.id, "_locator"), taken)
        push!(statements, Expr(:(=), locator, Expr(:call,
            _rk_value_callee!(bindings, brm_gmrf_cell_locator, taken), fitted)))
        selected[spec] = (; cells, locator)
    end
    selected
end

# A cell's field marker, its reads already the cell's: the cell's rows on its
# selected support, its field drawn from the field block, and the authored
# local bound to the field's row values.
function _rk_expand_cell_gmrf(statement, index, selected, defs, bindings, taken, avoid,
        coordinates, kernel)
    defs === nothing && error(
        "RK backend: internal: kernel `$kernel` emits a gmrf cell field without the program's definitions")
    lhs, marker = statement.args
    spec = marker.args[2]
    nrows = spec.channel === nothing ? 3 : 4
    rows_args = marker.args[3:(2 + nrows)]
    input_values = marker.args[(3 + nrows):end]
    support = Expr(:ref, selected[spec].cells, index)
    rows = _rk_ast_fresh_name(string(spec.id, "_rows"), union(avoid, taken))
    declared = _rk_ast_fresh_name(spec.id, union(avoid, taken))
    push!(avoid, rows, declared)
    push!(taken, declared)
    reader = _rk_value_callee!(bindings, brm_gmrf_cell_read, taken)
    law = _rk_value_callee!(bindings, _BRMGMRFLaw, taken)
    block = _rk_block_body()
    geometry = _rk_block_argument!(block, :geometry, _rk_ast_property(support, :geometry))
    reads = _rk_block_argument!(block, :rows, rows)
    values = Dict{Symbol,Any}(zip(spec.inputs, input_values))
    correlation = spec.correlation === nothing ?
        Expr(:call, _rk_value_callee!(bindings, brm_gmrf_identity_factor, taken),
            _rk_ast_property(support, :geometry)) :
        spec.correlation.kind === :parameter ? values[spec.correlation.name] :
        _rk_ast_level_factor_value(spec.correlation)
    L = _rk_block_argument!(block, :L, correlation)
    renamed = Dict{Symbol,Symbol}(name => _rk_block_argument!(block, name, values[name])
        for name in spec.inputs if spec.correlation === nothing ||
            spec.correlation.kind !== :parameter || name !== spec.correlation.name)
    prior = map(spec.priors) do p
        _rk_rename_symbols(_rk_ast_positive_prior(p, bindings, taken; default=:LogNormal), renamed)
    end
    accessors = map(f -> _rk_value_callee!(bindings, f, taken), _BRM_GMRF_READ_FIELDS)
    locals = _rk_ast_gmrf_block!(defs, taken, law, block, prior, geometry, reads, L;
        cell=true, accessors)
    _rk_coordinate_record!(coordinates, (; kind=:cell_gmrf, declaration=declared,
        name=spec.target, kernel=kernel, id=spec.id, labels=spec.labels, fit=spec.fit,
        cells=selected[spec].cells, correlation=spec.correlation,
        nchannels=spec.correlation === nothing ? nothing : spec.correlation.K, locals))
    entry = Expr(:ref, declared, index)
    Any[Expr(:(=), rows, Expr(:., reader, Expr(:tuple, Expr(:call, :Ref, selected[spec].locator),
            Expr(:call, :Ref, support), rows_args...))),
        Expr(:call, :~, entry, locals.call),
        Expr(:(=), lhs, entry)]
end

brm_gmrf_identity_factor(geometry) = Matrix{Float64}(I, geometry.nchannels, geometry.nchannels)


struct _BRMGMRFMultiresolutionLaw{S,T,L,G}
    spatial_length::S
    temporal_length::T
    correlation_factor::L
    geometry::G
end

struct _BRMGMRFMultiresolutionSupport{C}
    geometry::_BRMGMRFMultiresolution
    channels::C
end
function brm_gmrf_multiresolution_support(east, north, time, channel, rank)
    channels = _brm_fit_levels(channel)
    _BRMGMRFMultiresolutionSupport(
        _brm_multiresolution_support(east, north, time, length(channels); rank), channels)
end
function brm_gmrf_rows(s::_BRMGMRFMultiresolutionSupport, east, north, time, channel)
    reads, _ = _brm_multiresolution_rows(s.geometry, s.channels, east, north, time, channel)
    (; read=reads)
end
# The extension provides the RK-owned derivative rule and visible read graph.
function _rk_gmrf_multiresolution_basis end

# Emit the weighted read as a visible plate and scalar reduction. The two
# ordinary helpers read data only; all parameter gathers stay in the graph.
function _rk_gmrf_multiresolution_reader!(definitions, taken, bindings)
    reader = _rk_ast_fresh_name("brm_gmrf_multiresolution_read", taken)
    coordinate = _rk_value_callee!(bindings, _brm_multiresolution_read_coordinate, taken)
    domain = _rk_value_callee!(bindings, _brm_multiresolution_basis_range, taken)
    push!(definitions, :(ReactiveKernels.@kernel function $reader(field, weights, geometry, reads)
        values = plate(reads) do read
            sum((weights[j] * field[$coordinate(geometry, read, j)]
                for j in $domain(geometry, read)); init=0.0)
        end
        return values
    end))
    reader
end
