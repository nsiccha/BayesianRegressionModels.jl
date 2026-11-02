# The caller owns this joint law. RK's extension supplies its visible density
# graph and identity geometry; neither changes the vertex/channel/time law.
struct _BRMGMRFLaw{S,T,L,G}
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

function _brm_gmrf_graph_geometry(mesh, layout)
    factor = _brm_gmrf_factor_data(mesh)
    columns = Vector{Int}(undef, length(factor.lrow))
    stiffness = Vector{Float64}(undef, length(factor.lrow))
    for j in eachindex(factor.mass), p in factor.lptr[j]:(factor.lptr[j + 1] - 1)
        columns[p] = j
        source = _brm_gmrf_sparse_slot(factor.grow, factor.gptr, j, factor.lrow[p])
        stiffness[p] = source == 0 ? 0.0 : factor.gval[source]
    end
    (; factor..., columns, stiffness, times=layout.times,
       nchannels=length(layout.channels), ncoordinates=layout.ncoordinates)
end

function brm_gmrf_support(vertices, cells, fit_time, fit_channel)
    mesh = _brm_gmrf_mesh(vertices, cells)
    layout = _BRMGMRFStateLayout(size(vertices, 1), _brm_fit_levels(fit_channel),
        _brm_gmrf_times(fit_time))
    (; mesh, layout, geometry=_brm_gmrf_graph_geometry(mesh, layout))
end

function brm_gmrf_rows(support, east, north, time, channel)
    interpolation = _brm_gmrf_interpolation(support.mesh, east, north)
    indices = _brm_gmrf_state_indices(interpolation.indices, time, channel, support.layout)
    weights = interpolation.weights
    (; index1=indices[:, 1], index2=indices[:, 2], index3=indices[:, 3],
       weight1=weights[:, 1], weight2=weights[:, 2], weight3=weights[:, 3])
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
    fit_time = _rk_fit_twin!(derived, columns, taken, data, sources[3])
    fit_channel = length(prepared.source) == 3 ?
        _rk_fit_input!(derived, columns, taken, sources[4], [1]) :
        _rk_fit_twin!(derived, columns, taken, data, sources[4])
    vertices = _rk_fit_input!(derived, columns, taken, state.vertices_source, [state.vertices])
    cells = _rk_fit_input!(derived, columns, taken, state.cells_source, [state.cells])
    _RKTermSpec(:gmrf, sources,
        (; id, vertices, cells, fit_time, fit_channel, correlation=state.correlation,
           nchannels=length(state.channels), priors=_brm_term_prior_expressions(prepared),
           state_axes=(; nvertices=state.nvertices, times=state.times, channels=state.channels)),
        id, id)
end

function _rk_ast_gmrf!(definitions, statements, taken, bindings, term)
    options = term.options
    support = _rk_ast_fresh_name(string(options.id, "_support"), taken)
    rows = _rk_ast_fresh_name(string(options.id, "_rows"), taken)
    fitter = _rk_value_callee!(bindings, brm_gmrf_support, taken)
    reader = _rk_value_callee!(bindings, brm_gmrf_rows, taken)
    law = _rk_value_callee!(bindings, _BRMGMRFLaw, taken)
    push!(statements, Expr(:(=), support, Expr(:call, fitter,
        Expr(:ref, options.vertices, 1), Expr(:ref, options.cells, 1),
        options.fit_time, options.fit_channel)))
    push!(statements, Expr(:(=), rows, Expr(:call, reader, support, term.columns...)))
    prior = map(p -> _rk_ast_positive_prior(p, bindings, taken; default=:LogNormal), options.priors)
    block = _rk_block_body(prior...)
    property(value, name) = Expr(:., value, QuoteNode(name))
    geometry = _rk_block_argument!(block, :geometry, property(support, :geometry))
    reads = _rk_block_argument!(block, :rows, rows)
    correlation = options.correlation === nothing ?
        Expr(:call, :hcat, (Expr(:vect, (i == j ? 1.0 : 0.0 for i in 1:options.nchannels)...)
            for j in 1:options.nchannels)...) : _rk_ast_level_factor_value(options.correlation)
    L = _rk_block_argument!(block, :L, correlation)
    sigma = _rk_block_local!(block, :sigma)
    spatial_length = _rk_block_local!(block, :spatial_length)
    temporal_length = _rk_block_local!(block, :temporal_length)
    field = _rk_block_local!(block, :field)
    push!(block.statements,
        _rk_ast_hyper_statement(sigma, prior.sigma),
        _rk_ast_hyper_statement(spatial_length, prior.spatial_length),
        _rk_ast_hyper_statement(temporal_length, prior.temporal_length),
        Expr(:call, :~, Expr(:ref, field, Expr(:call, :(:), 1, property(geometry, :ncoordinates))),
            Expr(:call, law, spatial_length, temporal_length, L, geometry)))
    summands = [Expr(:call, :.*, property(reads, Symbol(:weight, j)),
        Expr(:ref, field, property(reads, Symbol(:index, j)))) for j in 1:3]
    value = Expr(:call, :.*, sigma, Expr(:call, :.+, summands...))
    call = _rk_ast_block_call!(definitions, taken, "brm_gmrf", block, value)
    push!(statements, Expr(:call, :~, options.id, call))
    (; sigma, spatial_length, temporal_length, field, geometry, support, rows)
end
