# The spatial support and state identities are fitted data. The RK planner
# requests metadata only; its emitted graph computes geometry from raw inputs.
_brm_prepares_term(::ExprColumn{typeof(gmrf)}) = true
const _BRM_GMRF_PARAMETER_BINDINGS =
    (; sd=:sigma, spatial_length=:spatial_length, temporal_length=:temporal_length,
       field=:field)
_brm_term_prior_slots(::typeof(gmrf)) = (
    _BRMTermPriorSlot(:sigma, :term_sd, :positive),
    _BRMTermPriorSlot(:spatial_length, :term_length_scale, :positive; component=:space),
    _BRMTermPriorSlot(:temporal_length, :term_length_scale, :positive; component=:time))
_brm_term_slot_ambiguity(::typeof(gmrf)) = "a space-time field has separate spatial and temporal lengths"
_brm_term_prior_expressions(term::_BRMPreparedTerm{typeof(gmrf)}) =
    (; sigma=term.state.sigma_prior, spatial_length=term.state.spatial_prior,
       temporal_length=term.state.temporal_prior)

function _brm_prepared_term_key(term::ExprColumn{typeof(gmrf)})
    args = getargs(term)
    axes = first(args)
    axes isa Tuple && length(axes) == 2 && all(x -> x isa NamedColumn, axes) ||
        throw(ArgumentError("gmrf needs two named spatial columns as `(east, north)`"))
    rest = map(a -> _brm_term_arg_name(a), args[2:end])
    all(!isnothing, rest) || throw(ArgumentError("gmrf time and channel must name data columns"))
    Symbol("gmrf((", join(name.(axes), ","), "),", join(rest, ","), ")")
end

function _brm_gmrf_arguments(term)
    args, kw = getargs(term), getkwargs(term)
    length(args) in (2, 3) || throw(ArgumentError(
        "gmrf takes `(east, north), time` and an optional channel column"))
    axes = first(args)
    axes isa Tuple && length(axes) == 2 || throw(ArgumentError(
        "gmrf needs two spatial columns as `(east, north)`"))
    haskey(kw, :by) && throw(ArgumentError("use @plate for independent gmrf fields"))
    unknown = setdiff(keys(kw), (:mesh, :neighbors, :multiresolution, :spatial, :temporal, :cor))
    isempty(unknown) || throw(ArgumentError("unknown gmrf keywords: $(join(unknown, ", "))"))
    get(kw, :spatial, :matern) === :matern || throw(ArgumentError(
        "gmrf currently supports spatial=:matern (two-dimensional smoothness ν=1)"))
    get(kw, :temporal, :ou) === :ou || throw(ArgumentError("gmrf currently supports temporal=:ou"))
    mesh = get(kw, :mesh, nothing)
    neighbors = get(kw, :neighbors, nothing)
    multiresolution = get(kw, :multiresolution, nothing)
    _brm_check_frozen_setting(:gmrf, :neighbors, neighbors)
    _brm_check_frozen_setting(:gmrf, :multiresolution, multiresolution)
    count(!isnothing, (mesh, neighbors, multiresolution)) == 1 || throw(ArgumentError(
        "gmrf takes exactly one of mesh=(vertices=xy, cells=triangles), neighbors=m, or multiresolution=m"))
    if mesh !== nothing
        mesh isa NamedTuple && Set(keys(mesh)) == Set((:vertices, :cells)) || throw(ArgumentError(
            "gmrf needs mesh=(vertices=xy, cells=triangles)"))
    end
    (; axes, time=args[2], channel=length(args) == 3 ? args[3] : nothing,
       mesh, neighbors=_brm_gmrf_count(:neighbors, neighbors),
       neighbors_source=_brm_gmrf_count_source(neighbors),
       multiresolution=_brm_gmrf_count(:multiresolution, multiresolution),
       multiresolution_source=_brm_gmrf_count_source(multiresolution),
       cor=get(kw, :cor, nothing))
end

# A read-support count (`neighbors=`, `multiresolution=`) is a number the model
# reads, not part of its structure: an integer literal, or a scalar data field or
# `name = constant(expr)` holding one, so one declaration serves every count.
# The count sizes the fitted support only; neither backend's program spells it.
# A name not yet bound to its value (a constant before resolution) stays a name:
# structural readers (labels, axes) need no count, and preparation reads it bound.
_brm_gmrf_count(::Symbol, ::Nothing) = nothing
function _brm_gmrf_count(option::Symbol, value)
    value isa NamedColumn && !(parent(value) isa DataColumn) && return value
    raw = value isa NamedColumn ? parent(parent(value)) : value
    raw isa Integer && !(raw isa Bool) && 1 <= raw <= 100 || throw(ArgumentError(
        "gmrf $option must be an integer from 1 through 100, written as a literal or " *
        "held by a scalar data field or constant; got " *
        (value isa NamedColumn ? "`$(name(value))` = $(repr(raw))" : repr(raw))))
    Int(raw)
end
_brm_gmrf_count_source(value) =
    value isa NamedColumn && parent(value) isa DataColumn ? name(value) : nothing

function _brm_prepare_term(term::ExprColumn{typeof(gmrf)}, target::Symbol, context;
        geometry::Bool=true)
    arguments = _brm_gmrf_arguments(term)
    spatial = map(arg -> _brm_term_data(:gmrf, arg, context), arguments.axes)
    time_source, time = _brm_term_data(:gmrf, arguments.time, context)
    nrows = length(time)
    all(pair -> last(pair) isa AbstractVector{<:Real} && length(last(pair)) == nrows, spatial) ||
        throw(ArgumentError("gmrf spatial columns must be numeric vectors on the time rows"))
    time isa AbstractVector{<:Real} || throw(ArgumentError("gmrf time must be a numeric vector"))
    times = _brm_fitted(_brm_gmrf_times, :gmrf_times, time_source, time)
    channel_source, row_channels, channels = if arguments.channel === nothing
        nothing, ones(Int, nrows), [1]
    else
        source, raw = _brm_term_data(:gmrf, arguments.channel, context)
        raw isa AbstractVector && length(raw) == nrows && !any(ismissing, raw) ||
            throw(ArgumentError("gmrf channel must have one nonmissing label per row"))
        source, raw, _brm_fitted_levels(source, raw)
    end
    correlation = _brm_gmrf_correlation(arguments.cor)
    correlation === nothing || correlation.K == length(channels) || throw(DimensionMismatch(
        "gmrf channel correlation dimension must equal the fitted channel count"))
    source = (first.(spatial)..., time_source,
        (channel_source === nothing ? () : (channel_source,))...)
    priors = (;
        sigma_prior=_brm_term_prior_expression(term, target, context, :term_sd;
            default=ExprColumn(LogNormal, 0.0, 1.0)),
        spatial_prior=_brm_term_prior_expression(term, target, context, :term_length_scale;
            component=:space, default=ExprColumn(LogNormal, 0.0, 1.0)),
        temporal_prior=_brm_term_prior_expression(term, target, context, :term_length_scale;
            component=:time, default=ExprColumn(LogNormal, 0.0, 1.0)))
    correlation_source = correlation === nothing ? () : (correlation.name,)
    for (option, count) in ((:neighbors, arguments.neighbors),
            (:multiresolution, arguments.multiresolution))
        count isa NamedColumn && throw(ArgumentError(
            "gmrf $option reads `$(name(count))`, which holds no data value here; " *
            "pass it as a scalar data field or declare it `$(name(count)) = constant(expr)`"))
    end
    if arguments.multiresolution !== nothing
        return _brm_gmrf_count_sources(_brm_prepare_gmrf_multiresolution(source,
            correlation_source, spatial, time, row_channels, channels, correlation, times,
            priors, target, arguments.multiresolution, geometry), arguments)
    end
    if arguments.neighbors !== nothing
        return _brm_gmrf_count_sources(_brm_prepare_gmrf_footprint(source,
            correlation_source, spatial, time, row_channels, channels, correlation, times,
            priors, target, arguments.neighbors, geometry), arguments)
    end
    vertices_source, vertices = _brm_term_data(:gmrf, arguments.mesh.vertices, context)
    cells_source, cells = _brm_term_data(:gmrf, arguments.mesh.cells, context)
    vertices = _brm_fitted(copy, :gmrf_vertices, vertices_source, vertices)
    cells = _brm_fitted(copy, :gmrf_cells, cells_source, cells)
    vertices isa AbstractMatrix{<:Real} && cells isa AbstractMatrix{<:Integer} ||
        throw(ArgumentError("gmrf mesh needs numeric vertex and integer cell matrices"))
    size(vertices, 2) == 2 && size(cells, 2) == 3 || throw(ArgumentError(
        "gmrf mesh needs two vertex coordinates and three vertex indices per cell"))
    dependencies = (source..., vertices_source, cells_source, correlation_source...)
    layout = _BRMGMRFStateLayout(size(vertices, 1), channels, times)
    state = (; target, times=layout.times, channels=layout.channels, layout, correlation, vertices_source, cells_source,
        vertices, cells, nvertices=size(vertices, 1), neighbors=nothing, multiresolution=nothing, priors...)
    prepared = _BRMPreparedTerm(gmrf, source, state, dependencies)
    geometry || return prepared
    mesh = _brm_gmrf_mesh(vertices, cells)
    interpolation = _brm_gmrf_interpolation(mesh, last.(spatial)...)
    indices = _brm_gmrf_state_indices(interpolation.indices, time, row_channels, layout)
    _BRMPreparedTerm(gmrf, source,
        merge(state, (; mesh, indices, weights=interpolation.weights)), dependencies)
end

# The data field that holds a read-support count, when one does: a fresh replay
# reads the count from its own data, and the term reads that field.
function _brm_gmrf_count_sources(prepared, arguments)
    sources = (; neighbors_source=arguments.neighbors_source,
        multiresolution_source=arguments.multiresolution_source)
    read = Tuple(s for s in values(sources) if s !== nothing)
    _BRMPreparedTerm(gmrf, prepared.source, merge(prepared.state, sources),
        (prepared.dependencies..., read...))
end

# A footprint field has no mesh: its support is the training rows' read pairs,
# and its state follows from them and the neighbour count alone.
function _brm_prepare_gmrf_footprint(source, correlation_source, spatial, time,
        row_channels, channels, correlation, times, priors, target, neighbors, geometry)
    state = (; target, times, channels=copy(channels), correlation, neighbors, multiresolution=nothing, priors...)
    dependencies = (source..., correlation_source...)
    prepared = _BRMPreparedTerm(gmrf, source, state, dependencies)
    geometry || return prepared
    footprint = _brm_fitted_value(:gmrf_footprint, Symbol(target, :_, join(source, :_))) do
        _brm_vecchia_footprint(_brm_fit_input_source(source[1], last(spatial[1])),
            _brm_fit_input_source(source[2], last(spatial[2])),
            _brm_fit_input_source(source[3], time), length(channels); neighbors)
    end
    index = _brm_vecchia_field_index(footprint, channels, last(spatial[1]), last(spatial[2]),
        time, row_channels)
    _BRMPreparedTerm(gmrf, source, merge(state, (; footprint, index)), dependencies)
end

function _brm_gmrf_correlation(value)
    what = "gmrf(...; cor=...)"
    if value isa NamedColumn && parent(value) isa DataColumn
        factor = _brm_fitted(:gmrf_correlation, name(value), parent(parent(value))) do raw
            _brm_correlation_factor(_brm_correlation_matrix(raw; what); what)
        end
        return _BRMLevelCorrelation(name(value), :data, size(factor, 1), factor)
    end
    _brm_level_correlation(value; what)
end

function _brm_replay_gmrf(training, context)
    state = training.state
    x, y, time = map(s -> context.data[s], training.source[1:3])
    row_channels = length(training.source) == 3 ? ones(Int, length(time)) :
        context.data[training.source[4]]
    if get(state, :multiresolution, nothing) !== nothing
        reads, index = _brm_multiresolution_rows(state.support, state.channels, x, y, time, row_channels)
        return _BRMPreparedTerm(gmrf, training.source, merge(state, (; reads, index)), training.dependencies)
    end
    if state.neighbors !== nothing
        index = _brm_vecchia_field_index(state.footprint, state.channels, x, y, time, row_channels)
        return _BRMPreparedTerm(gmrf, training.source, merge(state, (; index)), training.dependencies)
    end
    interpolation = _brm_gmrf_interpolation(state.mesh, x, y)
    indices = _brm_gmrf_state_indices(interpolation.indices, time, row_channels, state.layout)
    _BRMPreparedTerm(gmrf, training.source,
        merge(state, (; indices, weights=interpolation.weights)), training.dependencies)
end
function _brm_replay_term(::typeof(gmrf), training, fresh::ExprColumn, context)
    args = _brm_gmrf_arguments(fresh)
    source = (name.(args.axes)..., name(args.time),
        (args.channel === nothing ? () : (name(args.channel),))...)
    source == training.source || throw(ArgumentError("gmrf fitted columns changed"))
    _brm_replay_gmrf(training, context)
end
_brm_replay_term(::typeof(gmrf), training, fresh::_BRMPreparedTerm, context) =
    _brm_replay_gmrf(training, context)

# Shared spatial coefficients lifted to exact OU paths on their readers' times.
function _brm_prepare_gmrf_multiresolution(source, correlation_source, spatial, time,
        row_channels, channels, correlation, times, priors, target, multiresolution, geometry)
    state = (; target, times, channels=copy(channels), correlation, neighbors=nothing,
        multiresolution, priors...)
    dependencies = (source..., correlation_source...)
    prepared = _BRMPreparedTerm(gmrf, source, state, dependencies)
    geometry || return prepared
    support = _brm_fitted_value(:gmrf_multiresolution, Symbol(target, :_, join(source, :_))) do
        _brm_multiresolution_support(_brm_fit_input_source(source[1], last(spatial[1])),
            _brm_fit_input_source(source[2], last(spatial[2])),
            _brm_fit_input_source(source[3], time), length(channels); rank=multiresolution)
    end
    reads, index = _brm_multiresolution_rows(support, channels, last(spatial[1]), last(spatial[2]),
        time, row_channels)
    _BRMPreparedTerm(gmrf, source, merge(state, (; support, reads, index)), dependencies)
end
