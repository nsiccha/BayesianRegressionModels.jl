# A fitted read-pair field at new points, draw by draw: the public face of
# `_brm_vecchia_predict!` (src/gmrf_vecchia_predict.jl).

# Implemented by the ReactiveKernels extension: named constrained values of a draw.
function _rk_constrained_draw end

"""
    brm_gmrf_predict(rk::RKBRMI, draws, field; east, north, time, channel=nothing,
        neighbors=nothing, rng=nothing) -> Matrix

The read-pair field `field` (`field ~ 0 + gmrf(...; neighbors=m)`, named by its predictor)
at points the fit need not have read, one row per point and one column per column of
`draws`, on the scale the model reads it: the term's SD times the standardized field.
`draws` are `rk`'s packed coordinates, as `brm_svi_fit` and the samplers return them.

A point follows the term's covariance given its `neighbors` nearest read positions at the
read times around it (default: the term's own count). With `rng` each entry is drawn from
that conditional; without it, each is the conditional mean at that draw. A point on a
read state returns that state.

Marginals are the law's, but the points are conditionally independent of each other given
the states, so a joint quantity over several new points, such as an areal total, is not.
The mesh form predicts through [`reprocess`](@ref) and has no such restriction.
"""
function brm_gmrf_predict(rk::RKBRMI, draws::AbstractMatrix, field::Symbol; east, north, time,
        channel=nothing, neighbors=nothing, rng=nothing)
    names = _rk_layout_coordinate_names(rk)
    U = _brm_cut_draws(draws, names; prefix="brm_gmrf_predict")
    record = _brm_gmrf_footprint_record(rk, field)
    support = only(rk.plan.columns[record.fitted])
    f = support.geometry
    channels = _brm_gmrf_predict_channels(support, channel, length(east))
    predictor = _brm_vecchia_predictor(f, east, north, time; neighbors=something(neighbors, f.neighbors))
    out = Matrix{Float64}(undef, length(east), size(U, 2))
    for j in axes(U, 2)
        law = _brm_gmrf_draw_law(rk, record, f, U[:, j])
        column = view(out, :, j)
        _brm_vecchia_predict!(column, predictor, law.field, law.spatial_length, law.temporal_length,
            law.L, f; channels, rng)
        column .*= law.sigma
    end
    out
end

function _brm_gmrf_footprint_record(rk::RKBRMI, field::Symbol)
    records = [r for r in _rk_coordinate_records(rk.plan)
        if r.kind in (:gmrf, :gmrf_footprint, :gmrf_multiresolution) && r.predictor === field]
    isempty(records) && throw(ArgumentError(
        "brm_gmrf_predict: the model has no model-scope gmrf term on `$field`"))
    record = only(records)
    record.kind === :gmrf_footprint || throw(ArgumentError(
        "brm_gmrf_predict: `$field` is a `$(record.kind)` term. This function is for " *
        "`gmrf(...; neighbors=m)`; the mesh form predicts through `reprocess`"))
    record
end

function _brm_gmrf_predict_channels(support, channel, n)
    channel === nothing && return support.geometry.nchannels == 1 ? nothing : throw(ArgumentError(
        "brm_gmrf_predict: this field has $(support.geometry.nchannels) channels; pass `channel`"))
    length(channel) == n || throw(DimensionMismatch(
        "brm_gmrf_predict: `channel` needs one entry per point"))
    place = Dict(c => i for (i, c) in enumerate(support.channels))
    map(c -> get(place, c) do
        throw(ArgumentError("brm_gmrf_predict: unknown channel $(repr(c))"))
    end, channel)
end

# One draw's law arguments. A hyperparameter with a known value has no coordinate.
function _brm_gmrf_draw_law(rk::RKBRMI, record, f, u)
    values = _rk_constrained_draw(rk, u)
    term = getproperty(values, record.declaration)
    known(slot) = (prior = getproperty(record.locals.priors, slot);
        prior isa Real ? Float64(prior) : Float64(getproperty(term, getproperty(record.locals, slot))))
    correlation = record.correlation
    L = correlation === nothing ? Matrix{Float64}(I, f.nchannels, f.nchannels) :
        correlation.factor !== nothing ? Matrix{Float64}(correlation.factor) :
        Matrix{Float64}(getproperty(values, correlation.name))
    (; sigma=known(:sigma), spatial_length=known(:spatial_length),
       temporal_length=known(:temporal_length), L,
       field=getproperty(term, record.locals.field))
end
