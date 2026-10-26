# Readers of the multigroup coherence over known level coordinates,
# `hsgp(t; by = g, cor = multigroup(coordinates, lambda; k))`
# (src/level_correlation.jl), per posterior draw: each basis frequency's
# period, spatial range and coherence by distance, and the fitted term at given
# times and coordinates. Both read the same values the compiled model uses:
# the term's frequencies, basis interval and inducing grid from its emitted
# data, its length scale, amplitude, spatial scale, standardized weights and
# (when declared) free inducing positions from the draws.

"""
    brm_multigroup_coherence(d::BRMDescriptor, draws, constrained_names;
                             predictor=nothing, distances=Float64[]) -> NamedTuple

The multigroup coherence over known coordinates of a grouped HSGP,
`hsgp(t; by = g, cor = multigroup((x, y), lambda; k))`, per posterior draw.
`draws` is draws × constrained coordinates in `constrained_names` order, as
BridgeStan's `param_constrain` returns them. `predictor` selects the term's
linear predictor when the model has more than one such term.

- `term`, `predictor`, `group`, `coordinates` — the term's label, linear
  predictor, grouping factor and coordinate fields.
- `frequency`, `period` — per basis function, the angular frequency
  `ω = √(Σ_a ω_a²)` of the temporal basis and its period `2π / ω`, in the
  time axis' unit.
- `range` — draws × basis functions: each frequency's spatial range
  `λ / (ρ ω)`.
- `resolution`, `resolved` — the finest range the approximation resolves and,
  per draw and frequency, whether the range is at least that. While
  `λ ≥ 2√(ln 100) · resolution`, every unresolved frequency has less than 1/100
  of the zero-frequency spectral amplitude.
- `distances`, `coherence` — draws × basis functions × distances: the
  correlation `exp(-d² / (2 r²))` of two levels at each distance `d`.
"""
function brm_multigroup_coherence(d::BRMDescriptor, draws::AbstractMatrix,
                                  constrained_names; predictor=nothing,
                                  distances=Float64[])
    _brm_check_draw_names(draws, constrained_names)
    term = _brm_multigroup_coordinate_term(d, predictor)
    values = _brm_multigroup_draw_values(term, draws, constrained_names)
    _brm_multigroup_coherence(term, values, collect(Float64, distances))
end

"""
    brm_multigroup_field(d::BRMDescriptor, draws, constrained_names;
                         times, coordinates, predictor=nothing) -> Array

The fitted multigroup term `f(t, s)` of a grouped HSGP with known level
coordinates at every pair of `times` (a vector on the term's time axis) and
locations (`coordinates`, one row per location and one column per coordinate
axis): an array of draws × times × locations. At a fitted level's own
coordinates this is that level's fitted series; at any other location it is
the term's prediction there from the same draws (a neighbour prediction for a
held-out level). `draws` and `constrained_names` are as for
[`brm_multigroup_coherence`](@ref).
"""
function brm_multigroup_field(d::BRMDescriptor, draws::AbstractMatrix,
                              constrained_names; times, coordinates,
                              predictor=nothing)
    _brm_check_draw_names(draws, constrained_names)
    term = _brm_multigroup_coordinate_term(d, predictor)
    values = _brm_multigroup_draw_values(term, draws, constrained_names)
    _brm_multigroup_field(term, values, collect(Float64, times),
                          _brm_multigroup_locations(term, coordinates))
end

# The emitted grouped-HSGP declaration of a coordinate multigroup term and the
# data it reads: one per model, or the one on `predictor`. Each family has a
# `_lambda` twin that declares the term's own spatial scale.
_brm_multigroup_family(x, names...) =
    any(n -> x.family === n || x.family === Symbol(n, :_lambda), names)
_brm_multigroup_iso(x) = _brm_multigroup_family(x,
    :_sb_hsgp_by_multigroup_coordinates, :_sb_hsgp_by_multigroup_inducing)
_brm_multigroup_free(x) = _brm_multigroup_family(x,
    :_sb_hsgp_by_multigroup_inducing, :_sb_hsgp_by_aniso_multigroup_inducing)
function _brm_multigroup_coordinate_term(d::BRMDescriptor, predictor)
    found = [x for x in d.plan.declarations if _brm_multigroup_free(x) || _brm_multigroup_family(x,
        :_sb_hsgp_by_multigroup_coordinates, :_sb_hsgp_by_aniso_multigroup_coordinates)]
    matches = isnothing(predictor) ? found :
        [x for x in found if _brm_multigroup_term_predictor(d, x) === predictor]
    isempty(found) && error(
        "brm_multigroup_coherence: the model has no `hsgp(...; by = g, cor = " *
        "multigroup(coordinates, lambda; k))` term")
    length(matches) == 1 || error(
        "brm_multigroup_coherence: " * (isempty(matches) ?
            "no coordinate multigroup term is on predictor `$predictor`" :
            "the model has $(length(matches)) coordinate multigroup terms; " *
            "select one with `predictor`"))
    x = only(matches)
    k = x.keywords
    data = d.plan.data
    coordinates = data[k.coordinates_level]
    common = (; target=x.target, predictor=_brm_multigroup_term_predictor(d, x),
        iso=_brm_multigroup_iso(x),
        omega2=_brm_multigroup_matrix(data[k.omega2]),
        fits=d.plan.preproc[k.PHI].const_.fits, K=d.plan.preproc[k.PHI].const_.K,
        coordinates, block=k.beta, n_basis=size(data[k.omega2], 1),
        # a declared or constant `lambda`, or the term's own (`<term>_lambda`)
        lambda=haskey(k, :lambda_level) ? k.lambda_level : Symbol(x.target, :_lambda))
    if _brm_multigroup_free(x)
        n_points = _brm_multigroup_position_rows(k.positions_level, data)
        return (; common..., free=true, positions=k.positions_level, data, n_points,
                r_min=brm_multigroup_inducing_resolution(coordinates, n_points))
    end
    (; common..., free=false, center=data[k.center_level], scale=data[k.scale_level],
       r_min=_brm_multigroup_resolution(_brm_multigroup_approximation(k.method_level),
                                        data[k.center_level], data[k.scale_level], k.k_level),
       grid_k=k.k_level, method=k.method_level,
       n_points=k.k_level^size(coordinates, 2))
end

# Free inducing positions as emitted (src/sbimpl.jl `_sb_multigroup_positions!`):
# a data matrix by key, or the axes side by side, each a data vector by key or
# a declared vector parameter by name, read per draw.
function _brm_multigroup_position_rows(spec, data)
    spec isa Symbol && return haskey(data, spec) ? size(data[spec], 1) : data[Symbol(spec, :_n)]
    spec.args[1] === :to_matrix && return spec.args[3]
    _brm_multigroup_position_rows(spec.args[2], data)
end
function _brm_multigroup_positions_at(spec, data, column, draws, s)
    if spec isa Symbol
        haskey(data, spec) && return data[spec]
        return [draws[s, column("$spec.$p")] for p in 1:data[Symbol(spec, :_n)]]
    end
    part(x) = _brm_multigroup_positions_at(x, data, column, draws, s)
    spec.args[1] === :to_matrix && return reshape(part(spec.args[2]), :, 1)
    spec.args[1] === :append_col && return hcat(part(spec.args[2]), part(spec.args[3]))
    error("brm_multigroup_field: internal: unexpected positions `$spec`")
end

_brm_multigroup_matrix(omega2::AbstractVector) = reshape(omega2, :, 1)
_brm_multigroup_matrix(omega2::AbstractMatrix) = omega2

function _brm_multigroup_term_predictor(d, x)
    for o in d.outputs
        o.name === x.target && !isempty(o.predictors) && return first(o.predictors)
    end
    nothing
end

# Per draw: the length scale(s), amplitude, spatial scale and the
# standardized grid weights, read from the constrained draws by name.
function _brm_multigroup_draw_values(term, draws, names)
    index = Dict(String(n) => i for (i, n) in enumerate(names))
    column(name) = get(index, name) do
        error("brm_multigroup_coherence: the draws have no `$name`; pass " *
              "BridgeStan's constrained names with `include_tp=true`")
    end
    target = string(term.target)
    D = size(term.omega2, 2)
    rho = term.iso ? draws[:, column(target * "_rho_iso")] :
        draws[:, [column(target * "_rho.$a") for a in 1:D]]
    sigma = draws[:, column(target * "_sigma")]
    lambda = term.lambda isa Symbol ? draws[:, column(string(term.lambda))] :
        fill(Float64(term.lambda), size(draws, 1))
    P, M = term.n_points, term.n_basis
    block = string(term.block)
    z = [[draws[s, column("$block.$p.$b")] for p in 1:P, b in 1:M] for s in axes(draws, 1)]
    positions = term.free ?
        [_brm_multigroup_positions_at(term.positions, term.data, column, draws, s)
         for s in axes(draws, 1)] : nothing
    (; rho, sigma, lambda, z, positions)
end

_brm_multigroup_rho(values, s) = values.rho isa AbstractMatrix ?
    values.rho[s, :] : values.rho[s]

function _brm_multigroup_coherence(term, values, distances)
    S, J = length(values.sigma), term.n_basis
    frequency = [sqrt(sum(term.omega2[b, :])) for b in 1:J]
    range = Matrix{Float64}(undef, S, J)
    for s in 1:S, b in 1:J
        e = _brm_hsgp_frequency_exponent(term.omega2, _brm_multigroup_rho(values, s), b)
        range[s, b] = values.lambda[s] / sqrt(e)
    end
    coherence = [exp(-d^2 / (2range[s, b]^2)) for s in 1:S, b in 1:J, d in distances]
    (; term=term.target, predictor=term.predictor, frequency, period=2pi ./ frequency,
       range, resolution=term.r_min, resolved=range .>= term.r_min, distances, coherence)
end

function _brm_multigroup_locations(term, coordinates)
    C = coordinates isa AbstractVector ? reshape(collect(Float64, coordinates), :, 1) :
        Matrix{Float64}(coordinates)
    size(C, 2) == size(term.coordinates, 2) || error(
        "brm_multigroup_field: `coordinates` needs one column per coordinate " *
        "axis ($(size(term.coordinates, 2))), got $(size(C, 2))")
    all(isfinite, C) || error("brm_multigroup_field: `coordinates` must be finite")
    # A location must lie where the approximation is defined at every
    # frequency: inside the inducing grid at its finest spacing or, for free
    # positions, the levels' bounding box; or inside the Hilbert-space box.
    lower, upper = vec(minimum(term.coordinates; dims=1)), vec(maximum(term.coordinates; dims=1))
    center, half = term.free ? ((lower .+ upper) ./ 2, (upper .- lower) ./ 2) :
        (term.center, term.method == 1 ? term.scale .* (term.grid_k - 1) ./ 2 : term.scale)
    for i in axes(C, 1), a in axes(C, 2)
        abs(C[i, a] - center[a]) <= half[a] * (1 + 1e-12) || error(
            "brm_multigroup_field: location $i lies outside the region the " *
            "approximation covers on axis $a ([$(center[a] - half[a]), " *
            "$(center[a] + half[a])]); it would be an extrapolation")
    end
    C
end

function _brm_multigroup_field(term, values, times, locations)
    D = size(term.omega2, 2)
    D == 1 || error("brm_multigroup_field: the term has $D time axes; " *
                    "`times` reads one")
    center, L = only(term.fits)
    all(t -> abs(t - center) <= L, times) || error(
        "brm_multigroup_field: `times` must lie in the term's basis interval " *
        "[$(center - L), $(center + L)]")
    PHI, _ = _brm_apply_hsgp(term.fits, (times,), term.K)
    S = length(values.sigma)
    out = Array{Float64}(undef, S, length(times), size(locations, 1))
    for s in 1:S
        rho = _brm_multigroup_rho(values, s)
        scale = brm_hsgp_sqrt_spd(term.omega2, values.sigma[s], rho)
        W = term.free ?
            brm_hsgp_multigroup_inducing_weights(values.z[s], term.omega2, rho, locations,
                values.positions[s], values.lambda[s]) :
            brm_hsgp_multigroup_coordinate_weights(values.z[s], term.omega2, rho,
                locations, term.center, term.scale, values.lambda[s], term.method)
        out[s, :, :] = PHI * (scale .* transpose(W))
    end
    out
end
