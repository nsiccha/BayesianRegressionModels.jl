# A separable spatial × periodic HSGP. Spatial axes live on distinct locations;
# the last, periodic axis lives on observation rows. `at=` maps observations to
# locations, so the spatial basis and its intercept projection are fitted once
# per location. The period axis has Fourier harmonics but no boundary box.
function _brm_prepare_space_periodic_term(term::ExprColumn{typeof(hsgp)},
                                          target::Symbol, context)
    kw = getkwargs(term)
    args = getargs(term)
    n_axes = length(args)
    n_axes >= 2 || error(
        "hsgp: `periodic=` needs at least one spatial axis and one periodic axis")
    periodic = kw.periodic
    periodic isa Symbol || error(
        "hsgp: `periodic=` names the last axis as a Symbol")
    names = Tuple(name(arg) for arg in args if arg isa NamedColumn)
    length(names) == n_axes && last(names) === periodic || error(
        "hsgp: `periodic=$(repr(periodic))` must name the last raw-data axis")
    for key in (:by, :cor, :rows, :anchor, :extrapolate, :multiply)
        haskey(kw, key) && error(
            "hsgp: spatial × periodic `hsgp(...; periodic=...)` does not accept `$key=`")
    end
    haskey(kw, :domain) || error(
        "hsgp: spatial × periodic `hsgp(...; periodic=...)` needs the " *
        "spatial `domain=` to fix its basis box")
    n_space = n_axes - 1
    K = _brm_axis_option(:hsgp, :k, get(kw, :k, 20), n_axes,
        x -> x isa Integer && !(x isa Bool) && x >= 1, "positive integers")
    K = Tuple(Int(k) for k in K)
    K_space, K_period = K[1:end-1], last(K)
    iso = _brm_gp_iso(kw, :hsgp)
    iso || error(
        "hsgp: spatial × periodic HSGP currently shares one spatial " *
        "length scale; use `iso=true`")
    period = _brm_gp_period(kw, :hsgp, :periodic)
    fits = _brm_hsgp_domain_fits(kw, n_space; required=true)
    orthogonal = _brm_hsgp_orthogonal_to(kw, n_space)
    orthogonal === :linear && error(
        "hsgp: spatial × periodic `orthogonal_to=:linear` is not supported")
    spatial_sources = names[1:end-1]
    spatial_axes = Tuple(begin
        source, raw = _brm_term_data(:hsgp, arg, context)
        values = _brm_axis_values(raw, nothing,
            "BRM term preparation: spatial HSGP axis `$source`")
        isempty(values) && error("hsgp: spatial axes need at least one location")
        all(isfinite, values) || error("hsgp: spatial axes must be finite")
        values
    end for arg in args[1:end-1])
    n_locations = length(first(spatial_axes))
    all(axis -> length(axis) == n_locations, spatial_axes) || error(
        "hsgp: spatial axes must have equal lengths")
    _, raw_time = _brm_term_data(:hsgp, last(args), context)
    time = _brm_axis_values(raw_time, nothing,
        "BRM term preparation: periodic HSGP axis `$periodic`")
    all(isfinite, time) || error("hsgp: periodic axis must be finite")
    at = _brm_hsgp_at(kw, context, n_locations)
    isnothing(at) && error(
        "hsgp: spatial × periodic `hsgp(...; periodic=...)` needs `at=` " *
        "to map observation rows to spatial locations")
    length(at.idx) == length(time) || error(
        "hsgp: `at=$(at.source)` has $(length(at.idx)) rows but periodic " *
        "axis `$periodic` has $(length(time))")
    centeredness = _brm_hsgp_centeredness(kw, prod(K_space) * 2K_period)
    all(iszero, centeredness) || error(
        "hsgp: spatial × periodic weights currently use noncentred coordinates; " *
        "external adaptive centring remains available")
    _brm_term_prior_spec(term, target, context, :term_length_scale) === nothing ||
        error("hsgp: an explicit length-scale prior for spatial × periodic " *
              "HSGP is not yet supported")
    spatial = _brm_hsgp_basis_state(spatial_axes, K_space, :exp_quad, iso,
        nothing; fits, orthogonal)
    temporal = _brm_hsgp_basis_state((time,), (K_period,), :periodic, true,
        period)
    spatial_floor = spatial.rho_lower isa Real ?
        [Float64(spatial.rho_lower)] : Float64.(spatial.rho_lower)
    rho_lower = [spatial_floor; Float64(temporal.rho_lower)]
    setting_fields = _brm_setting_fields(kw, (:k, :domain, :period))
    state = (; target, latent=false, cov=:space_periodic, iso,
        K, K_space, K_period, period, fits, orthogonal, at, spatial, temporal,
        rho_lower, centeredness, setting_fields,
        _brm_gp_priors(term, target, context)...)
    deps = (names..., at.source, (first.(setting_fields))...)
    _BRMPreparedTerm(hsgp, names, state, deps)
end
