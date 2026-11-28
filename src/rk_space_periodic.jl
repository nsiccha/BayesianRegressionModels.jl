# A spatial tensor basis is evaluated on distinct locations, while the Fourier
# basis is evaluated on observation rows. The only observation-sized arrays are
# the small Fourier basis and the final predictor. In particular, the product
# basis is never expanded into an observations × weights matrix.
function _rk_plan_space_periodic_hsgp_term!(prepared, target, data, columns, taken;
        derived, hyper_plans, rho_stated, sigma_stated)
    state = prepared.state
    isempty(hyper_plans) || error(
        "RK backend: hyper-predictors on a spatial × periodic HSGP are not supported")
    (rho_stated || sigma_stated) && error(
        "RK backend: explicit length-scale and SD declarations on a spatial × " *
        "periodic HSGP are not supported")
    axes = Tuple(prepared.source)
    for axis in axes
        _rk_plan_spline_axis!(axis, target, "hsgp", data, columns)
    end
    columns[state.at.source] = data[state.at.source]
    id = _rk_mint_smooth_id!(taken, columns,
        "hsgp_" * join(string.(axes), "_"))
    modes = _rk_fit_input!(derived, columns, taken, Symbol(id, :_modes),
        collect(Int, state.K_space))
    # The first D slots are reserved for a boundary factor in the shared
    # exp-quad graph; a fixed domain reads only the following center/widths.
    D = length(state.K_space)
    settings = _rk_fit_input!(derived, columns, taken, Symbol(id, :_settings),
        Float64[ones(D)..., Iterators.flatten(state.fits)...])
    period_modes = _rk_fit_input!(derived, columns, taken,
        Symbol(id, :_period_modes), [state.K_period])
    period_settings = _rk_fit_input!(derived, columns, taken,
        Symbol(id, :_period_settings), [Float64(state.period)])
    options = (; id, k=state.K, cov=:space_periodic, iso=state.iso,
        spatial_axes=axes[1:end-1], periodic_axis=last(axes),
        fits=state.fits, fixed_fits=state.fits, latent=false,
        orthogonal=state.orthogonal,
        at=state.at.source, modes, settings, period_modes, period_settings,
        rho_prior=state.rho_prior, sigma_prior=state.sigma_prior,
        rho_truncated=true, rho_floor_named=false, hyper_plans=(),
        rho_stated=false, sigma_stated=false)
    _RKTermSpec(:hsgp, collect(axes), options, id, id)
end

function _rk_space_periodic_proxy(term, cov, columns, modes, settings)
    k = cov === :periodic ? last(term.options.k) : term.options.k[1:end-1]
    options = merge(term.options, (; cov, k, modes, settings,
        fixed_fits=cov === :periodic ? nothing : term.options.fixed_fits,
        orthogonal=cov === :periodic ? nothing : term.options.orthogonal,
        mask=nothing, fit_axes=nothing, anchor=nothing))
    _RKTermSpec(:hsgp, collect(columns), options, term.addressee, term.label)
end

function _rk_ast_value_space_periodic_hsgp(definitions, term, taken, bindings)
    stmts = Expr[]
    term = _rk_ast_hsgp_settings!(stmts, term, taken)
    o = term.options
    pm = _rk_ast_fresh_name(string(o.id, "_period_modes"), taken)
    ps = _rk_ast_fresh_name(string(o.id, "_period_settings"), taken)
    push!(stmts, :($pm = copy($(o.period_modes))),
        :($ps = copy($(o.period_settings))))
    spatial = _rk_space_periodic_proxy(term, :exp_quad, o.spatial_axes,
        o.modes, o.settings)
    temporal = _rk_space_periodic_proxy(term, :periodic, (o.periodic_axis,), pm, ps)
    space_basis = _rk_ast_hsgp_basis_graph!(definitions, spatial, taken)
    period_basis = _rk_ast_hsgp_periodic_basis_graph!(definitions, temporal, taken)
    P = _rk_ast_fresh_name(string(o.id, "_space_basis"), taken)
    W = _rk_ast_fresh_name(string(o.id, "_space_omega2"), taken)
    Q = _rk_ast_fresh_name(string(o.id, "_period_basis"), taken)
    H = _rk_ast_fresh_name(string(o.id, "_harmonics"), taken)
    push!(stmts,
        Expr(:(=), Expr(:tuple, P, W),
            Expr(:call, space_basis, o.spatial_axes..., o.modes, o.settings)),
        Expr(:(=), Expr(:tuple, Q, H),
            Expr(:call, period_basis, o.periodic_axis, pm, ps)))
    sf = _rk_ast_hsgp_floor!(definitions, stmts, spatial, taken)
    pf = _rk_ast_hsgp_periodic_floor!(definitions, stmts, temporal, taken, bindings)
    floor = _rk_ast_fresh_name(string(o.id, "_rho_floor"), taken)
    push!(stmts, :($floor = [$sf, $pf]))
    bessel = _rk_value_callee!(bindings, SpecialFunctions.besselix, taken)
    scale_body = Base.remove_linenums!(quote
        space_freq = vec(sum(omega2_space; dims=2))
        space_scale = sigma * (rho[1] * sqrt(2pi))^(size(omega2_space, 2) / 2) .*
            exp.(-0.25 .* rho[1]^2 .* space_freq)
        a = inv(rho[2]^2)
        period_scale = $(_rk_ast_graph_plate([:(1:length(harmonics))], [:j],
            :(sqrt(2 * $bessel(harmonics[j], a)))))
        return vec(space_scale .* transpose(period_scale))
    end)
    scale_fn = _rk_ast_shared_definition!(definitions, taken,
        "brm_space_periodic_hsgp_scales",
        [:omega2_space, :harmonics, :sigma, :rho], scale_body; kernel=true)
    value_body = Base.remove_linenums!(quote
        weights = reshape(scales .* z, size(PHI_space, 2), size(PHI_period, 2))
        fields = PHI_space * weights
        rows = fields[row_index, :]
        return vec(sum(rows .* PHI_period; dims=2))
    end)
    value_fn = _rk_ast_shared_definition!(definitions, taken,
        "brm_space_periodic_hsgp_rows",
        [:PHI_space, :PHI_period, :row_index, :scales, :z], value_body;
        kernel=true)
    inputs = Pair{Symbol,Any}[:PHI_space => P, :omega2_space => W,
        :PHI_period => Q, :harmonics => H, :row_index => o.at]
    priors = [_rk_ast_hsgp_rho_prior(o, floor, j, bindings, taken) for j in 1:2]
    sigma_prior = _rk_ast_positive_prior(o.sigma_prior, bindings, taken)
    scales = Expr(:call, scale_fn, :omega2_space, :harmonics, :sigma, :rho)
    value = Expr(:call, value_fn, :PHI_space, :PHI_period, :row_index,
        :spectral_scale, :beta_raw)
    extent = :(1:(size(omega2_space, 1) * size(PHI_period, 2)))
    _rk_ast_hsgp_component!(definitions, stmts, taken, o.id, inputs, floor,
        priors, sigma_prior, value, extent; scales,
        base="brm_space_periodic_hsgp_effect")
    stmts, nothing, nothing
end
