# Planner summaries used to validate a coordinate field's frozen replay.
# The graph fits and freezes its grid itself; this keeps only the level map,
# covered region, never observation rows. The default spatial prior is fitted
# in the graph alongside its grid.
function _rk_multigroup_replay_fit(lc, source, id, data)
    _brm_fitted_value(:multigroup, id) do
        frozen = _brm_frozen_replay()
        training = frozen === nothing ? Dict{Symbol,Any}() : frozen.training
        fitted_levels = _brm_fit_levels(get(training, source, data[source]))
        coordinates = hcat((get(training, axis, data[axis]) for axis in lc.axes)...)
        lower, upper = vec(minimum(coordinates; dims=1)), vec(maximum(coordinates; dims=1))
        center = (lower .+ upper) ./ 2
        half = (upper .- lower) ./ 2
        isempty(lc.positions) && lc.method === :hsgp && (half = lc.c .* half)
        locations = Dict(label => Tuple(coordinates[i, :])
            for (i, label) in enumerate(fitted_levels))
        (; locations, center, half)
    end
end

function _rk_check_multigroup_replay(fit, levels, coordinates; source)
    size(coordinates, 1) == length(levels) || error(
        "RK reprocess: coordinate multigroup `$source` needs one location per " *
        "declared group level ($(length(levels))), got $(size(coordinates, 1))")
    all(isfinite, coordinates) || error(
        "RK reprocess: coordinate multigroup `$source` locations must be finite")
    for (i, label) in enumerate(levels)
        haskey(fit.locations, label) || continue
        Tuple(coordinates[i, :]) == fit.locations[label] || error(
            "RK reprocess: coordinate multigroup `$source` level `$label` has " *
            "coordinates different from its fitted location $(fit.locations[label])")
    end
    _brm_check_multigroup_region(coordinates, fit.center, fit.half;
        prefix="RK reprocess: coordinate multigroup `$source`")
    nothing
end

# Emit summaries outside the parameter-dependent HSGP block so RKPPL can
# partially evaluate them and fitted-state export can freeze them by value.
function _rk_ast_multigroup_fit!(definitions, statements, taken, term)
    options = term.options
    lc = get(options, :level_cor, nothing)
    lc isa _BRMMultigroupCoordinates || return
    # The weights' size reads a fitted value, as an hsgp term's `modes` does.
    push!(taken, options.coordinate_points)
    push!(statements, Expr(:(=), options.coordinate_points,
        Expr(:call, :copy, options.coordinate_points_fit)))
    if isempty(lc.positions)
        name = options.coordinate_grid
        push!(taken, name)
        coordinates = Expr(:call, :hcat, options.coordinate_fit_columns...)
        call = Expr(:call, :brm_multigroup_coordinate_grid, coordinates,
            Expr(:ref, options.coordinate_k_fit, 1), _brm_multigroup_method_code(lc.method),
            Expr(:ref, options.coordinate_c_fit, 1))
        push!(statements, Expr(:(=), name, call))
    end
    if isnothing(lc.lambda)
        name = options.coordinate_prior
        push!(taken, name)
        formals = [Symbol(:coordinate_, j) for j in eachindex(options.coordinate_fit_columns)]
        method = _brm_multigroup_method_code(lc.method)
        n_points = isempty(lc.positions) ? 0 : :(points_level[1])
        # A data kernel materializes the prior's two numbers at binding, as
        # the temporal rho-floor kernel does. Layout construction reads the
        # frozen lower bound without traversing absent fit-role columns.
        body = quote
            coordinates = hcat($(formals...))
            prior = brm_multigroup_coordinate_prior(coordinates, k_level[1],
                $method, c_level[1], $n_points)
            values = [prior[1], prior[2]]
            return values
        end
        doc = @_rk_definition_doc """
            The two numbers of a multigroup HSGP's coordinate prior, from the training
            coordinates and the basis settings (`k`, `c`, number of points); computed once
            when the data are bound."""
        entry = _rk_ast_shared_definition!(definitions, taken,
            "brm_multigroup_prior_graph", [formals; :k_level; :c_level; :points_level],
            body; kernel=true, doc)
        call = Expr(:call, entry, options.coordinate_fit_columns...,
            options.coordinate_k_fit, options.coordinate_c_fit, options.coordinate_points_fit)
        push!(statements, Expr(:(=), name, call))
    end
    for (name, input) in zip(options.position_columns, options.position_fit_columns)
        input === nothing && continue
        push!(taken, name)
        push!(statements, Expr(:(=), name, Expr(:call, :copy, input)))
    end
    nothing
end

function _rk_design_multigroup_check(term, groups)
    options = term.options
    source, fit = options.group_column, options.multigroup_fit
    # Without group resampling, coordinate vectors still name all fitted
    # levels in fitted order; with it they name this window's declared levels.
    fixed_levels = options.group_levels
    design -> begin
        levels = source in groups ? _brm_fit_levels(_brm_df_column(design, source)) : fixed_levels
        columns = [_brm_df_column(design, axis) for axis in options.level_cor.axes]
        allequal(length.(columns)) || error(
            "RK reprocess: coordinate multigroup `$source` axes have different lengths")
        _rk_check_multigroup_replay(fit, levels, hcat(columns...); source)
    end
end
