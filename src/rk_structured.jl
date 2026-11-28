# General prepared group fields: BRM owns prior declarations and block algebra;
# the consumer's existing native effect owns how each block enters the term.
# One term's fitted statistical inputs cross through an ordinary data port.
# Capturing its arrays in a callable would hide them from the AD interface.
function brm_structured_effect(block, prepared_inputs, field_index)
    prepared = only(prepared_inputs)
    _brm_native_structured_effect(prepared, block, prepared.state.fields[field_index])
end

# The authored prior of one HSGP hyperparameter, in the original statement
# order: a shared scale's positive law, or a stated hyper-intercept law.
function _rk_ast_hsgp_hyper_prior(options, hyper, bindings, taken)
    prior = hyper === :length_scale ? options.rho_prior : options.sigma_prior
    position = findfirst(p -> p.hyper === hyper, options.hyper_plans)
    position === nothing && return _rk_ast_positive_prior(prior, bindings, taken)
    stated = hyper === :length_scale ? options.rho_stated : options.sigma_stated
    options.hyper_plans[position].intercept && stated ?
        _rk_ast_declared_prior(prior, bindings, taken) : nothing
end

# The hyperparameter's value, its per-group effects `(; scale, z)` when its
# authored predictor has a `(1 | g)` term (`nothing` otherwise), and the local
# it is sampled as: the hyperparameter itself, or its hyper-predictor's
# intercept (`nothing` for a known value or an intercept-free predictor).
function _rk_ast_hsgp_hyper!(block, options, hyper, prior, floor, G)
    stated = hyper === :length_scale ? options.rho_stated : options.sigma_stated
    plans = options.hyper_plans
    position = findfirst(p -> p.hyper === hyper, plans)
    stem = hyper === :length_scale ? "rho" : "sigma"
    if position === nothing
        value = _rk_block_local!(block, stem)
        hyper === :length_scale && options.rho_truncated &&
            (prior = Expr(:call, :restricted, prior, floor, Inf))
        statement = _rk_ast_hyper_statement(value, prior)
        push!(block.statements, statement)
        return value, nothing, Meta.isexpr(statement, :call) ? value : nothing
    end
    plan = plans[position]
    parts = Any[]
    beta = nothing
    if plan.intercept
        beta = _rk_block_local!(block, string(stem, "_Intercept"))
        push!(block.statements, Expr(:call, :~, beta,
            stated ? prior : Expr(:call, :Normal, 0, 1)))
        push!(parts, beta)
    elseif stated
        error("RK backend: HSGP hyper prior needs an intercept in its authored predictor")
    end
    ranef = nothing
    if !isempty(plan.ranefs)
        sd = _rk_block_local!(block, string(stem, "_sd"))
        z = _rk_block_local!(block, string(stem, "_z"))
        push!(block.statements, Expr(:call, :~, sd,
            Expr(:call, :restricted, Expr(:call, :Normal, 0, 1), 0.0, Inf)))
        push!(block.statements, Expr(:call, :.~, Expr(:ref, z, Expr(:call, :(:), 1, G)),
            _rk_ast_dotted(:Normal, 0, 1)))
        push!(parts, Expr(:call, :.*, sd, z))
        ranef = (; scale=sd, z)
    end
    eta = length(parts) == 1 ? only(parts) : Expr(:call, :.+, parts...)
    value = _rk_block_local!(block, string(stem, "_value"))
    expression = _rk_ast_dotted(:exp, eta)
    hyper === :length_scale && (expression = _rk_ast_dotted(:max, expression, floor))
    push!(block.statements, Expr(:(=), value, expression))
    value, ranef, beta
end

# The value a grouped HSGP's per-level rows are sized by: a collection cell's
# fitted level table, or the term's in-graph level set (`g_fit_levels`, the
# level set its group index reads). `nothing` for a coordinate multigroup
# field, whose weights have one row per point, and an ungrouped term.
_rk_hsgp_level_set(options) = get(options, :collection_levels, get(options, :level_set, nothing))

# A grouped HSGP term with the level set its group index reads (`level_sets`,
# by index), unless a collection cell supplies its own or its weights are a
# coordinate field's.
function _rk_hsgp_with_level_set(term, level_sets, literal_hsgp_levels)
    options = term.options
    haskey(options, :group_fit_source) && !haskey(options, :collection_levels) &&
        !(get(options, :level_cor, nothing) isa _BRMMultigroupCoordinates) || return term
    # For the identity check of an older export, reproduce its literal row
    # count from the saved levels, including on a design with new levels.
    literal = get(literal_hsgp_levels, options.group_index, nothing)
    size = literal === nothing ? (; level_set=level_sets[options.group_index]) :
        (; n_groups=length(literal))
    _RKTermSpec(term.kind, term.columns, merge(options, size), term.addressee,
        term.label)
end

# Grouped or hyper-predicted HSGP: one block allocating the hyperparameters,
# their group-level effects and the basis weights, returning the summand. Also
# returns the block's sampled locals that its coordinate record names: the
# standardized weights `z` (one row per level, or per point of a multigroup
# coordinate field), each hyperparameter's group effects (`hypers`), the local
# each hyperparameter is sampled as (`scales`: itself, or its hyper-predictor's
# intercept) and a multigroup coordinate field's own spatial scale (`lambda`).
function _rk_ast_hsgp_grouped(definitions, term, inputs, floors, extent, taken, bindings)
    options = term.options
    G = get(options, :n_groups, 1)
    level_set = _rk_hsgp_level_set(options)
    rho_priors = options.iso ?
        [_rk_ast_hsgp_hyper_prior(options, :length_scale, bindings, taken)] :
        [_rk_ast_hsgp_rho_prior(options, floors, j, bindings, taken)
         for j in eachindex(term.columns)]
    # A floor named in the length-scale prior is read there (decision 1fk9b29);
    # only a truncation or a hyper-predictor reads it as a block argument.
    floor_named = get(options, :rho_floor_named, false)
    floor_named && options.iso &&
        (rho_priors[1] = _rk_ast_replace_floor_marker(only(rho_priors), floors))
    sigma_prior = _rk_ast_hsgp_hyper_prior(options, :sd, bindings, taken)
    grouped = haskey(options, :group_index)
    block = _rk_block_body(rho_priors..., sigma_prior)
    # The level set comes first, and the block reads only its length: a design
    # binder replaces it with the fit's to keep a draw of the fit's
    # coordinates (`_rk_sizes_only`).
    if level_set !== nothing
        levels = _rk_block_argument!(block, :fitted_levels, level_set)
        G = :(length($levels))
    end
    arguments = Pair{Symbol,Any}[formal => _rk_block_argument!(block, formal, value)
        for (formal, value) in inputs]
    floor = floors === nothing || floor_named ? nothing :
        _rk_block_argument!(block, :floor, floors)
    group_index = grouped ? _rk_block_argument!(block, :group_index, options.group_index) :
        nothing
    rho, rho_effects, rho_sampled = _rk_ast_hsgp_lengths!(block, options,
        rho_priors, floor, G, Val(options.iso))
    sigma, sigma_effects, sigma_sampled = _rk_ast_hsgp_hyper!(block, options, :sd,
        sigma_prior, floor, G)
    z = _rk_block_local!(block, :z)
    # A periodic extent reads its prepared basis argument.
    extent = _rk_ast_hsgp_substitute(extent, Dict(arguments))
    level_cor = get(options, :level_cor, nothing)
    weights = grouped ? (Expr(:call, :(:), 1,
                              _rk_ast_hsgp_weight_rows!(block, level_cor, options, G)),
                         extent) : (extent,)
    push!(block.statements, Expr(:call, :.~, Expr(:ref, z, weights...),
        _rk_ast_dotted(:Normal, 0, 1)))
    weights_value, multigroup = _rk_ast_hsgp_level_weights!(block, z, level_cor, options)
    # The weights' spectral scales are a value of the block (`spectral`).
    spectral = _rk_block_local!(block, :spectral_scale)
    value = _rk_block_local!(block, :value)
    scales_call, value_call = _rk_ast_hsgp_value_graph!(definitions, term, taken, arguments,
        sigma, rho, weights_value, spectral, bindings; group_index, multigroup)
    push!(block.statements, Expr(:(=), spectral, scales_call), Expr(:(=), value, value_call))
    doc = @_rk_definition_doc """
        An HSGP term$(G isa Expr ? " with one coefficient row per fitted group" : G > 1 ? " with $G groups" : "") whose length scale and marginal scale may vary by group.

        Draws the length scale(s) `$rho` and marginal scale(s) `$sigma` with their
        priors (and their group-level effects, when they have predictors), and the
        standardized basis coefficients `$z ~ Normal(0, 1)`. Each basis function is weighted
        by the square root of the covariance's spectral density (`$spectral`); the term at
        each row is its basis times the weighted coefficients of its group."""
    _rk_ast_block_call!(definitions, taken, "brm_grouped_hsgp", block, value; doc),
        (; weights=z, hypers=(length_scale=rho_effects, sd=sigma_effects),
           scales=(length_scale=rho_sampled, sd=sigma_sampled), spectral,
           lambda=isnothing(multigroup) ? nothing : get(multigroup, :lambda, nothing))
end

_rk_ast_hsgp_lengths!(block, options, priors, floor, G, ::Val{true}) =
    _rk_ast_hsgp_hyper!(block, options, :length_scale, only(priors), floor, G)

# Each axis has its own positive parameter and bound, just as in the
# ungrouped component. The spectral graph consumes their assembled vector;
# the sampled locals retain axis order for coordinate transport and starts.
function _rk_ast_hsgp_lengths!(block, options, priors, floor, _G, ::Val{false})
    rhos = Symbol[]
    sampled = Symbol[]
    for (j, prior) in enumerate(priors)
        rho = _rk_block_local!(block, "rho_$j")
        push!(rhos, rho)
        lower = length(priors) == 1 ? floor : Expr(:ref, floor, j)
        options.rho_truncated && (prior = Expr(:call, :restricted, prior, lower, Inf))
        statement = _rk_ast_hyper_statement(rho, prior)
        push!(block.statements, statement)
        Meta.isexpr(statement, :call) && push!(sampled, rho)
    end
    value = _rk_block_local!(block, :rho)
    push!(block.statements, Expr(:(=), value, Expr(:vect, rhos...)))
    value, nothing, isempty(sampled) ? nothing : Tuple(sampled)
end

# The level weights of a grouped HSGP's `cor=` (src/level_correlation.jl).
# `cor=Omega` correlates every basis function's group weights by the factor's
# `L`; a `multigroup` coherence hands its arguments to the spectral graph,
# which forms the frequency-dependent weights once it has the frequencies:
# `weights` is that graph statement over `z`, `omega2`, `rho` and `formals`,
# bound to the block values `values`.
_rk_ast_hsgp_level_weights!(_block, z, ::Nothing, _options) = (z, nothing)
function _rk_ast_hsgp_level_weights!(block, z, lc::_BRMLevelCorrelation, _options)
    L = _rk_block_argument!(block, :L_level, _rk_ast_level_factor_value(lc))
    weights = _rk_block_local!(block, :z_level)
    push!(block.statements, Expr(:(=), weights, Expr(:call, :*, L, z)))
    (weights, nothing)
end
function _rk_ast_hsgp_level_weights!(block, z, lc::_BRMMultigroupCorrelation, options)
    L = _rk_block_argument!(block, :L_level, _rk_ast_level_factor_value(lc.base))
    tau = _rk_block_argument!(block, :tau_level, something(lc.tau_source, lc.tau))
    exponent = _rk_ast_hsgp_frequency_exponent(options, length(_rk_hsgp_modes(options)))
    # One independent Cholesky-times-vector product per frequency. Set the
    # diagonal explicitly, as in brm_hsgp_multigroup_weights, so it has no
    # derivative with respect to the sampled level factor.
    columns = _rk_ast_graph_plate([:(axes(z, 2))], [:b], quote
        exponent = $exponent
        coherence = ifelse.(same_level, 1.0 + 1e-6, exp.((-exponent * tau_level) .* distance))
        factor = ReactiveKernels.rk_cholesky_lower(coherence)
        factor * z[:, b]
    end)
    weights = quote
        omega = L_level * transpose(L_level)
        distance = 1 .- omega
        levels_axis = 1:size(L_level, 1)
        same_level = levels_axis .== transpose(levels_axis)
        weight_columns = $columns
        stack(weight_columns; dims=2)
    end
    (z, (; weights, formals=[:L_level, :tau_level], values=Any[L, tau]))
end
# The coordinates are current locations; the grid is a fitted graph value.
function _rk_ast_hsgp_level_weights!(block, z, lc::_BRMMultigroupCoordinates, options)
    columns = Any[_rk_block_argument!(block, Symbol(:coordinate_, j), column)
                  for (j, column) in enumerate(options.coordinate_columns)]
    positions = Any[_rk_block_argument!(block, Symbol(:position_, j), column)
                    for (j, column) in enumerate(get(options, :position_columns, ()))]
    own = isnothing(lc.lambda)
    lambda = own ? _rk_multigroup_own_lambda!(block, options.coordinate_prior) :
        _rk_block_argument!(block, :lambda_level, something(lc.lambda_source, lc.lambda))
    # The term's own spatial scale is a sampled local of the block.
    own_lambda = own ? (; lambda) : (;)
    formals = [Symbol(:coordinate_, j) for j in eachindex(columns)]
    if !isempty(lc.positions)
        position_formals = [Symbol(:position_, j) for j in eachindex(positions)]
        weights = quote
            coordinates_level = hcat($(formals...))
            positions_level = hcat($(position_formals...))
            brm_hsgp_multigroup_inducing_weights(z, omega2, rho, coordinates_level,
                positions_level, lambda_level)
        end
        return (z, (; weights, formals=[formals; position_formals; :lambda_level],
                    values=Any[columns; positions; lambda], own_lambda...))
    end
    method = _brm_multigroup_method_code(lc.method)
    grid = _rk_block_argument!(block, :grid_level, options.coordinate_grid)
    weights = quote
        coordinates_level = hcat($(formals...))
        brm_hsgp_multigroup_coordinate_weights(z, omega2, rho, coordinates_level,
            grid_level, lambda_level, $method)
    end
    (z, (; weights, formals=[formals; :grid_level; :lambda_level],
           values=Any[columns; grid; lambda],
           own_lambda...))
end

# `multigroup((x, y); k)`: the term's own spatial scale, with SBBRMI's prior
# (`_sb_multigroup_own_lambda`, decision `100gkpq`).
function _rk_multigroup_own_lambda!(block, prior)
    fitted = _rk_block_argument!(block, :spatial_prior, prior)
    location, lower = Expr(:ref, fitted, 1), Expr(:ref, fitted, 2)
    lambda = _rk_block_local!(block, :lambda)
    push!(block.statements, Expr(:call, :~, lambda,
        Expr(:call, :restricted, Expr(:call, :LogNormal, location, 1.0), lower, Inf)))
    lambda
end

# The rows of a grouped HSGP's standardized weights: one per level, or one per
# grid point, basis function or inducing position for
# `multigroup(coordinates, lambda; ...)`.
_rk_ast_hsgp_weight_rows!(_block, _level_cor, _options, n_groups) = n_groups
_rk_ast_hsgp_weight_rows!(block, ::_BRMMultigroupCoordinates, options, _n_groups) =
    Expr(:call, :prod, _rk_block_argument!(block, :points_level, options.coordinate_points))
function _rk_ast_structured_block(definitions, field, taken, bindings)
    K, G = field.n_per_group, length(field.levels)
    K isa Integer && K > 0 || error("RK backend: structured field needs a positive width")
    G > 0 || error("RK backend: structured field has no declared group levels")
    block = _rk_ast_fresh_name(string("b_", field.name, "_", field.source), taken)
    axes = (Expr(:call, :(:), 1, G), Expr(:call, :(:), 1, K))
    stmts = Expr[]
    if field.prior === :correlated_normal
        call = _rk_ast_varying_draws!(definitions, taken, K, 1.0, first(axes);
            scale_priors=Expr(:call, :restricted, Expr(:call, :Normal, 0, 1), 0.0, Inf))
        push!(stmts, Expr(:call, :~, block, call))
    else
        z = _rk_ast_fresh_name(string(block, "_z"), taken)
        prior = field.prior === :iid_normal ? ExprColumn(Normal, 0, 1) :
            _brm_structured_prior_expression(field)
        expression = _rk_value_expr!(bindings, _brm_prepare_expr(prior), taken)
        if field.prior isa NamedTuple &&
                (haskey(field.prior, :lower) || haskey(field.prior, :upper))
            expression = Expr(:call, :truncated, expression,
                get(field.prior, :lower, -Inf), get(field.prior, :upper, Inf))
        end
        push!(stmts, Expr(:call, :.~, Expr(:ref, z, axes...),
            _rk_ast_dotted(expression.args[1], expression.args[2:end]...)))
        push!(stmts, Expr(:(=), block, z))
    end
    block, stmts
end

function _rk_ast_structured_term(definitions, term, blocks, taken, bindings)
    prepared = term.options.prepared
    stmts = Expr[]
    indices = Symbol[]
    for (field, source) in zip(prepared.state.fields, term.options.field_sources)
        index = _rk_ast_fresh_name(string(term.options.id, "_", field.name, "_indices"), taken)
        call = _rk_ast_statistical_call!(definitions, taken, :brm_prepared_indices,
            source, _rk_ast_level_values(field.levels); kernel=true)
        push!(stmts, Expr(:(=), index, call))
        push!(indices, index)
    end
    inputs = _rk_ast_statistical_call!(definitions, taken, :brm_structured_inputs,
        term.options.metadata, Expr(:tuple, indices...))
    push!(stmts, Expr(:(=), term.options.prepared_data, inputs))
    parts = Symbol[]
    for (field_index, field) in enumerate(prepared.state.fields)
        key = (field.name, field.source)
        signature = (field.n_per_group, field.levels, field.prior)
        block = if haskey(blocks, key)
            old = blocks[key]
            isequal(old.signature, signature) || error(
                "RK backend: structured field `$(field.name)` over `$(field.source)` " *
                "has conflicting widths, levels or priors")
            old.name
        else
            name, body = _rk_ast_structured_block(definitions, field, taken, bindings)
            append!(stmts, body)
            blocks[key] = (; name, signature)
            name
        end
        callee = _rk_value_callee!(bindings, brm_structured_effect, taken)
        value = _rk_ast_fresh_name(string(term.options.id, "_", field.name), taken)
        push!(stmts, Expr(:(=), value, Expr(:call, callee, block,
            term.options.prepared_data, field_index)))
        push!(parts, value)
    end
    isempty(parts) && error("RK backend: structured term has no prepared fields")
    value = length(parts) == 1 ? only(parts) : Expr(:call, :.+, parts...)
    push!(stmts, Expr(:(=), term.options.id, value))
    stmts
end
