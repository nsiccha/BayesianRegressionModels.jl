# General prepared group fields: BRM owns prior declarations and block algebra;
# the consumer's existing native effect owns how each block enters the term.
# One term's fitted statistical inputs cross through an ordinary data port.
# Capturing its arrays in a callable would hide them from the AD interface.
function brm_structured_effect(block, prepared_inputs, field_index)
    prepared = only(prepared_inputs)
    _brm_native_structured_effect(prepared, block, prepared.state.fields[field_index])
end

# The statistical reader applies each group's weights to the same fitted
# Hilbert basis. Shared hypers remain scalars; authored hyper predictors are
# vectors with one value per declared group, and rho floors act per group.
function brm_hsgp_grouped(PHI, omega2, z, group_index, rho, sigma)
    values = similar(z, size(PHI,1))
    if rho isa Number && sigma isa Number
        spectrum = brm_hsgp_sqrt_spd(omega2, sigma, rho)
        for i in axes(PHI,1)
            value = zero(eltype(z))
            for j in axes(PHI,2)
                value += PHI[i,j] * z[group_index[i],j] * spectrum[j]
            end
            values[i] = value
        end
        return values
    end
    spectra = similar(z)
    for g in axes(z,1)
        spectrum = brm_hsgp_sqrt_spd(omega2,
            sigma isa Number ? sigma : sigma[g], rho isa Number ? rho : rho[g])
        for j in axes(z,2)
            spectra[g,j] = spectrum[j]
        end
    end
    for i in axes(PHI,1)
        value = zero(eltype(z))
        for j in axes(PHI,2)
            value += PHI[i,j] * z[group_index[i],j] * spectra[group_index[i],j]
        end
        values[i] = value
    end
    values
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

function _rk_ast_hsgp_hyper!(block, options, hyper, prior, floor, G)
    stated = hyper === :length_scale ? options.rho_stated : options.sigma_stated
    plans = options.hyper_plans
    position = findfirst(p -> p.hyper === hyper, plans)
    stem = hyper === :length_scale ? "rho" : "sigma"
    if position === nothing
        value = _rk_block_local!(block, stem)
        hyper === :length_scale && options.rho_truncated &&
            (prior = Expr(:call, :restricted, prior, floor, Inf))
        push!(block.statements, Expr(:call, :~, value, prior))
        return value
    end
    plan = plans[position]
    parts = Any[]
    if plan.intercept
        beta = _rk_block_local!(block, string(stem, "_Intercept"))
        push!(block.statements, Expr(:call, :~, beta,
            stated ? prior : Expr(:call, :Normal, 0, 1)))
        push!(parts, beta)
    elseif stated
        error("RK backend: HSGP hyper prior needs an intercept in its authored predictor")
    end
    if !isempty(plan.ranefs)
        sd = _rk_block_local!(block, string(stem, "_sd"))
        z = _rk_block_local!(block, string(stem, "_z"))
        push!(block.statements, Expr(:call, :~, sd,
            Expr(:call, :restricted, Expr(:call, :Normal, 0, 1), 0.0, Inf)))
        push!(block.statements, Expr(:call, :.~, Expr(:ref, z, Expr(:call, :(:), 1, G)),
            _rk_ast_dotted(:Normal, 0, 1)))
        push!(parts, Expr(:call, :.*, sd, z))
    end
    eta = length(parts) == 1 ? only(parts) : Expr(:call, :.+, parts...)
    value = _rk_block_local!(block, string(stem, "_value"))
    expression = _rk_ast_dotted(:exp, eta)
    hyper === :length_scale && (expression = _rk_ast_dotted(:max, expression, floor))
    push!(block.statements, Expr(:(=), value, expression))
    value
end

# Grouped or hyper-predicted HSGP: one block allocating the hyperparameters,
# their group-level effects and the basis weights, returning the summand.
function _rk_ast_hsgp_grouped(definitions, term, inputs, floors, extent, taken, bindings)
    options = term.options
    G = get(options, :n_groups, 1)
    rho_prior = _rk_ast_hsgp_hyper_prior(options, :length_scale, bindings, taken)
    # A floor named in the length-scale prior is read there (decision 1fk9b29);
    # only a truncation or a hyper-predictor reads it as a block argument.
    floor_named = get(options, :rho_floor_named, false)
    floor_named && (rho_prior = _rk_ast_replace_floor_marker(rho_prior, floors))
    sigma_prior = _rk_ast_hsgp_hyper_prior(options, :sd, bindings, taken)
    grouped = haskey(options, :group_index)
    block = _rk_block_body(rho_prior, sigma_prior)
    arguments = Pair{Symbol,Any}[formal => _rk_block_argument!(block, formal, value)
        for (formal, value) in inputs]
    floor = floors === nothing || floor_named ? nothing :
        _rk_block_argument!(block, :floor, floors)
    group_index = grouped ? _rk_block_argument!(block, :group_index, options.group_index) :
        nothing
    rho = _rk_ast_hsgp_hyper!(block, options, :length_scale, rho_prior, floor, G)
    sigma = _rk_ast_hsgp_hyper!(block, options, :sd, sigma_prior, floor, G)
    z = _rk_block_local!(block, :z)
    # A periodic extent reads its prepared basis argument.
    extent = _rk_ast_hsgp_substitute(extent, Dict(arguments))
    level_cor = get(options, :level_cor, nothing)
    weights = grouped ? (Expr(:call, :(:), 1, _rk_ast_hsgp_weight_rows(level_cor, G)),
                         extent) : (extent,)
    push!(block.statements, Expr(:call, :.~, Expr(:ref, z, weights...),
        _rk_ast_dotted(:Normal, 0, 1)))
    weights_value, multigroup = _rk_ast_hsgp_level_weights!(block, z, level_cor, options)
    value = _rk_block_local!(block, :value)
    push!(block.statements, Expr(:(=), value, _rk_ast_hsgp_value_graph!(definitions,
        term, taken, arguments, sigma, rho, weights_value; group_index, multigroup)))
    _rk_ast_block_call!(definitions, taken, "brm_grouped_hsgp", block, value)
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
function _rk_ast_hsgp_level_weights!(block, z, lc::_BRMMultigroupCorrelation, _options)
    L = _rk_block_argument!(block, :L_level, _rk_ast_level_factor_value(lc.base))
    tau = _rk_block_argument!(block, :tau_level, lc.tau)
    (z, (; weights=:(brm_hsgp_multigroup_weights(z, omega2, rho, L_level, tau_level)),
           formals=[:L_level, :tau_level], values=Any[L, tau]))
end
# The coordinates are fit-role data columns; the graph assembles them and
# derives the inducing grid (`brm_multigroup_coordinate_grid`) in place.
function _rk_ast_hsgp_level_weights!(block, z, lc::_BRMMultigroupCoordinates, options)
    columns = Any[_rk_block_argument!(block, Symbol(:coordinate_, j), column)
                  for (j, column) in enumerate(options.coordinate_columns)]
    positions = Any[_rk_block_argument!(block, Symbol(:position_, j), column)
                    for (j, column) in enumerate(get(options, :position_columns, ()))]
    lambda = isnothing(lc.lambda) ? _rk_multigroup_own_lambda!(block, lc) :
        _rk_block_argument!(block, :lambda_level, lc.lambda)
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
                    values=Any[columns; positions; lambda]))
    end
    method = _brm_multigroup_method_code(lc.method)
    weights = quote
        coordinates_level = hcat($(formals...))
        grid_level = brm_multigroup_coordinate_grid(coordinates_level, $(lc.k), $method, $(lc.c))
        brm_hsgp_multigroup_coordinate_weights(z, omega2, rho, coordinates_level,
            grid_level, lambda_level, $method)
    end
    (z, (; weights, formals=[formals; :lambda_level], values=Any[columns; lambda]))
end

# `multigroup((x, y); k)`: the term's own spatial scale, with SBBRMI's prior
# (`_sb_multigroup_own_lambda`, decision `100gkpq`).
function _rk_multigroup_own_lambda!(block, lc)
    (; location, lower) = _brm_multigroup_default_lambda(lc)
    lambda = _rk_block_local!(block, :lambda)
    push!(block.statements, Expr(:call, :~, lambda,
        Expr(:call, :restricted, Expr(:call, :LogNormal, location, 1.0), lower, Inf)))
    lambda
end

# The rows of a grouped HSGP's standardized weights: one per level, or one per
# grid point, basis function or inducing position for
# `multigroup(coordinates, lambda; ...)`.
_rk_ast_hsgp_weight_rows(_level_cor, n_groups) = n_groups
_rk_ast_hsgp_weight_rows(lc::_BRMMultigroupCoordinates, _n_groups) = lc.n_points
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
