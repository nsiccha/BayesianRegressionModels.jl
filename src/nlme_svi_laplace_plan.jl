# Structural certificate for the initial, whole-window Gaussian collapse.
# No numerical derivative checks participate in eligibility. Unknown expression
# routes are refused rather than inferred affine from sampled evaluations.
_brm_svi_gaussian_error(message) = throw(ArgumentError("GaussianBlockLaplace: " * message))
_brm_svi_gaussian_require(ok, message) = ok || _brm_svi_gaussian_error(message)

_brm_svi_gaussian_literal(x) = x isa Real && isfinite(x)
# A prior argument's number: a literal, or a data scalar the program reads by name.
_brm_svi_gaussian_number(x, reg) = x isa Symbol ? get(reg.inputs, x, nothing) : x
_brm_svi_gaussian_args(args, reg) = map(arg -> _brm_svi_gaussian_number(arg, reg), args)
function _brm_svi_gaussian_fixed_prior(prior)
    # The random-effect emitter's missing override is its literal default
    # LogNormal(0, 1), not a model-dependent prior expression.
    prior === nothing && return true
    p = _brm_prepare_expr(prior)
    p isa _BRMPreparedExpr || return false
    all(_brm_svi_gaussian_literal, p.args) &&
        all(_brm_svi_gaussian_literal, values(p.kwargs))
end

function _brm_svi_gaussian_scale(scale, parameters)
    scale isa Real && return isfinite(scale) && scale > 0
    scale isa _BRMPreparedRef && return scale.axis in (:scalar, :whole) &&
        _brm_svi_gaussian_scale(scale.name, parameters)
    scale isa Symbol && return scale in parameters
    false
end

function _brm_svi_gaussian_kernel(kernel::_RKPreparedKernelAssignment, predictors, parameters)
    _brm_svi_gaussian_require(isempty(kernel.body) && kernel.collected === nothing &&
        isempty(kernel.outputs) && isempty(kernel.scan_observed),
        "UncertifiedPlateBody: only direct observation-only plates are certified")
    routes = Dict{Symbol,NamedTuple}()
    for o in kernel.observations
        _brm_svi_gaussian_require((o.callable === Normal || o.callable === StanBlocks.normal) &&
            length(o.arguments) == 2 && o.weight === nothing && o.modifier === nothing && isempty(o.bounds),
            "UnsupportedObservation: plate requires an unmodified Normal law")
        mean, scale = o.arguments
        k = mean isa Symbol ? findfirst(==(mean), kernel.params) : nothing
        _brm_svi_gaussian_require(k !== nothing,
            "UncertifiedMean: plate location must directly read a predictor input")
        input = kernel.inputs[k]
        _brm_svi_gaussian_require(input.source in predictors && input.kind in (:gather, :element),
            "UncertifiedMean: plate input is not a certified predictor")
        if input.kind === :gather
            rows = get(kernel.columns, input.rows, nothing)
            _brm_svi_gaussian_require(rows isa AbstractVector &&
                all(r -> r isa AbstractVector && all(i -> i isa Integer, r), rows),
                "UncertifiedIndex: predictor gather must use fixed integer rows")
        end
        scale_name = scale isa Symbol ? scale : scale isa _BRMPreparedRef ? scale.name : nothing
        _brm_svi_gaussian_require(!(scale_name in kernel.params) &&
            _brm_svi_gaussian_scale(scale, parameters),
            "UncertifiedScale: plate scale must be a positive literal or independent scalar")
        _brm_svi_gaussian_require(!haskey(routes, o.source), "AmbiguousObservation: repeated plate response")
        argument_routes = Tuple(_rk_kernel_cell_argument(kernel, name, arg)
            for (name, arg) in zip(o.argument_names, o.arguments))
        refs = Tuple(a.ref for a in argument_routes)
        cells = Dict(a.ref => a.ref for a in argument_routes if a.route in (:input, :plate))
        routes[o.source] = (; refs, cells, nested=o.nested, predictor=input.source, callable=o.callable)
    end
    routes
end
_brm_svi_gaussian_kernel(kernel, predictors, parameters) =
    _brm_svi_gaussian_error("UncertifiedAssignment: only direct observation plates are certified")

function _brm_svi_gaussian_observations(plan::_RKValuePlan, predictors, parameters)
    _brm_svi_gaussian_require(isempty(plan.completions), "UnsupportedCompletion: missing-value completion")
    routes = Dict{Symbol,NamedTuple}()
    for kernel in plan.assignments
        current = _brm_svi_gaussian_kernel(kernel, predictors, parameters)
        _brm_svi_gaussian_require(isempty(intersect(keys(routes), keys(current))),
            "AmbiguousObservation: multiple plate routes")
        merge!(routes, current)
    end
    used = Set{Symbol}()
    _brm_svi_gaussian_require(!isempty(plan.observations), "MissingObservation: no Gaussian observations")
    for o in plan.observations
        _brm_svi_gaussian_require(!(o isa _RKScanObservation), "UnsupportedObservation: scan")
        d = o.distribution
        _brm_svi_gaussian_require(d isa _BRMPreparedExpr &&
            (d.callable === Normal || d.callable === StanBlocks.normal) &&
            length(d.args) == 2 && isempty(d.kwargs) && o.modifier === nothing &&
            o.weight === nothing && o.missing_response === nothing,
            "UnsupportedObservation: requires an unmodified Normal law")
        if haskey(routes, o.name)
            route = routes[o.name]
            _brm_svi_gaussian_require(d.callable === route.callable &&
                all(a -> a isa _BRMPreparedRef, d.args) &&
                Tuple(a.name for a in d.args) == route.refs,
                "UncertifiedObservationRoute: lowered plate arguments differ")
            _brm_svi_gaussian_require(route.nested ?
                (o isa _RKNestedObservation && o.cells == route.cells) : !(o isa _RKNestedObservation),
                "UncertifiedObservationRoute: lowered plate indexing differs")
            push!(used, o.name)
        else
            mean, scale = d.args
            _brm_svi_gaussian_require(!(o isa _RKNestedObservation) &&
                mean isa _BRMPreparedRef && mean.name in predictors,
                "UncertifiedMean: location must directly read a certified predictor")
            _brm_svi_gaussian_require(_brm_svi_gaussian_scale(scale, parameters),
                "UncertifiedScale: scale depends on an uncertified expression")
        end
    end
    _brm_svi_gaussian_require(used == Set(keys(routes)), "UncertifiedObservationRoute: unused plate observation")
end
function _brm_svi_gaussian_observations(plan::_RKStructuralPlan, predictors, parameters)
    _brm_svi_gaussian_require(!isempty(plan.responses), "MissingObservation: no Gaussian observations")
    for o in plan.responses
        _brm_svi_gaussian_require(o.family === :gaussian && o.link === :identity &&
            o.predictor in predictors && o.scale_predictor === nothing && o.weights === nothing &&
            o.evidence.kind === :none && o.mi_jobs === nothing && o.interval === nothing,
            "UnsupportedObservation: requires direct unmodified Gaussian regression")
        _brm_svi_gaussian_require(_brm_svi_gaussian_scale(o.scale, parameters),
            "UncertifiedScale: scale depends on an uncertified expression")
    end
end
_brm_svi_gaussian_observations(plan, predictors, parameters) =
    _brm_svi_gaussian_error("UnsupportedPlan: held-out or unknown observation plan")

"""Certify an affine Gaussian block without evaluating a density or its derivatives.

`latent`, `globals`, `field`, and `unit_blocks` use full sampler indices.
`prior_mean` and diagonal `prior_precision` are aligned with `latent`.
The certificate covers initial whole-window inference, not streaming carry.
"""
function _brm_svi_gaussian_certificate(backend::RKBRMI, view::BRMNLMEView)
    plan = backend.plan
    _brm_svi_gaussian_require(plan isa Union{_RKValuePlan,_RKStructuralPlan}, "UnsupportedPlan: expected initial regression plan")
    reg = _brm_nlme_regression(plan)
    # Frozen fitting twins are bound data, emitted without any sampler
    # expression. HSGP axes/coordinates/group levels need these even in the
    # direct affine model. Other derived-expression routes remain uncertified.
    fit_inputs=all(reg.derived) do d
        Meta.isexpr(d.expression,:_rk_fit_input,1) &&
            only(d.expression.args) isa AbstractVector &&
            haskey(reg.columns,d.name) && isequal(reg.columns[d.name],only(d.expression.args))
    end
    _brm_svi_gaussian_require(isempty(reg.assignments) && fit_inputs &&
        isempty(reg.r2d2_priors) && isempty(reg.horseshoe_priors) && isempty(reg.vector_parameters),
        "UncertifiedRegression: only bound fit inputs may be derived; structured priors are unsupported")
    parameters = Set{Symbol}()
    for p in reg.parameters
        bounds = p.support_override
        _brm_svi_gaussian_require(all(_brm_svi_gaussian_literal, _brm_svi_gaussian_args(p.args, reg)) &&
            (bounds === nothing || bounds isa Symbol || (bounds isa Tuple &&
                all(x -> x isa Real, _brm_svi_gaussian_args(Base.tail(bounds), reg)))),
            "UncertifiedPrior: scalar prior must have literal arguments")
        push!(parameters, p.name)
    end
    predictors = Set(p.name for p in reg.predictors)
    for p in reg.predictors
        _brm_svi_gaussian_require(p.link === :identity, "NonAffinePredictor: predictor link must be identity")
        for t in p.terms
            _brm_svi_gaussian_term(t, p.name, reg)
        end
    end
    for b in reg.ranef_buckets
        _brm_svi_gaussian_require(b.dist === nothing && b.decomposition === nothing &&
            b.level_cor === nothing && b.grouping.form === :plain && b.grouping.by === nothing &&
            all(_brm_svi_gaussian_fixed_prior, b.sd_priors),
            "UnsupportedUnitPrior: requires Gaussian plain groups with independent fixed-argument scale priors")
        for margin in b.margins
            z = margin.z
            _brm_svi_gaussian_require(z.kind === :ones || (z.kind === :column &&
                haskey(reg.columns, z.column) && all(_brm_svi_gaussian_literal, reg.columns[z.column])),
                "UncertifiedUnitDesign: requires fixed intercept or continuous column")
        end
    end
    _brm_svi_gaussian_observations(plan, predictors, parameters)
    records = _rk_coordinate_records(plan)
    prior = Dict{Int,Tuple{Float64,Float64}}()
    field = Int[]
    for r in records
        _brm_svi_gaussian_record!(prior, field, r, reg, view)
    end
    units = sort!(unique(reduce(vcat, view.subject_coordinates; init=Int[])))
    _brm_svi_gaussian_require(all(i -> haskey(prior, i), units), "UncertifiedUnitPrior: uncovered unit coordinates")
    _brm_svi_gaussian_require(!isempty(field), "MissingSharedField: requires a coordinate-multigroup HSGP")
    latent = sort!(collect(keys(prior)))
    globals = setdiff(collect(eachindex(view.coordinates)), latent)
    (; globals, latent, field=sort!(field), prior_mean=[prior[i][1] for i in latent],
        prior_precision=[prior[i][2] for i in latent], unit_blocks=deepcopy(view.subject_coordinates),
        scope=(; likelihood=:affine_gaussian, integration=:joint_conditional_gaussian,
            window=:whole, streaming=false, spectral_bound=:prior_precision))
end

function _brm_svi_gaussian_term(t, predictor, reg)
    if t.kind in (:intercept, :continuous)
        nrows = t.kind === :intercept ? reg.n_obs :
            length(get(reg.columns, only(t.columns), ()))
        _brm_svi_gaussian_require(_brm_nlme_population_column(reg, predictor, t.addressee, nrows) !== nothing,
            "UncertifiedPopulationDesign: requires a fixed scalar design column")
    elseif t.kind === :ranef_gather
        _brm_svi_gaussian_require(any(b -> b.group === t.options.bucket_group && b.id === t.options.bucket_id,
            reg.ranef_buckets), "UncertifiedUnitDesign: unmatched random-effect gather")
    elseif t.kind === :hsgp
        o = t.options
        cor = get(o, :level_cor, nothing)
        _brm_svi_gaussian_require(cor isa _BRMMultigroupCoordinates && cor.method === :hsgp &&
            cor.lambda isa Real && isfinite(cor.lambda) && cor.lambda > 0 && isempty(cor.positions) &&
            all(isfinite, cor.coordinates) && !get(o, :latent, true) && isempty(get(o, :hyper_plans, ())) &&
            get(o, :anchor, nothing) === nothing && get(o, :mask, nothing) === nothing &&
            get(o, :orthogonal, nothing) === nothing && get(o, :cov, :exp_quad) === :exp_quad &&
            all(c -> haskey(reg.columns, c) && all(_brm_svi_gaussian_literal, reg.columns[c]), t.columns) &&
            _brm_svi_gaussian_fixed_prior(o.rho_prior) && _brm_svi_gaussian_fixed_prior(o.sigma_prior),
            "UnsupportedSharedField: requires fixed-coordinate multigroup HSGP with scalar hyperpriors")
    else
        _brm_svi_gaussian_error("UncertifiedTerm: $(t.kind)")
    end
end

function _brm_svi_gaussian_record!(prior, field, r, reg, view)
    if r.kind === :population
        ps = filter(p -> p.predictor === r.predictor && p.addressee === r.coefficient, reg.population_priors)
        args = length(ps) == 1 ? _brm_svi_gaussian_args(only(ps).args, reg) : ()
        _brm_svi_gaussian_require(length(ps) == 1 && only(ps).family === :Normal &&
            length(args) == 2 && all(_brm_svi_gaussian_literal, args) && args[2] > 0,
            "UnsupportedPopulationPrior: requires an unbounded literal Normal prior")
        indices = findall(n -> _brm_nlme_owned(n, r.declaration), view.coordinates)
        _brm_svi_gaussian_require(length(indices) == 1, "UnsupportedPopulationPrior: nonscalar coefficient")
        mu, sd = args
        precision = inv(Float64(sd)^2)
        _brm_svi_gaussian_require(isfinite(precision) && precision > 0,
            "UnsupportedPopulationPrior: precision is not representable as positive finite Float64")
        prior[only(indices)] = (Float64(mu), precision)
    elseif r.kind === :ranef
        for i in findall(n -> _brm_nlme_owned(n, r.z), view.coordinates)
            prior[i] = (0., 1.)
        end
    elseif r.kind === :hsgp
        _brm_svi_gaussian_require(r.grouped && r.level_correlation && r.locals !== nothing,
            "UnsupportedSharedField: field is not shared across groups")
        weights = Symbol(r.declaration, ".", r.locals.weights)
        indices = findall(n -> _brm_nlme_owned(n, weights), view.coordinates)
        _brm_svi_gaussian_require(!isempty(indices), "MissingSharedField: standardized weights not found")
        append!(field, indices)
        for i in indices
            prior[i] = (0., 1.)
        end
    elseif r.kind !== :scalar
        _brm_svi_gaussian_error("UnsupportedCoordinateBlock: $(r.kind)")
    end
end
