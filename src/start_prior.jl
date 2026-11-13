# Deterministic initialization from scalar prior medians. This is deliberately
# separate from the data-based/Laplace start: no model partition or fit is built.
# A recorded law as a Distributions.jl distribution. Names resolve through
# `value`: another declaration's own start, or a bound scalar. `nothing` when
# the law has no median (`Flat()`) or an argument has no value.
_brm_prior_law(x::Real, _value) = Float64(x)
_brm_prior_law(x::Symbol, value) = value(x)
function _brm_prior_law(x::Expr, value)
    # The DSL broadcasts a prior's inner calls (`log(0.2)` arrives as `log.(0.2)`,
    # `a * b` as `a .* b`), which mean the same on the scalars read here.
    if x.head === :. && length(x.args) == 2 && x.args[1] isa Symbol &&
            x.args[2] isa Expr && x.args[2].head === :tuple
        call, inputs = x.args[1], x.args[2].args
    elseif x.head === :call && first(x.args) isa Symbol
        name = string(first(x.args))
        call = startswith(name, ".") && length(name) > 1 ? Symbol(name[2:end]) : first(x.args)
        inputs = x.args[2:end]
    else
        return nothing
    end
    args = Any[_brm_prior_law(a, value) for a in inputs]
    any(isnothing, args) && return nothing
    _brm_prior_law(Val(call), args...)
end
_brm_prior_law(_x, _value) = nothing

_brm_prior_law(::Val{:Flat}) = nothing
_brm_prior_law(::Val{:HalfNormal}, s::Real) = truncated(Normal(0, s), 0, Inf)
_brm_prior_law(::Val{:HalfCauchy}, s::Real) = truncated(Cauchy(0, s), 0, Inf)
# `restricted` keeps its family's kernel on the support, so its median within
# the bounds is the normalized truncation's.
_brm_prior_law(::Val{:restricted}, d::UnivariateDistribution, lo::Real, hi::Real) =
    truncated(d, lo, hi)
_brm_prior_law(::Val{:truncated}, d::UnivariateDistribution, lo::Real, hi::Real) =
    truncated(d, lo, hi)
# RKPPL's three-argument Student t is located and scaled.
_brm_prior_law(::Val{:StudentT}, nu::Real, mu::Real, s::Real) = mu + s * TDist(nu)
for op in (:+, :-, :*, :/, :^, :sqrt, :exp, :log, :abs)
    @eval _brm_prior_law(::Val{$(QuoteNode(op))}, args::Real...) = $op(args...)
end
# Any other law names its Distributions.jl family.
function _brm_prior_law(::Val{F}, args...) where {F}
    isdefined(Distributions, F) || return nothing
    family = getfield(Distributions, F)
    family isa Type && family <: UnivariateDistribution &&
        all(a -> a isa Real, args) || return nothing
    family(args...)
end

_brm_prior_median(law::UnivariateDistribution) = Float64(median(law))
_brm_prior_median(_law) = nothing

_brm_start_prior_requested(base) = base === :prior_median
function _brm_start_base_vector(base, n, what)
    base isa Symbol && base !== :prior_median && error(
        "brm_start: unknown base `$base`; use a vector or :prior_median")
    _brm_start_base(base === :prior_median ? zeros(n) : base, n, what)
end

# Evaluate a captured law's arguments against data and already chosen starts.
# This walks the model's captured calls; it never evals initializer addresses.
_brm_start_law(x, value, floor=nothing) = x
_brm_start_law(x::NamedColumn, value, floor=nothing) = value(name(x))
_brm_start_law(x::DataColumn, value, floor=nothing) = parent(x)
_brm_start_law(x::Symbol, value, floor=nothing) = value(x)
_brm_start_law(x::_BRMLengthScaleFloor, value, floor=nothing) = floor
function _brm_start_law(x::ExprColumn, value, floor=nothing)
    args = map(a -> _brm_start_law(a, value, floor), getargs(x))
    kwargs = map(a -> _brm_start_law(a, value, floor), getkwargs(x))
    any(isnothing, args) && return nothing
    any(isnothing, kwargs) && return nothing
    f = getf(x)
    # BRM also accepts Family(args...; lower, upper), a restricted kernel.
    # Distribution constructors themselves do not accept those keywords.
    if f isa Type && f <: UnivariateDistribution &&
            any(k -> k in (:lower, :upper), keys(kwargs))
        rest = (; (k => v for (k, v) in pairs(kwargs) if !(k in (:lower, :upper)))...)
        return truncated(f(args...; rest...), get(kwargs, :lower, -Inf), get(kwargs, :upper, Inf))
    end
    f(args...; kwargs...)
end
_brm_start_law(x::Expr, value, floor=nothing) = _brm_prior_law(x, value)

function _brm_start_evaluate_law(prior, value, spelling, floor=nothing)
    try
        _brm_start_law(prior, value, floor)
    catch err
        error("brm_start: cannot resolve the prior of `$spelling`: " *
            sprint(showerror, err) * "; supply an explicit initializer or a numeric base")
    end
end

function _brm_start_median(law, spelling; lower=-Inf, upper=Inf)
    law isa UnivariateDistribution || error(
        "brm_start: no univariate prior median for `$spelling`; supply an explicit initializer or a numeric base")
    bounded = isfinite(lower) || isfinite(upper) ? truncated(law, lower, upper) : law
    x = try
        Float64(median(bounded))
    catch err
        error("brm_start: cannot compute the prior median of `$spelling`: " *
            sprint(showerror, err) * "; supply an explicit initializer or a numeric base")
    end
    isfinite(x) && lower < x < upper && insupport(bounded, x) || error(
        "brm_start: `$spelling` has no finite interior prior median; supply an explicit initializer or a numeric base")
    x
end

# Ordinary scalar dependencies read explicit values first, then their own
# median, then deterministic assignments/data. No random draw is a fallback.
function _brm_start_prior_values(brmi, inits; constants=(;), inputs=Dict(),
        component_value=key -> nothing)
    # A constant is a declaration marker, not a callable distribution argument.
    # Restore the backend's fitted values before traversing ordinary priors.
    brmi, _ = _brm_resolve_constants(brmi; prefix="brm_start",
        fitted=(key, _source) -> get(constants, key, nothing))
    # Match backend preparation: derived column assignments (for example a
    # subset used as a CAR area or CDAR axis) are data too. Resolve them after
    # restoring fitted constants and before collecting the prior context.
    brmi, _ = _brm_resolve_data_assignments(brmi; prefix="brm_start")
    context = _brm_backend_context(brmi)
    program = _brm_prepare_program(brmi; context)
    operations = Dict(op.name => op for op in program.operations)
    cache = Dict{Symbol,Any}(k => v for (k, v) in pairs(inits) if !(v isa NamedTuple))
    pending = Set{Symbol}()
    function value(key)
        haskey(cache, key) && return cache[key]
        haskey(inputs, key) && return get!(inputs[key], cache, key)
        haskey(context.data, key) && return context.data[key]
        key === :Inf && return Inf
        key in pending && error("brm_start: cyclic prior dependency at `$key`")
        op = get(operations, key, nothing)
        op === nothing && return nothing
        op.role in (:parameter, :assignment, :predictor) || return nothing
        rhs = last(getargs(op.expression))
        component = op.role === :predictor && rhs isa ExprColumn && getf(rhs) isa BRMComponent
        (component || op.role in (:parameter, :assignment)) || return nothing
        push!(pending, key)
        result = component ? component_value(key) : _brm_start_evaluate_law(rhs, value, key)
        if op.role === :parameter
            result = _brm_start_median(result, key)
        end
        delete!(pending, key)
        cache[key] = result
    end
    (; context, program, value)
end

function _brm_start_term_prior(prepared, slot)
    role = Symbol(_brm_term_prior_label(Val(slot.class)))
    priors = _brm_term_prior_expressions(prepared)
    role === :length_scale && (role = :rho)
    role === :sd && haskey(priors, :sigma) && (role = :sigma)
    prior = get(priors, role, nothing)
    slot.component === nothing ? prior : prior[findfirst(==(slot.component), (:rr, :rn, :nr))]
end
# Both of a field's lengths share the length-scale class: its laws go by slot.
_brm_start_term_prior(prepared::_BRMPreparedTerm{typeof(gmrf)}, slot) =
    _brm_term_prior_expressions(prepared)[slot.name]

# An authored HSGP hyper prior is the log-scale intercept's law. The
# unconfigured intercept is standard normal, not the shared scale's lognormal.
function _brm_start_hyper_prior(plan, context)
    class = plan.hyper === :sd ? :term_sd : :term_length_scale
    spec = _brm_term_prior_spec(plan.term, plan.lp, context, class)
    spec === nothing ? _brmd_default_normal() : spec.expression
end

function _brm_start_prior_term_targets(inventory, explicit, value)
    targets = NamedTuple[]
    for r in inventory, slot in _brm_term_prior_slots(getf(r.term))
        slot.support === :simplex && continue # retain the valid neutral simplex
        haskey(r, :bindings) && get(r.bindings, slot.name, nothing) === nothing && continue
        any(t -> t.predictor === r.predictor && t.slot.name === slot.name &&
            _brm_term_full_key(t.term) === _brm_term_full_key(r.term), explicit) && continue
        prior = _brm_start_term_prior(r.prepared, slot)
        # ICAR has no mixing parameter.
        prior === nothing && continue
        # A known or derived parameter is no coordinate to start.
        _brm_is_assigned_term_value(prior) && continue
        spec = (; class=slot.class, term=_brm_prepared_term_key(r.term),
            predictor=r.predictor, component=slot.component)
        spelling = _brm_term_prior_spelling(spec)
        floor = get(r, :floor, get(r.prepared.state, :rho_lower, 0.0))
        # Only the default HSGP declaration retains an implicit floor; an
        # explicitly configured law replaces it unless it names the marker.
        bound = get(r, :lower, get(r.prepared.state, :rho_lower, 0.0))
        lowers = slot.class === :term_length_scale ? (bound isa Real ? [bound] : bound) :
            [slot.support === :positive || slot.support === :unit_interval ? 0.0 : -Inf]
        uppers = fill(slot.support === :unit_interval ? 1.0 : Inf, length(lowers))
        if getf(r.term) === interval_censored
            lowers, uppers = r.prepared.state.x_lower, r.prepared.state.x_upper
        end
        xs = map(eachindex(lowers)) do j
            fj = floor isa Real ? floor : floor[j]
            law = _brm_start_evaluate_law(prior, value, spelling, fj)
            _brm_start_median(law, spelling; lower=lowers[j], upper=uppers[j])
        end
        start = length(xs) == 1 ? only(xs) : xs
        push!(targets, (; kind=:term, predictor=r.predictor, term=r.term, slot, spelling, value=start))
    end
    targets
end

function _brm_sb_start_term_inventory(d, context)
    inventory = NamedTuple[]
    for lp in linear_predictors(d.plan.parent), terms in values(_brm_term_address_map(d.plan.parent, lp.name)), term in terms
        prepared = _brm_prepare_term(term, lp.name, context)
        prepared === nothing && continue
        floor = get(prepared.state, :rho_lower, 0.0)
        lower = floor
        if getf(term) === hsgp
            entry = _brm_term_entry(d, lp.name,
                e -> _brm_term_full_key(e.value) === _brm_term_full_key(term), "term")
            owner = _brm_term_owner_output(d, lp.name, entry).declaration
            raw = get(owner.keywords, :rho_lower, floor)
            floor = raw isa Symbol ? d.plan.data[raw] : raw
            spec = _brm_term_prior_spec(term, lp.name, context, :term_length_scale)
            lower = spec === nothing ? floor : _brm_zero_rho_lower(floor)
        end
        hyper = _brm_term_hyper_plans(d.plan, lp.name, term)
        bindings = (; (slot.name => (any(p -> p.hyper ===
            Symbol(_brm_term_prior_label(Val(slot.class))), hyper) ? nothing : true)
            for slot in _brm_term_prior_slots(getf(term)))...)
        push!(inventory, (; predictor=lp.name, term, prepared, floor, lower, bindings))
    end
    inventory
end

# Writes of scalar/broadcast laws on RK's emitted declarations. Read laws from
# the same records the existing NLME median start consumes.
function _brm_rk_start_prior_writes(rk, records, values, explicit, inits, term_targets; env=nothing)
    chosen = Set((d, i) for (d, i, _) in explicit)
    covered(d, i) = (d, ()) in chosen || (d, i) in chosen
    writes = Tuple{Symbol,Tuple,Any}[]
    fitted = _rk_fitted_state(rk)
    env === nothing && (env = _brm_start_prior_values(parent(rk), inits;
        constants=fitted === nothing ? (;) : fitted.constants))
    function add(d, i, prior, spelling; lower=-Inf, upper=Inf)
        covered(d, i) && return
        x = _brm_start_median(_brm_start_evaluate_law(prior, env.value, spelling), spelling; lower, upper)
        push!(writes, (d, i, x))
    end
    _brm_rk_start_prior_records!(add, records, values; scalars=true)
    for p in _sb_collect_hyper_plans(parent(rk))
        term = _brm_rk_start_term_record((; predictor=p.lp, term=p.term, spelling=p.spelling), records)
        record = only(r for r in records if r.kind === :hsgp && r.declaration === term.declaration)
        if p.intercept
            declaration = Symbol(record.declaration, ".", record.locals.scales[p.hyper])
            add(declaration, (), _brm_start_hyper_prior(p, env.context), "log($(p.spelling)) intercept")
        end
        if !isempty(p.ranefs)
            declaration = Symbol(record.declaration, ".", record.locals.hypers[p.hyper].scale)
            add(declaration, (), _brmd_default_normal(), "log($(p.spelling)) group SD"; lower=0.0)
        end
    end
    inventory = [r for r in records if r.kind === :term_start]
    targets = _brm_start_prior_term_targets(inventory, term_targets, env.value)
    _brm_rk_start_terms!(writes, targets, records, values)
    writes
end

# The medians of population coefficients, `|ID|` scales and (with `scalars`)
# named scalar parameters among RK coordinate records; `add(declaration,
# index, law, spelling; lower)` writes one unless an explicit start covers it.
# Top-level records and a component call's scoped records share it.
function _brm_rk_start_prior_records!(add, records, values; scalars)
    for r in records
        if r.kind === :population
            add(r.declaration, get(r, :index, ()), r.prior, "effect($(r.predictor), $(r.coefficient))")
        elseif r.kind === :class_population
            # One coefficient of a class block, for one non-reference class.
            add(r.declaration, r.index, r.prior,
                "effect($(r.response), $(r.coefficient)) in class $(r.class)")
        elseif r.kind === :population_block
            for (i, label) in enumerate(r.labels)
                add(r.declaration, (i,), _brm_nlme_block_law(r.prior, i), "effect($(r.predictor), $label)")
            end
        elseif r.kind === :scalar
            scalars && add(r.declaration, (), r.prior, r.declaration)
        elseif r.kind in (:ranef, :stratified_ranef)
            laws = get(r, :scale_priors, nothing)
            laws === nothing && continue
            for (k, prior) in enumerate(laws)
                r.scales !== nothing && r.scales[k] === nothing && continue
                declaration = r.scales === nothing ? r.scale : r.scales[k]
                current = _brm_rk_declaration_value(values, declaration)
                if r.kind === :stratified_ranef
                    for s in axes(current, 1)
                        add(declaration, r.scales === nothing ? (s, k) : (s,), prior, "sd margin $k of $(r.id)")
                    end
                else
                    add(declaration, current isa Real ? () : (k,), prior, "sd margin $k of $(r.id)")
                end
            end
        end
    end
    nothing
end

function _brm_sb_start_prior_writes(sb, d, names, explicit, inits, term_targets; env=nothing)
    chosen = Set(first(w) for w in explicit)
    writes = Tuple{Int,Float64}[]
    env === nothing && (env = _brm_start_prior_values(parent(sb), inits; constants=d.plan.data))
    bounds = _brm_sb_start_bounds(d, names)
    function add(indices, prior, spelling; lower=-Inf, upper=Inf)
        remaining = filter(i -> !(i in chosen), indices)
        isempty(remaining) && return
        law = _brm_start_evaluate_law(prior, env.value, spelling)
        law isa MultivariateDistribution && return
        for i in remaining
            lo, hi = get(bounds, i, (-Inf, Inf))
            x = _brm_start_median(law, spelling; lower=max(lower, lo), upper=min(upper, hi))
            push!(writes, (i, x))
        end
    end
    for op in env.program.operations
        op.role === :parameter || continue
        outputs = brm_outputs(d; logical=op.name)
        isempty(outputs) && continue
        output = brm_output(d, op.name)
        # Joint constrained objects keep their neutral transform coordinates.
        output.type in (:simplex, :ordered, :positive_ordered, :cholesky_factor_corr,
            :cholesky_factor_cov, :corr_matrix, :cov_matrix) && continue
        add(_brm_element_coordinates(output, names), last(getargs(op.expression)), op.name)
    end
    _brm_sb_start_prior_population!(add, sb, d, names)
    _brm_sb_start_prior_ranefs!(writes, d, names, chosen, env.value)
    for p in _sb_collect_hyper_plans(parent(sb))
        entry = _brm_term_entry(d, p.lp,
            e -> _brm_term_full_key(e.value) === _brm_term_full_key(p.term), p.spelling)
        coordinates(role) = _brm_element_coordinates(
            _brm_term_parameter_output(d, p.lp, entry, Symbol(p.hyper, :_, role)), names)
        p.intercept && add(coordinates(:intercept), _brm_start_hyper_prior(p, env.context),
            "log($(p.spelling)) intercept")
        isempty(p.ranefs) || add(coordinates(:ranef_sd), _brmd_default_normal(),
            "log($(p.spelling)) group SD"; lower=0.0)
    end
    inventory = _brm_sb_start_term_inventory(d, env.context)
    targets = _brm_start_prior_term_targets(inventory, term_targets, env.value)
    _brm_sb_start_terms!(writes, targets, d, names)
    writes
end

# The medians of an SBBRMI's ordinary population coefficients, numeric and
# categorical, through `add(indices, law, spelling)`. `names` may be a component
# call's local names (`_brm_sb_component_local_names`).
function _brm_sb_start_prior_population!(add, sb, d, names)
    overrides = _sb_effect_prior_overrides(parent(sb); frozen_preproc=d.plan.preproc)
    for entry in _brm_population_effect_entries(parent(sb))
        scheme = get(get(d.plan.bindings, entry.block, (;)), :prior_scheme, (; kind=:ordinary))
        scheme.kind === :ordinary || continue # joint shrinkage blocks stay neutral
        settings = _sb_pop_effect_overrides(overrides, entry.logical)
        outputs = [o for o in d.outputs if o.kind === :parameter && o.declaration !== nothing &&
            o.declaration.target === entry.block && o.labels !== nothing]
        for output in outputs
            indices = _brm_element_coordinates(output, names)
            for (j, label) in enumerate(output.labels)
                prior = settings === nothing ? nothing : settings[j]
                add([indices[j]], something(prior, _brmd_default_normal()), "effect($(entry.logical), $label)")
            end
        end
        cats = _sb_cat_effect_overrides(overrides, entry.logical)
        for cat in _brm_categorical_effect_entries(d, entry.logical, entry.link)
            q = brm_categorical_effect_coordinates(d, entry.logical, names; coefficient=cat.address)
            prior = get(cats, cat.emitted, nothing)
            for (j, i) in enumerate(q.coordinates)
                p = prior isa AbstractVector ? prior[j] : prior
                add([i], something(p, _brmd_default_normal()), "effect($(entry.logical), $(cat.address))")
            end
        end
    end
    nothing
end

# Public StanBlocks reflection supplies evaluated bounds, including vector
# bounds. Parameter-dependent bounds are already part of the evaluated law.
function _brm_sb_start_bounds(d, names)
    published = Dict(o.name => o.constraint_values for o in d.stan.outputs)
    result = Dict{Int,Tuple{Float64,Float64}}()
    for output in d.outputs
        output.kind === :parameter || continue
        bounds = get(published, output.name, (;))
        lower, upper = get(bounds, :lower, -Inf), get(bounds, :upper, Inf)
        lower === nothing && (lower = -Inf)
        upper === nothing && (upper = Inf)
        for (j, i) in enumerate(_brm_element_coordinates(output, names))
            result[i] = (lower isa Real ? lower : lower[j], upper isa Real ? upper : upper[j])
        end
    end
    result
end

function _brm_sb_start_prior_ranefs!(writes, d, names, chosen, value)
    for group in _brmd_ranef_metadata(d)
        b = group.block
        if b.family === :brm_total
            k = group.total.column
            output = only(o for o in d.outputs if o.name === group.total.binding.total.scales)
            i = _brm_element_coordinates(output, names)[k]
            i in chosen && continue
            prior = something(group.total.binding.scale_priors[k], _brmd_default_normal())
            law = _brm_start_evaluate_law(prior, value, "sd margin $k of $(b.id)")
            push!(writes, (i, _brm_start_median(law, "sd margin $k of $(b.id)"; lower=0.0)))
            continue
        end
        b.generated && continue
        spec = get(_RANEF_FAMILIES, b.family, nothing)
        spec === nothing && continue
        cfg = group.configuration
        tau = get(spec, :tau, nothing)
        # The ordinary single-margin block's log-scale has N(0,1).
        tau === nothing && b.by === nothing && continue
        record = get(d.plan.bindings, b.binding, (;))
        known = get(record, :known_scales, nothing)
        free = known === nothing ? collect(1:b.n_terms) : findall(isnothing, collect(known))
        carrier = known === nothing ? nothing : _sb_known_scale_carrier(b.binding, known)
        outputs = [o for o in d.outputs if o.kind === :parameter && o.component === :scale &&
            o.declaration !== nothing && (o.declaration.target === b.binding || o.name === carrier)]
        for output in outputs
            indices = _brm_element_coordinates(output, names)
            for (j, i) in enumerate(indices)
                i in chosen && continue
                # Stan's stratified carrier is margins × strata; a free
                # known-value carrier retains the surviving margin order.
                k = free[mod1(j, length(free))]
                prior = cfg === nothing ? nothing : cfg.sd_prior[k]
                prior isa _BRMKnownScale && continue
                law = _brm_start_evaluate_law(something(prior, _brmd_default_normal()), value,
                    "sd margin $k of $(b.id)")
                push!(writes, (i, _brm_start_median(law, "sd margin $k of $(b.id)"; lower=0.0)))
            end
        end
    end
    writes
end
